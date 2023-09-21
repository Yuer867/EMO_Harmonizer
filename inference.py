import os
import sys
import time
import yaml
import random
import argparse
import numpy as np
from itertools import chain
from collections import defaultdict
from midi2audio import FluidSynth
import torch

from dataloader import REMISkylineToMidiTransformerDataset
from model.music_performer import MusicPerformer
from convert2midi import event_to_midi, TempoEvent
from representations.convert_key import degree2pitch, roman2majorDegree, roman2minorDegree, find_key_emopia
from utils import pickle_load

sys.path.append('./model')

temp, top_p = 1.1, 0.99

MAJOR_KEY = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
MINOR_KEY = np.array(['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'])


###############################################
# sampling utilities
###############################################
def construct_inadmissible_set(tempo_val, event2idx, tolerance=20):
    inadmissibles = []

    for k, i in event2idx.items():
        if 'Tempo' in k and 'Conti' not in k and abs(int(k.split('_')[-1]) - tempo_val) > tolerance:
            inadmissibles.append(i)

    print(inadmissibles)

    return np.array(inadmissibles)


def temperature(logits, temperature, inadmissibles=12):
    if inadmissibles is not None:
        logits[inadmissibles] -= np.inf

    try:
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        assert np.count_nonzero(np.isnan(probs)) == 0
    except:
        print('overflow detected, use 128-bit')
        logits = logits.astype(np.float128)
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        probs = probs.astype(float)
    return probs


def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3]  # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


##############################################
# data manipulation utilities
##############################################
def merge_tracks(melody_track, chord_track):
    events = ['Bar_None']

    melody_beat = defaultdict(list)
    if len(melody_track) > 2:
        note_seq = []
        beat = melody_track[2]
        melody_track = melody_track[3:]
        for p in range(len(melody_track)):
            if 'Beat' in melody_track[p]:
                melody_beat[beat] = note_seq
                note_seq = []
                beat = melody_track[p]
            else:
                note_seq.append(melody_track[p])
        melody_beat[beat] = note_seq

    chord_beat = defaultdict(list)
    if len(chord_track) > 2:
        chord_seq = []
        beat = chord_track[2]
        chord_track = chord_track[3:]
        for p in range(len(chord_track)):
            if 'Beat' in chord_track[p]:
                chord_beat[beat] = chord_seq
                chord_seq = []
                beat = chord_track[p]
            else:
                chord_seq.append(chord_track[p])
        chord_beat[beat] = chord_seq

    for b in range(16):
        beat = 'Beat_{}'.format(b)
        if beat in chord_beat or beat in melody_beat:
            events.append(beat)
            if beat in chord_beat:
                events.extend(chord_beat[beat])
            if beat in melody_beat:
                events.extend(melody_beat[beat])

    return events


def events2bars(key, events, relative_melody=False):
    if relative_melody:
        new_events = []
        keyname = key.split('_')[1]
        for evs in events:
            if 'Note_Octave' in evs:
                octave = int(evs.split('_')[2])
            elif 'Note_Degree' in evs:
                roman = evs.split('_')[2]
                pitch = degree2pitch(keyname, octave, roman)
                pitch = max(21, pitch)
                pitch = min(108, pitch)
                if pitch < 21 or pitch > 108:
                    raise ValueError('Pitch value must be in (21, 108), but gets {}'.format(pitch))
                new_events.append('Note_Pitch_{}'.format(pitch))
            elif 'Chord_' in evs:
                if 'None' in evs:
                    new_events.append(evs)
                else:
                    root, quality = evs.split('_')[1], evs.split('_')[2]
                    if keyname in MAJOR_KEY:
                        root = roman2majorDegree[root]
                    else:
                        root = roman2minorDegree[root]
                    new_events.append('Chord_{}_{}'.format(root, quality))
            else:
                new_events.append(evs)
        events = new_events

    melody_pos = np.where(np.array(events) == 'Track_Melody')[0].tolist()
    chord_pos = np.where(np.array(events) == 'Track_Chord')[0].tolist()
    assert len(melody_pos) == len(chord_pos)
    n_bars = len(melody_pos)
    melody_pos.append(len(events))

    lead_sheet_bars = []
    for b in range(n_bars):
        melody_track = events[melody_pos[b]: chord_pos[b]]
        chord_track = events[chord_pos[b]: melody_pos[b + 1]]
        lead_sheet_bars.append(merge_tracks(melody_track, chord_track))

    events = [key] + events[melody_pos[0]:melody_pos[-1]]
    return events, lead_sheet_bars


def word2event(word_seq, idx2event):
    return [idx2event[w] for w in word_seq]


def get_position_idx(event):
    return int(event.split('_')[-1])


def event_to_txt(events, output_event_path):
    f = open(output_event_path, 'w')
    print(*events, sep='\n', file=f)


def midi_to_wav(midi_path, output_path):
    sound_font_path = 'SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2'
    fs = FluidSynth(sound_font_path)
    fs.midi_to_audio(midi_path, output_path)


################################################
# main generation function
################################################
def generate_conditional(model, event2idx, idx2event, melody_events, generated, seg_inp,
                         max_events=10000, skip_check=False, max_bars=None,
                         temp=1.2, top_p=0.9, inadmissibles=None):
    generated = generated + [event2idx['Track_Melody']] + melody_events[0] + [event2idx['Track_Chord']]
    _seg_inp = [0 for _ in range(len(generated))]
    _seg_inp[:len(seg_inp)] = seg_inp
    _seg_inp[-1] = 1
    seg_inp = _seg_inp

    target_bars, generated_bars = len(melody_events), 0
    if max_bars is not None:
        target_bars = min(max_bars, target_bars)

    steps = 0
    time_st = time.time()
    cur_pos = 0
    failed_cnt = 0

    while generated_bars < target_bars:
        assert len(generated) == len(seg_inp)
        if len(generated) < max_dec_inp_len:
            dec_input = torch.tensor([generated]).long().to(next(model.parameters()).device)
            dec_seg_inp = torch.tensor([seg_inp]).long().to(next(model.parameters()).device)
        else:
            dec_input = torch.tensor([generated[-max_dec_inp_len:]]).long().to(next(model.parameters()).device)
            dec_seg_inp = torch.tensor([seg_inp[-max_dec_inp_len:]]).long().to(next(model.parameters()).device)

        # sampling
        logits = model(
            dec_input,
            seg_inp=dec_seg_inp,
            chord_inp=None,
            keep_last_only=True,
            attn_kwargs={'omit_feature_map_draw': steps > 0}
        )
        logits = (logits[0]).cpu().detach().numpy()
        probs = temperature(logits, temp, inadmissibles=inadmissibles)
        word = nucleus(probs, top_p)
        word_event = idx2event[word]

        if not skip_check:
            if 'Beat' in word_event:
                event_pos = get_position_idx(word_event)
                if not event_pos >= cur_pos:
                    failed_cnt += 1
                    print('[info] position not increasing, failed cnt:', failed_cnt)
                    if failed_cnt >= 256:
                        print('[FATAL] model stuck, exiting with generated events ...')
                        return generated
                    continue
                else:
                    cur_pos = event_pos
                    failed_cnt = 0

        if word_event == 'Track_Melody':
            steps += 1
            generated.append(word)
            seg_inp.append(0)
            generated_bars += 1
            print('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated)))

            if generated_bars < target_bars:
                generated.extend(melody_events[generated_bars])
                seg_inp.extend([0 for _ in range(len(melody_events[generated_bars]))])

                generated.append(event2idx['Track_Chord'])
                seg_inp.append(1)
                cur_pos = 0

            continue

        if word_event == 'PAD_None' or (word_event == 'EOS_None' and generated_bars < target_bars - 1):
            continue
        elif word_event == 'EOS_None' and generated_bars == target_bars - 1:
            print('[info] gotten eos')
            generated.append(word)
            break

        generated.append(word)
        seg_inp.append(1)
        steps += 1

        if len(generated) > max_events:
            print('[info] max events reached')
            break

    print('-- generated events:', len(generated))
    print('-- time elapsed  : {:.2f} secs'.format(time.time() - time_st))
    print('-- time per event: {:.2f} secs'.format((time.time() - time_st) / len(generated)))
    return generated[:-1]


def generate_key(model, event2idx, idx2event, emotion,
                 temp=1.2, top_p=0.9, inadmissibles=None):
    generated = [event2idx[emotion]]
    seg_inp = [2]
    dec_input = torch.tensor([generated]).long().to(next(model.parameters()).device)
    dec_seg_inp = torch.tensor([seg_inp]).long().to(next(model.parameters()).device)

    generate_key = True
    while generate_key:
        logits = model(
            dec_input,
            seg_inp=dec_seg_inp,
            chord_inp=None,
            keep_last_only=True,
            attn_kwargs={'omit_feature_map_draw': False}
        )
        logits = (logits[0]).cpu().detach().numpy()
        probs = temperature(logits, temp, inadmissibles=inadmissibles)
        word = nucleus(probs, top_p)
        word_event = idx2event[word]

        if 'Key' in word_event:
            generated.append(word)
            seg_inp.append(3)
            key = word_event
            print('[info] generated {}, #events = {}'.format(key, len(generated)))
            generate_key = False
        else:
            print('[info] generated key failed')

    return generated, seg_inp, key


if __name__ == '__main__':
    """
    REMI
    (1) absolute:                                       False, False, False, False, False
    (2) transpose:                                      True, False, False, False, False !!!
    (3) transpose + rule-based:                         True, False, False, True, False !!!
    functional
    (4) absolute melody + relative chord:               False, True, False, False, False
    (5) relative melody + relative chord:               False, True, True, False, False
    (6) relative melody + relative chord + rule-based:  False, True, True, True, False !!!
    (7) relative melody + relative chord + model-based: False, True, True, False, True !!!
    """
    # configuration
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--configuration',
                          choices=['config/hooktheory_pretrain.yaml', 'config/emopia_finetune.yaml'],
                          help='configurations of training', required=True)
    required.add_argument('-r', '--representation',
                          choices=['absolute', 'transpose', 'ablated', 'functional'],
                          help='representation for symbolic music', required=True)
    required.add_argument('-k', '--key_determine',
                          choices=['rule', 'model', 'none'],
                          help='how to determine keys (rule-based, model-based or remain unchanged)', required=True)
    parser.add_argument('-i', '--inference_params',
                        default='emo_harmonizer_ckpt_functional/best_params.pt',
                        help='inference parameters')
    parser.add_argument('-o', '--output_dir',
                        default='generation/emopia_functional',
                        help='output directory')
    parser.add_argument('-p', '--play_midi',
                        default=False,
                        help='play midi to audio using FluidSynth', action='store_true')
    args = parser.parse_args()

    train_conf_path = args.configuration
    train_conf = yaml.load(open(train_conf_path, 'r'), Loader=yaml.FullLoader)
    print(train_conf)

    representation = args.representation
    key_determine = args.key_determine
    if representation != 'functional' and key_determine == 'model':
        raise ValueError('only support model-based key determination when representation is functional')
    if representation == 'absolute' and key_determine == 'rule':
        raise ValueError('not support rule-based key determination when representation is absolute')

    if representation == 'absolute':
        transpose_to_C, relative_chord, relative_melody = False, False, False
    elif representation == 'transpose':
        transpose_to_C, relative_chord, relative_melody = True, False, False
    elif representation == 'ablated':
        transpose_to_C, relative_chord, relative_melody = False, True, False
    elif representation == 'functional':
        transpose_to_C, relative_chord, relative_melody = False, True, True
    else:
        raise ValueError("invalid representation {}, choose from [absolute, transpose, ablated, functional]"
                         .format(representation))

    if key_determine == 'rule':
        rule_based = True
        model_based = False
    elif key_determine == 'model':
        rule_based = False
        model_based = True
    else:
        rule_based = False
        model_based = False

    print('whether transpose_to_C: {}, whether relative_chord: {}, whether relative_melody: {}, \n'
          'whether enforce_key: {}, whether predict_key: {}'.
          format(transpose_to_C, relative_chord, relative_melody, rule_based, model_based))

    inference_param_path = args.inference_params
    out_dir = args.output_dir
    play_midi = args.play_midi

    n_pieces = 10
    emotion_events = ['Emotion_Positive', 'Emotion_Negative']
    samp_per_piece = 1
    max_bars = 128
    max_dec_inp_len = 1024

    # training configurations
    gpuid = train_conf['training']['gpuid']
    torch.cuda.set_device(gpuid)

    # dataloader configurations
    data_path = train_conf['data_loader']['data_path'].format(representation)
    vocab_path = train_conf['data_loader']['vocab_path'].format(representation)

    # model configurations
    model_conf = train_conf['model']
    if representation != 'functional':
        model_conf['max_len'] = 800
    if key_determine == 'model':
        model_conf['n_segment_types'] = 4
        predict_key = True
    else:
        predict_key = False

    # load dataset
    val_split = train_conf['data_loader']['val_split']
    dset = REMISkylineToMidiTransformerDataset(
        data_dir=data_path,
        vocab_file=vocab_path,
        model_dec_seqlen=model_conf['max_len'],
        pieces=pickle_load(val_split),
        pad_to_same=True,
        predict_key=predict_key
    )

    # load model
    model = MusicPerformer(
        dset.vocab_size, model_conf['n_layer'], model_conf['n_head'],
        model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
        use_segment_emb=model_conf['use_segemb'],
        n_segment_types=model_conf['n_segment_types'],
        favor_feature_dims=model_conf['feature_map']['n_dims'],
        use_chord_mhot_emb=False
    ).cuda()

    pretrained_dict = torch.load(inference_param_path)
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if 'feature_map.omega' not in k
    }
    model_state_dict = model.state_dict()
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)

    model.eval()
    print('[info] model loaded')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('[# pieces]', n_pieces)
    # sample_pieces = dset.pieces[:n_pieces]
    sample_pieces = random.sample(dset.pieces, n_pieces)

    # --- clip2key dictionary --- #
    clip2keyname, clip2keymode = find_key_emopia()

    # --- generation --- #
    for i, p in enumerate(sample_pieces):
        piece_data = pickle_load(p)
        melody_pos, chord_pos, piece_evs = piece_data[0], piece_data[1], piece_data[2]

        sample_name = p.split('/')[-1].split('.')[0]
        emotion = sample_name[:2]
        out_path = os.path.join(out_dir, 'sample_{:02d}-{}'.format(i + 1, sample_name))
        os.makedirs(out_path, exist_ok=True)
        print(out_path)

        # --- melody events --- #
        melody_events = []
        for pos in melody_pos:
            bar_melody_events = piece_evs[pos[0]+1:pos[1]]
            bar_melody_events = ['{}_{}'.format(e['name'], e['value']) for e in bar_melody_events]
            melody_events.append([dset.event2idx[e] for e in bar_melody_events])

        for n in range(samp_per_piece):
            for emotion_env in emotion_events:
                # --- emotion event --- #
                # if (emotion in ['Q1', 'Q4'] and emotion_env.split('_')[1] == 'Positive') or \
                #         (emotion in ['Q2', 'Q3'] and emotion_env.split('_')[1] == 'Negative'):
                #     continue

                # --- key event --- #
                if not model_based:
                    key_evs = None
                    # get key from samples
                    for i in range(len(piece_evs)):
                        if piece_evs[i]['name'] == 'Key':
                            key_evs = piece_evs[i]
                            break

                    # get key from dictionary if no key in sample sequence
                    if key_evs is None:
                        keymode = int(clip2keymode[sample_name])
                        if transpose_to_C:
                            keyname = 'C'
                        else:
                            keyname = clip2keyname[sample_name]
                        if keymode == 0:
                            key_evs = {'name': 'Key', 'value': keyname.upper()}
                        else:
                            key_evs = {'name': 'Key', 'value': keyname.lower()}

                    # enforce major key when Positive and minor key when Negative
                    if rule_based:
                        if emotion_env == 'Emotion_Positive':
                            key = '{}_{}'.format(key_evs['name'], key_evs['value'].upper())
                        else:
                            key = '{}_{}'.format(key_evs['name'], key_evs['value'].lower())
                    else:
                        key = '{}_{}'.format(key_evs['name'], key_evs['value'])

                    if relative_chord:
                        generated = [dset.event2idx[emotion_env], dset.event2idx[key]]  # add key
                        seg_inp = [0, 0]
                    else:
                        generated = [dset.event2idx[emotion_env]]
                        seg_inp = [0]

                # generate key according to emotion
                else:
                    with torch.no_grad():
                        generated, seg_inp, key = generate_key(model, dset.event2idx, dset.idx2event, emotion_env,
                                                               temp=temp, top_p=top_p, inadmissibles=None)
                print(emotion_env, key)

                # --- chord events --- #
                with torch.no_grad():
                    generated = generate_conditional(model, dset.event2idx, dset.idx2event,
                                                     melody_events, generated, seg_inp,
                                                     max_bars=max_bars, temp=temp, top_p=top_p, inadmissibles=None)

                generated = word2event(generated, dset.idx2event)
                events, lead_sheet_bars = events2bars(key, generated, relative_melody)
                if not transpose_to_C and not relative_chord:
                    key = 'Key_C'

                output_midi_path = os.path.join(out_path, 'lead_sheet_{}_{}.mid'.format(emotion_env.split('_')[-1], n))
                event_to_midi(
                    key, list(chain(*lead_sheet_bars[:max_bars])),
                    mode='skyline',
                    play_chords=True,
                    enforce_tempo=True,
                    enforce_tempo_evs=[TempoEvent(110, 0, 0)],
                    output_midi_path=output_midi_path)

                if play_midi:
                    output_wav_path = os.path.join(out_path, 'lead_sheet_{}_{}.wav'.format(emotion_env.split('_')[-1], n))
                    midi_to_wav(output_midi_path, output_wav_path)

                output_event_path = os.path.join(out_path, 'lead_sheet_{}_{}.txt'.format(emotion_env.split('_')[-1], n))
                event_to_txt(events, output_event_path=output_event_path)
