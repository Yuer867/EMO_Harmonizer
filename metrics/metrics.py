import os
import math
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4

IDX_TO_KEY = {
    9: 'A',
    10: 'A#',
    11: 'B',
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#'
}
KEY_TO_IDX = {
    v: k for k, v in IDX_TO_KEY.items()
}
MAJOR_KEY = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
MINOR_KEY = np.array(['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'])
QUALITY_TO_INTERVAL = {
    #        1     2     3     4  5     6     7
    'M':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'm':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    '+':    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'o':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4': [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus2': [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'M7':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'm7':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'o7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    '/o7':  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
}


def pickle_load(f):
    return pickle.load(open(f, 'rb'))


def chord_vocab():
    """
    get all chord types (root_quality, e.g., 0_M)
    """
    vocab = []

    # scale
    scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    scale = [KEY_TO_IDX[s] for s in scale]

    # quality
    standard_qualities = ['M', 'm', 'o', '+', '7', 'M7', 'm7', 'o7', '/o7', 'sus2', 'sus4']

    # combination
    for s in scale:
        for q in standard_qualities:
            vocab.append('{}_{}'.format(s, q))

    return vocab


def get_chord_notes(chord, key, relative=False):
    """
    get all chord notes
    """
    root, quality = chord.split('_')
    interval = QUALITY_TO_INTERVAL[quality]
    if not relative:
        root = (KEY_TO_IDX[key.upper()] + int(root)) % 12
    else:
        root = int(root)
    chord_tone = interval[-root:] + interval[:-root]
    notes = np.where(np.array(chord_tone) == 1)[0]
    return notes


def tonal_distance(notes):
    """
    tonal distance: calculate the PCP feature of the chord and project it to a derived 6-D tonal space
    """
    fifths_lookup = {9: [1.0, 0.0], 2: [math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)],
                     7: [math.cos(2.0 * math.pi / 6.0), math.sin(2.0 * math.pi / 6.0)],
                     0: [0.0, 1.0], 5: [math.cos(4.0 * math.pi / 6.0), math.sin(4.0 * math.pi / 6.0)],
                     10: [math.cos(5.0 * math.pi / 6.0), math.sin(5.0 * math.pi / 6.0)],
                     3: [-1.0, 0.0], 8: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                     1: [math.cos(8.0 * math.pi / 6.0), math.sin(8.0 * math.pi / 6.0)],
                     6: [0.0, -1.0], 11: [math.cos(10.0 * math.pi / 6.0), math.sin(10.0 * math.pi / 6.0)],
                     4: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}
    minor_thirds_lookup = {3: [1.0, 0.0], 7: [1.0, 0.0], 11: [1.0, 0.0],
                           0: [0.0, 1.0], 4: [0.0, 1.0], 8: [0.0, 1.0],
                           1: [-1.0, 0.0], 5: [-1.0, 0.0], 9: [-1.0, 0.0],
                           2: [0.0, -1.0], 6: [0.0, -1.0], 10: [0.0, -1.0]}
    major_thirds_lookup = {0: [0.0, 1.0], 3: [0.0, 1.0], 6: [0.0, 1.0], 9: [0.0, 1.0],
                           2: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           5: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           8: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           11: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           1: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
                           4: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
                           7: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
                           10: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}

    fifths = [0.0, 0.0]
    minor = [0.0, 0.0]
    major = [0.0, 0.0]
    r1 = 1
    r2 = 1
    r3 = 0.5
    for note in notes:
        for i in range(2):
            fifths[i] += r1 * fifths_lookup[note][i]
            minor[i] += r2 * minor_thirds_lookup[note][i]
            major[i] += r3 * major_thirds_lookup[note][i]
    for i in range(2):
        fifths[i] /= len(notes)
        minor[i] /= len(notes)
        major[i] /= len(notes)

    return np.array(fifths + minor + major)


def key2scale(key, repre='alpha'):
    """
    get key scale given a key
    """
    major_keys = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
    minor_keys = np.array(['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'])

    if key in major_keys:
        start = np.where(major_keys == key)[0][0]
        scale_range = np.concatenate([major_keys[start:], major_keys[:start]], axis=0)
        scale_idx = np.array([0, 2, 4, 5, 7, 9, 11])
        scale = scale_range[scale_idx]
    elif key in minor_keys:
        start = np.where(minor_keys == key)[0][0]
        scale_range = np.concatenate([major_keys[start:], major_keys[:start]], axis=0)
        scale_idx = np.array([0, 2, 3, 5, 7, 8, 10])
        scale = scale_range[scale_idx]
    else:
        print('wrong key type')
        scale = None
        scale_idx = None

    if repre == 'alpha':
        return scale
    elif repre == 'number':
        return scale_idx


def compute_CHE_and_CC(chord_events, vocab):
    """
    Chord histogram entropy (CHE): Create a histogram of chord occurrences with |vocab| bins and calculate its entropy.
    Chord coverage (CC): The number of chord labels with non-zero counts in the chord histogram in a chord sequence.
    """
    if len(chord_events) == 0:
        return 0, 0

    # chord histogram
    chord_statistics = {k: 0 for k in vocab}
    for e in chord_events:
        chord_statistics[e] += 1
    chord_histogram = np.array(list(chord_statistics.values()))

    # normalize the counts to sum to 1
    sequence_length = len(chord_events)
    chord_histogram = chord_histogram / sequence_length

    # calculate entropy
    CHE = sum([- p_i * np.log(p_i + 1e-6) for p_i in chord_histogram])

    # chord coverage
    CC = np.where(chord_histogram > 0)[0].shape[0]

    return CHE, CC


def compute_CTD(chord_events, key):
    """
    Chord tonal distance (CTD): the average value of the tonal distance computed between every pair of adjacent chords.
    """
    if len(chord_events) == 1 or len(chord_events) == 0:
        return 0

    distances = []
    for i in range(len(chord_events)-1):
        distance = np.sqrt(np.sum(
            (tonal_distance(get_chord_notes(chord_events[i], key)) -
             tonal_distance(get_chord_notes(chord_events[i + 1], key))
             ) ** 2
        ))
        distances.append(distance)
    CTD = np.mean(distances)
    return CTD


def compute_chord_progression_metrics(events):
    """
    compute three chord progression metrics: CHE, CC, CTD
    """
    key = None
    for evs in events:
        if "Key" in evs:
            key = evs.split('_')[1]
            break
    if key is None:
        raise ValueError('No key information.')
    vocab = chord_vocab()
    chord_events = []
    prev_chord = None
    for evs in events:
        if 'Chord_' in evs and 'None' not in evs:
            if evs == prev_chord:
                continue
            else:
                chord_events.append('_'.join(evs.split('_')[1:]))
                prev_chord = evs

    CHE, CC = compute_CHE_and_CC(chord_events, vocab)
    CTD = compute_CTD(chord_events, key)

    return CHE, CC, CTD


def compute_CTnCTR(key, notes, chord):
    """
    Chord tone to non-chord tone ratio (CTnCTR): count the number of chord tones, and non-chord tones
    as well as "proper" non-chord tone in the melody sequence, and compute the ratio.
    """
    # no notes
    if len(notes) == 0:
        return 0, 0, 0

    # no chord
    if chord == 'None_None':
        return 0, 0, len(notes)

    n_c, n_p, n_n = 0, 0, 0
    # chord tone range
    root, quality = chord.split('_')
    interval = QUALITY_TO_INTERVAL[quality]
    root = (KEY_TO_IDX[key.upper()] + int(root)) % 12
    chord_tone = interval[-root:] + interval[:-root]

    for i in range(len(notes)):
        if chord_tone[notes[i] % 12] == 1:
            n_c += 1
        else:
            n_n += 1
            for j in range(i, len(notes)):
                if notes[i] != notes[j]:
                    if chord_tone[notes[j] % 12] == 1 and abs(notes[i]-notes[j]) <= 2:
                        n_p += 1
                    break
    assert n_c + n_n == len(notes)

    return n_c, n_p, n_n


def compute_PCS(key, note, chord):
    """
    Pitch consonance score (PCS): For each melody note, calculate a consonance score with each of the notes of
    its corresponding chord label.
    """
    pcs = 0

    pitch = note % 12
    root, quality = chord.split('_')
    interval = QUALITY_TO_INTERVAL[quality]
    root = (KEY_TO_IDX[key.upper()] + int(root)) % 12
    chord_tone = interval[-root:] + interval[:-root]
    chord_note = np.where(np.array(chord_tone) == 1)[0]

    # if key in MAJOR_KEY:
    #     consonance_intervals = [0, 4, 7, 9]
    # else:
    #     consonance_intervals = [0, 3, 7, 8]
    consonance_intervals = [0, 3, 4, 7, 8, 9]
    for n in chord_note:
        if abs(n - pitch) in consonance_intervals:
            pcs += 1
        elif abs(n - pitch) == 5:
            pcs += 0
        else:
            pcs += -1

    return pcs


def compute_MCTD(key, note, chord):
    """
    Melody-chord tonal distance (MCTD): the average of the tonal distance between every melody note
    and corresponding the chord label calculated across a melody sequence,
    with each distance weighted by the duration of the corresponding melody note.
    """
    melody_note = [note % 12]
    chord_notes = get_chord_notes(chord, key)
    distance = np.sqrt(np.sum(
        (tonal_distance(melody_note) -
         tonal_distance(chord_notes)
         ) ** 2
    ))
    return distance


def compute_harmonicity_metrics(events):
    """
    compute three melody & chord harmonicity metrics: CTnCTR, PCS, MCTD
    """
    key = None
    for evs in events:
        if "Key" in evs:
            key = evs.split('_')[1]
            break
    if key is None:
        raise ValueError('No key information.')
    melody_pos = np.where(np.array(events) == "Track_Melody")[0].tolist()
    chord_pos = np.where(np.array(events) == "Track_Chord")[0].tolist()
    assert len(melody_pos) == len(chord_pos)

    n_bar = len(melody_pos)
    n_c, n_p, n_n = 0, 0, 0  # CTnCTR
    pcs = []  # PCS
    mctd = []  # MCTD
    for i in range(n_bar):
        # --- process melody --- #
        melody_events = events[melody_pos[i]:chord_pos[i]]
        beat2melody = defaultdict(list)  # 4-th note window
        n_beat = 0
        for evs in range(len(melody_events)):
            if 'Beat' in melody_events[evs]:
                n_beat = int(melody_events[evs].split('_')[1])
            if 'Note_Pitch' in melody_events[evs]:
                pitch = int(melody_events[evs].split('_')[2])
                duration = int(melody_events[evs+1].split('_')[2])
                n_tick = duration // TICK_RESOL
                for t in range(n_tick):
                    beat2melody[(n_beat+t)//4].append(pitch)

        # --- process chord --- #
        if i < n_bar - 1:
            chord_events = events[chord_pos[i]:melody_pos[i+1]]
        else:
            chord_events = events[chord_pos[i]:]
        beat2chord = defaultdict(lambda: 'None_None')
        n_beat = 0
        for evs in range(len(chord_events)):
            if 'Beat' in chord_events[evs]:
                n_beat = int(chord_events[evs].split('_')[1])
            if 'Chord_' in chord_events[evs]:
                chord = '_'.join(chord_events[evs].split('_')[1:])
                beat2chord[n_beat//4] = chord

        # --- compute CTnCTR for each beat --- #
        for beat in range(4):
            chord = beat2chord[beat]
            notes = beat2melody[beat]
            c, p, n = compute_CTnCTR(key, notes, chord)
            n_c += c
            n_p += p
            n_n += n

        # --- compute PCS & MCTD for each tick --- #
        for beat in range(4):
            chord = beat2chord[beat]
            if chord == 'None_None':
                continue
            notes = beat2melody[beat]
            for note in notes:
                pcs.append(compute_PCS(key, note, chord))
                mctd.append(compute_MCTD(key, note, chord))

    CTnCTR = (n_c + n_p) / (n_c + n_n)
    PCS = np.mean(pcs)
    MCTD = np.mean(mctd)
    return CTnCTR, PCS, MCTD


def compute_key_metrics(events):
    """
    compute two key harmonicity metrics
    - root ratio: the ratio of chord roots in key scale
    - notes ratio: the ratio of chord notes in key scale
    """
    key = None
    for evs in events:
        if "Key" in evs:
            key = evs.split('_')[1]
            break
    if key is None:
        raise ValueError('No key information.')
    scale = key2scale(key, repre='number')

    chord_roots = []
    chord_notes = []
    for evs in events:
        if 'Chord_' in evs and 'None' not in evs:
            chord_roots.append(int(evs.split('_')[1]))
            chord_notes.append(get_chord_notes('_'.join(evs.split('_')[1:]), key, relative=True))

    if len(chord_roots) == 0:
        return 0, 0

    # root ratio
    root_ratio = np.mean([r in scale for r in chord_roots])

    # notes ratio
    notes_ratio = []
    for notes in chord_notes:
        notes_ratio.append(np.mean([note in scale for note in notes]))
    notes_ratio = np.mean(notes_ratio)

    return root_ratio, notes_ratio


def compute_metrics(gen_leadsheet_dir, ground_truth_dir, num_sample=None):
    """
    compute all melody harmonization metrics for both ground truth files and generation files
    - chord progression metrics: CHE, CC, CTD
    - melody & chord harmonicity metrics: CTnCTR, PCS, MCTD
    - key & chord harmonicity metrics: root ratio, notes ratio
    """
    if num_sample is not None:
        out_dir = os.listdir(gen_leadsheet_dir)[:num_sample]
    else:
        out_dir = os.listdir(gen_leadsheet_dir)
    n_file = 0

    ground_truth_CHE = []
    ground_truth_CC = []
    ground_truth_CTD = []
    ground_truth_CTnCTR = []
    ground_truth_PCS = []
    ground_truth_MCTD = []
    ground_truth_root_ratio = []
    ground_truth_notes_ratio = []

    generation_CHE = []
    generation_CC = []
    generation_CTD = []
    generation_CTnCTR = []
    generation_PCS = []
    generation_MCTD = []
    generation_root_ratio = []
    generation_notes_ratio = []

    print("compute metrics...")
    for i in tqdm(range(len(out_dir))):
        sample_dir = os.path.join(gen_leadsheet_dir, out_dir[i])
        sample_name = '-'.join(out_dir[i].split('-')[1:])

        ground_truth_file = os.path.join(ground_truth_dir, sample_name + '.pkl')
        lead_sheet_files = [x for x in os.listdir(sample_dir) if '.txt' in x]
        n_file += len(lead_sheet_files)

        # compute metrics for ground truth files
        ground_truth_evs = pickle_load(ground_truth_file)[2]
        ground_truth_evs = ['{}_{}'.format(s['name'], s['value']) for s in ground_truth_evs]

        # (1) chord progression metrics
        CHE, CC, CTD = compute_chord_progression_metrics(ground_truth_evs)
        ground_truth_CHE.append(CHE)
        ground_truth_CC.append(CC)
        ground_truth_CTD.append(CTD)

        # (2) melody & chord harmonicity metrics
        CTnCTR, PCS, MCTD = compute_harmonicity_metrics(ground_truth_evs)
        ground_truth_CTnCTR.append(CTnCTR)
        ground_truth_PCS.append(PCS)
        ground_truth_MCTD.append(MCTD)

        # (3) key & chord harmonicity metrics
        root_ratio, notes_ratio = compute_key_metrics(ground_truth_evs)
        ground_truth_root_ratio.append(root_ratio)
        ground_truth_notes_ratio.append(notes_ratio)

        # compute metrics for generation files
        for n in range(len(lead_sheet_files)):
            file_path = os.path.join(sample_dir, lead_sheet_files[n])
            events = open(file_path).read().splitlines()

            # (1) chord progression metrics
            CHE, CC, CTD = compute_chord_progression_metrics(events)
            generation_CHE.append(CHE)
            generation_CC.append(CC)
            generation_CTD.append(CTD)

            # (2) melody & chord harmonicity metrics
            CTnCTR, PCS, MCTD = compute_harmonicity_metrics(events)
            generation_CTnCTR.append(CTnCTR)
            generation_PCS.append(PCS)
            generation_MCTD.append(MCTD)

            # (3) key & chord harmonicity metrics
            root_ratio, notes_ratio = compute_key_metrics(events)
            generation_root_ratio.append(root_ratio)
            generation_notes_ratio.append(notes_ratio)

    print('ground truth:', ground_truth_dir)
    print('- CHE:', np.mean(ground_truth_CHE))
    print('- CC:', np.mean(ground_truth_CC))
    print('- CTD:', np.mean(ground_truth_CTD))
    print('- CTnCTR:', np.mean(ground_truth_CTnCTR))
    print('- PCS:', np.mean(ground_truth_PCS))
    print('- MCTD:', np.mean(ground_truth_MCTD))
    print('- root ratio:', np.mean(ground_truth_root_ratio))
    print('- notes ratio:', np.mean(ground_truth_notes_ratio))
    print()

    print('generation (number of samples: {}): {}'.format(n_file, gen_leadsheet_dir))
    print('- CHE:', np.mean(generation_CHE))
    print('- CC:', np.mean(generation_CC))
    print('- CTD:', np.mean(generation_CTD))
    print('- CTnCTR:', np.mean(generation_CTnCTR))
    print('- PCS:', np.mean(generation_PCS))
    print('- MCTD:', np.mean(generation_MCTD))
    print('- root ratio:', np.mean(generation_root_ratio))
    print('- notes ratio:', np.mean(generation_notes_ratio))
    print()


if __name__ == '__main__':
    compute_metrics(gen_leadsheet_dir='generation/emopia_functional',
                    ground_truth_dir='emopia_events/lead_sheet_chord11_ablated/events/')