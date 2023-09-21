import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from scipy.special import kl_div
from scipy.spatial import distance


DEFAULT_KEY = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
               'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
DEFAULT_ROOT = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
DEFAULT_QUALITY = ['M', 'm', 'o', '+', '7', 'M7', 'm7', 'o7', '/o7', 'sus2', 'sus4']
DEFAULT_PROGRESSION = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']


def pickle_load(f):
    return pickle.load(open(f, 'rb'))


def compute_metrics_generation(gen_leadsheet_dir, num_sample=None, plot_dist=False):
    """
    compute emotion-related metrics for generation files
    """
    if num_sample is not None:
        out_dir = os.listdir(gen_leadsheet_dir)[:num_sample]
    else:
        out_dir = os.listdir(gen_leadsheet_dir)
    n_file = 0

    emotion_keys = {'Q1': {i: 1e-6 for i in DEFAULT_KEY}, 'Q2': {i: 1e-6 for i in DEFAULT_KEY}}
    emotion_roots = {'Q1': {i: 1e-6 for i in DEFAULT_ROOT}, 'Q2': {i: 1e-6 for i in DEFAULT_ROOT}}
    emotion_qualities = {'Q1': {i: 1e-6 for i in DEFAULT_QUALITY}, 'Q2': {i: 1e-6 for i in DEFAULT_QUALITY}}
    emotion_progression = {'Q1': {i: 1e-6 for i in DEFAULT_PROGRESSION}, 'Q2': {i: 1e-6 for i in DEFAULT_PROGRESSION}}

    for i in tqdm(range(len(out_dir))):
        sample_dir = os.path.join(gen_leadsheet_dir, out_dir[i])
        lead_sheet_files = [x for x in os.listdir(sample_dir) if '.txt' in x]
        n_file += len(lead_sheet_files)

        # compute metrics for generation files
        for n in range(len(lead_sheet_files)):
            file_path = os.path.join(sample_dir, lead_sheet_files[n])
            events = open(file_path).read().splitlines()

            # emotion tag
            if 'Positive' in file_path:
                emo = 'Q1'
            elif 'Negative' in file_path:
                emo = 'Q2'

            # update distributions
            prev_chord = None
            prev_root = None
            for evs in events:
                if 'Key' in evs:
                    # update key distribution
                    key = evs.split('_')[1]
                    emotion_keys[emo][key] += 1
                if 'Chord_' in evs:
                    chord = '_'.join(evs.split('_')[1:])
                    if chord == prev_chord or 'None' in chord:
                        continue
                    root, quality = chord.split('_')

                    # update chord-related distributions
                    emotion_roots[emo][root] += 1
                    emotion_qualities[emo][quality] += 1
                    if prev_root is not None and root != prev_root:
                        emotion_progression[emo][str((int(root) - int(prev_root) + 12) % 12)] += 1

                    # update previous chord
                    prev_chord = chord
                    prev_root = root

    # plot distributions of emotion-related properties
    if plot_dist:
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#',
                'g', 'g#', 'a', 'a#', 'b']
        qualities = ['M', 'm7', 'M7', 'm', 'sus2', '7', 'sus4', 'o7', '+', 'o', '/o7']
        progressions = ['5', '7', '2', '10', '8', '9', '3', '4', '1', '11', '6']
        plot_distribution(emotion_dist=emotion_keys, keys=keys, title='Key')
        # plot_distribution(emotion_dist=emotion_roots, keys= , title='Root')
        plot_distribution(emotion_dist=emotion_qualities, keys=qualities, title='Quality')
        plot_distribution(emotion_dist=emotion_progression, keys=progressions, title='Progression')

    emotion_keys = sort_emotion_dist(emotion_keys)
    emotion_roots = sort_emotion_dist(emotion_roots)
    emotion_qualities = sort_emotion_dist(emotion_qualities)
    emotion_progression = sort_emotion_dist(emotion_progression)

    print('number of samples: ', n_file)

    return emotion_keys, emotion_roots, emotion_qualities, emotion_progression


def compute_metrics_truth(ground_truth_dir, plot_dist=False):
    """
    compute emotion-related metrics for ground truth files as reference
    """
    files = os.listdir(ground_truth_dir)
    train_split = pickle_load('../emopia_events/data_splits/train.pkl')
    files = [f for f in files if f in train_split]
    print('number of files', len(files))

    emotion_keys = {'Q1': {i: 1e-6 for i in DEFAULT_KEY}, 'Q2': {i: 1e-6 for i in DEFAULT_KEY}}
    emotion_roots = {'Q1': {i: 1e-6 for i in DEFAULT_ROOT}, 'Q2': {i: 1e-6 for i in DEFAULT_ROOT}}
    emotion_qualities = {'Q1': {i: 1e-6 for i in DEFAULT_QUALITY}, 'Q2': {i: 1e-6 for i in DEFAULT_QUALITY}}
    emotion_progression = {'Q1': {i: 1e-6 for i in DEFAULT_PROGRESSION}, 'Q2': {i: 1e-6 for i in DEFAULT_PROGRESSION}}

    for file in tqdm(files):
        ground_truth_file = os.path.join(ground_truth_dir, file)
        ground_truth_evs = pickle_load(ground_truth_file)[2]
        ground_truth_evs = ['{}_{}'.format(s['name'], s['value']) for s in ground_truth_evs]

        # emotion tag
        emo = file[:2]
        if emo in ['Q1', 'Q4']:
            emo = 'Q1'
        else:
            emo = 'Q2'

        # update distributions
        prev_chord = None
        prev_root = None
        for evs in ground_truth_evs:
            if 'Key' in evs:
                # update key distribution
                key = evs.split('_')[1]
                emotion_keys[emo][key] += 1
            if 'Chord_' in evs:
                chord = '_'.join(evs.split('_')[1:])
                if chord == prev_chord or 'None' in chord:
                    continue
                root, quality = chord.split('_')

                # update chord-related distributions
                emotion_roots[emo][root] += 1
                emotion_qualities[emo][quality] += 1
                if prev_root is not None and root != prev_root:
                    emotion_progression[emo][str((int(root) - int(prev_root) + 12) % 12)] += 1

                # update previous chord
                prev_chord = chord
                prev_root = root

    # plot distributions of emotion-related properties
    if plot_dist:
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#',
                'g', 'g#', 'a', 'a#', 'b']
        qualities = ['M', 'm7', 'M7', 'm', 'sus2', '7', 'sus4', 'o7', '+', 'o', '/o7']
        progressions = ['5', '7', '2', '10', '8', '9', '3', '4', '1', '11', '6']
        plot_distribution(emotion_dist=emotion_keys, keys=keys, title='Key')
        # plot_distribution(emotion_dist=emotion_roots, keys= , title='Root')
        plot_distribution(emotion_dist=emotion_qualities, keys=qualities, title='Quality')
        plot_distribution(emotion_dist=emotion_progression, keys=progressions, title='Progression')

    emotion_keys = sort_emotion_dist(emotion_keys)
    emotion_roots = sort_emotion_dist(emotion_roots)
    emotion_qualities = sort_emotion_dist(emotion_qualities)
    emotion_progression = sort_emotion_dist(emotion_progression)

    return emotion_keys, emotion_roots, emotion_qualities, emotion_progression


def sort_emotion_dist(emotion_dist, key=0, prob=True):
    """
    sort the values for keys(Q1, Q2) according to key name
    """
    new_dist = {}
    for emo in emotion_dist:
        dist = sorted(emotion_dist[emo].items(), key=lambda x: x[key], reverse=True)
        keys = np.array([i[0] for i in dist])
        values = np.array([i[1] for i in dist])
        if prob:
            probs = values / np.sum(values)
            new_dist[emo] = {'keys': keys, 'values': values, 'probs': probs}
        else:
            new_dist[emo] = {'keys': keys, 'values': values}
    return new_dist


def compute_distance(ground_truth_dist, generation_dist, metric='KL'):
    """
    compute distances between distributions of generated samples and real data
    for positive and negative emotions respectively
    """
    results = []
    for emo in ground_truth_dist:
        if emo not in generation_dist:
            continue
        ground_truth_probs = ground_truth_dist[emo]['probs']
        generation_probs = generation_dist[emo]['probs']
        if metric == 'KL':
            results.append(kl_div(ground_truth_probs, generation_probs).sum())
        elif metric == 'JS':
            results.append(distance.jensenshannon(ground_truth_probs, generation_probs))
        else:
            raise ValueError('invalid distance metric {}, should be KL or JS'.format(metric))
    return np.mean(results)


def compute_distance_truth(ground_truth_dist, metric='KL'):
    """
    compute distances between distributions of positive emotion and negative emotion for real data
    """
    ground_truth_probs_Q1 = ground_truth_dist['Q1']['probs']
    ground_truth_probs_Q2 = ground_truth_dist['Q2']['probs']
    if metric == 'KL':
        return kl_div(ground_truth_probs_Q1, ground_truth_probs_Q2).sum()
    elif metric == 'JS':
        return distance.jensenshannon(ground_truth_probs_Q1, ground_truth_probs_Q2)
    else:
        raise ValueError('invalid distance metric {}, should be KL or JS'.format(metric))


def plot_distribution(emotion_dist, keys, title):
    """
    plot distributions to compare different emotion classes
    """
    high_valence_dist = emotion_dist['Q1']
    total_high = sum([high_valence_dist[k] for k in keys])
    low_valence_dist = emotion_dist['Q2']
    total_low = sum([high_valence_dist[k] for k in keys])

    high_values = [high_valence_dist[k] / total_high for k in keys]
    low_values = [low_valence_dist[k] / total_low for k in keys]

    plt.figure(figsize=(len(keys), 8))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    x = np.arange(len(keys))
    bar_width = 0.4
    plt.bar(x, high_values, bar_width, label='High Valence', align='center', color='#5D9DD3')
    plt.bar(x + bar_width, low_values, bar_width, label='Low Valence', align='center', color='#FDD86F')
    plt.xticks(x + bar_width / 2, keys, fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=30)
    plt.savefig('figures/emo_{}_dist.png'.format(title), dpi=500, bbox_inches='tight')
    plt.show()

    # high_valence_dist = sorted(high_valence_dist.items(), key=lambda x: x[1], reverse=True)
    # keys = [i[0] for i in high_valence_dist]
    # high_values = [i[1] / total_high for i in high_valence_dist]
    #
    # plt.bar(keys, high_values, label='High Valence', color='#5D9DD3')
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    # plt.legend(fontsize=12)
    # plt.savefig('figures/' + title + '_High_Valence.png', dpi=500, bbox_inches='tight')
    # plt.show()
    #
    # low_valence_dist = sorted(low_valence_dist.items(), key=lambda x: x[1], reverse=True)
    # keys = [i[0] for i in low_valence_dist]
    # low_values = [i[1] / total_low for i in low_valence_dist]
    #
    # plt.bar(keys, low_values, label='Low Valence', color='#FDD86F')
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    # plt.legend(fontsize=12)
    # plt.savefig('figures/' + title + '_Low_Valence.png', dpi=500, bbox_inches='tight')
    # plt.show()


def to_percent(y, position):
    return f'{y * 100:.1f}%'


if __name__ == '__main__':
    ground_truth_dir = 'emopia_events/lead_sheet_chord11_ablated/events/'
    gen_leadsheet_dir = 'generation/emopia_functional'
    distance_metric = 'KL'

    # compute ground truth distributions
    ground_truth_dist = compute_metrics_truth(ground_truth_dir, plot_dist=False)
    ground_truth_keys, ground_truth_roots, ground_truth_qualities, ground_truth_progression = ground_truth_dist

    # compute generation distributions
    generation_dist = compute_metrics_generation(gen_leadsheet_dir, plot_dist=False)
    generation_keys, generation_roots, generation_qualities, generation_progression = generation_dist

    # compute KL divergence / JS distance
    print('Ground Truth ({})'.format(distance_metric))
    print('- keys:', compute_distance_truth(ground_truth_keys, metric=distance_metric))
    print('- roots:', compute_distance_truth(ground_truth_roots, metric=distance_metric))
    print('- qualities:', compute_distance_truth(ground_truth_qualities, metric=distance_metric))
    print('- progression:', compute_distance_truth(ground_truth_progression, metric=distance_metric))

    print('Generation ({})'.format(distance_metric))
    print(gen_leadsheet_dir)
    print('- keys:', compute_distance(ground_truth_keys, generation_keys, metric=distance_metric))
    print('- roots:', compute_distance(ground_truth_roots, generation_roots, metric=distance_metric))
    print('- qualities:', compute_distance(ground_truth_qualities, generation_qualities, metric=distance_metric))
    print('- progression:', compute_distance(ground_truth_progression, generation_progression, metric=distance_metric))


