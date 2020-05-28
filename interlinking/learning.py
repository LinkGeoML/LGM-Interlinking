import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import itertools

from interlinking import config, helpers
from interlinking import sim_measures


def learn_thres(fname, sim_group='basic'):
    """Compute optimal thresholds for each similarity metric in specified group regarding the `fname` dataset.

    Parameters
    -----------
    fname: :obj:`str`
        File name of the dataset.
    sim_group: :obj:`str`
        Name of the group of the similarity metrics. Accepted values are *basic* or *basic_sorted*.

    """
    low_thres = 30
    high_thres = 91
    step = 5

    assert (os.path.isfile(os.path.join(config.default_data_path, fname))), f'{fname} dataset does not exist'

    start_time = time.time()

    data_df = pd.read_csv(os.path.join(config.default_data_path, fname), sep=config.delimiter, names=config.fieldnames,
                          na_filter=False, encoding='utf8')
    print(f'The train data loaded in {(time.time() - start_time):.2f} sec.')

    sim_res = None
    if sim_group == 'basic':
        sim_res = np.asarray(list(
            map(_compute_basic_similarities, data_df[config.use_cols['s1']], data_df[config.use_cols['s2']])
        ), dtype=float)
    elif sim_group == 'sorted':
        sim_res = np.asarray(list(
            map(_compute_sorted_similarities, data_df[config.use_cols['s1']], data_df[config.use_cols['s2']])
        ), dtype=float)

    print(f'The similarity scores were computed in {(time.time() - start_time):.2f}.')

    res = {}
    for m in helpers.StaticValues.sim_metrics.keys(): res[m] = []

    separator = ''
    print('Computing stats for thresholds', end='')
    for i in range(low_thres, high_thres, step):
        sim_thres = float(i / 100.0)
        print('{0} {1}'.format(separator, sim_thres), end='', flush=True)
        separator = ','

        idx = 0
        for sim, val in helpers.StaticValues.sim_metrics.items():
            if sim_group in val:
                acc = accuracy_score(data_df[config.use_cols['status']], sim_res[:, idx] >= sim_thres)
                res[sim].append([acc, float(i / 100.0)])
                idx += 1

    print('\nThe process took {0:.2f} sec\n'.format(time.time() - start_time))

    for key, val in res.items():
        if len(val) == 0:
            print('{0} is empty'.format(key))
            continue

        print(key, max(val, key=lambda x: x[0]))


def learn_params_for_lgm(fname, encoding):
    """Compute optimal thresholds and weights for each similarity metric in the ``LGM-Sim`` group only regarding the
    `fname` dataset.

    :param fname: File name of the dataset.
    :type fname: str
    :param encoding: Encoding of the `fname` dataset.
    :type encoding: str
    """
    low_thres = 30
    high_thres = 91
    step = 5
    low_split_thres = 50
    high_split_thres = 91
    split_step = 10
    sim_group = 'lgm'

    assert (os.path.isfile(os.path.join(config.default_data_path, fname))), f'{fname} dataset does not exist'

    gstart_time = time.time()

    data_df = pd.read_csv(os.path.join(config.default_data_path, fname), sep=config.delimiter, names=config.fieldnames,
                          na_filter=False, encoding='utf8')
    sim_measures.LGMSimVars().load_freq_terms(encoding)

    print(f'The train data and frequent terms loaded in {(time.time() - gstart_time):.2f} sec.')

    res = {}
    for m in helpers.StaticValues.sim_metrics.keys(): res[m] = []

    for s in range(low_split_thres, high_split_thres, split_step):
        split_thres = float(s / 100.0)

        start_time = time.time()

        sim_res = np.asarray(
            list(map(
                _compute_lgm_similarities, data_df[config.use_cols['s1']], data_df[config.use_cols['s2']],
                [split_thres]*len(data_df.index))
            ), dtype=float
        )
        fscore = np.zeros(sim_res.shape[0])
        print(f'The similarity scores were computed in {(time.time() - start_time):.2f} sec.')

        print(f'Computing stats for thresholds split: {split_thres}', end='')
        separator = ' and similarity:'
        # print('Computing stats for sim thres ', end='', flush=True)
        for i in range(low_thres, high_thres, step):
            sim_thres = float(i / 100.0)
            print('{0} {1}'.format(separator, sim_thres), end='', flush=True)
            separator = ','

            for n in [3.34] + list(range(2, 8)):
                weight_combs = [
                    tuple(float(x / 10.0) for x in seq)
                    for seq in itertools.product([1, 2, 3, 4, 5, 6, 2.5, 3.33], repeat=2)
                    if sum(seq) == (10 - n)
                ]
                # print('Computing stats for weights ({})'.format(','.join(map(str, w))))
                for w in weight_combs:
                    w = (float(n / 10.0),) + w

                    idx = 0
                    for sim, val in helpers.StaticValues.sim_metrics.items():
                        if sim_group in val:
                            scols = [idx*9 + 1, idx*9 + 2, idx*9 + 4, idx*9 + 5, idx*9 + 7, idx*9 + 8]

                            lweights = sim_measures.recalculate_weights_opt(
                                sim_res[:, scols[0:2]],
                                sim_res[:, scols[2:4]],
                                sim_res[:, scols[4:6]],
                                avg=True, weights=np.full((sim_res.shape[0], 3), list(w))
                            )
                            fscore = sim_res[:, idx*9] * lweights[:, 0] + \
                                     sim_res[:, idx*9 + 3] * lweights[:, 1] + \
                                     sim_res[:, idx*9 + 6] * lweights[:, 2]

                            acc = accuracy_score(data_df[config.use_cols['status']], fscore >= sim_thres)
                            res[sim].append([acc, float(i / 100.0), [split_thres, list(w)]])
                            idx += 1
        print()

    print('\nThe process took {0:.2f} sec\n'.format(time.time() - gstart_time))

    for key, val in res.items():
        if len(val) == 0:
            print('{0} is empty'.format(key))
            continue

        max_val = max(val, key=lambda x: x[0])
        print('{}: {}'.format(key, list(max_val)))


def _compute_basic_similarities(a, b):
    f = []
    for sim, val in helpers.StaticValues.sim_metrics.items():
        if 'basic' in val:
            if '_reversed' in sim: f.append(getattr(sim_measures, sim[:-len('_reversed')])(a[::-1], b[::-1]))
            else: f.append(getattr(sim_measures, sim)(a, b))
    return f


def _compute_sorted_similarities(a, b):
    a, b = helpers.transform(a, b, sorting=True, canonical=True, simple_sorting=True)

    f = []
    for sim, val in helpers.StaticValues.sim_metrics.items():
        if 'sorted' in val:
            if '_reversed' in sim: f.append(getattr(sim_measures, sim[:-len('_reversed')])(a[::-1], b[::-1]))
            else: f.append(getattr(sim_measures, sim)(a, b))

    return f


def _compute_lgm_similarities(a, b, split_thres):
    a, b = helpers.transform(a, b, sorting=True, canonical=True)

    f = []
    for sim, val in helpers.StaticValues.sim_metrics.items():
        if 'lgm' in val:
            if '_reversed' in sim:
                base_t, mis_t, special_t = sim_measures.lgm_sim_split(a[::-1], b[::-1], split_thres)
                base_score, mis_score, special_score = sim_measures.score_per_term(
                    base_t, mis_t, special_t, sim[:-len('_reversed')])
            else:
                base_t, mis_t, special_t = sim_measures.lgm_sim_split(a, b, split_thres)
                base_score, mis_score, special_score = sim_measures.score_per_term(base_t, mis_t, special_t, sim)
            f.extend([
                base_score, base_t['len'], base_t['char_len'],
                mis_score, mis_t['len'], mis_t['char_len'],
                special_score, special_t['len'], special_t['char_len'],
            ])

    return f
