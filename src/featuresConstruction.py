# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import pandas as pd
import numpy as np
import config
from itertools import chain
import os
import glob

from helpers import StaticValues, transform, getBasePath
from sim_measures import LSimilarityVars, lsimilarity_terms, score_per_term, weighted_terms


class Features:
    """
    It loads data and builds the features to be used as input to various scenarios: basic, basic_sorted or lgm
    """
    max_freq_terms = 200

    fields = [
        "s1",
        "s2",
        "status",
        "gid1",
        "gid2",
        "alphabet1",
        "alphabet2",
        "alpha2_cc1",
        "alpha2_cc2",
    ]

    dtypes = {
        's1': str, 's2': str,
        'status': str,
        'gid1': np.int32, 'gid2': np.int32,
        'alphabet1': str, 'alphabet1': str,
        'alpha2_cc1': str, 'alpha2_cc2': str
    }

    d = {
        'TRUE': True,
        'FALSE': False
    }

    def __init__(self):
        self.clf_method = config.initialConfig.classification_method
        self.data_df = None

    def load_data(self, fname, encoding):
        self.data_df = pd.read_csv(fname, sep='\t', names=self.fields, dtype=self.dtypes, na_filter=False)
        self._get_freqterms(encoding)

    def build_features(self):
        y = self.data_df['status'].str.upper().map(self.d).values

        fX = None
        if self.clf_method.lower() == 'basic':
            fX = np.asarray(
                list(map(self._compute_basic_features, self.data_df['s1'], self.data_df['s2'])), dtype=float)
        elif self.clf_method.lower() == 'basic_sorted':
            fX = np.asarray(
                list(map(self._compute_sorted_features, self.data_df['s1'], self.data_df['s2'])), dtype=float)
        else:  # lgm
            fX = np.asarray(list(map(self._compute_all_features, self.data_df['s1'], self.data_df['s2'])), dtype=float)

        return fX, y

    def _compute_all_features(self, s1, s2, sorting=True, all_features=True):
        f = []
        for status in list({False, sorting}):
            a, b = transform(s1, s2, sorting=status, canonical=status)

            sim1 = StaticValues.algorithms['damerau_levenshtein'](a, b)
            sim8 = StaticValues.algorithms['jaccard'](a, b)
            sim2 = StaticValues.algorithms['jaro'](a, b)
            sim3 = StaticValues.algorithms['jaro_winkler'](a, b)
            sim4 = StaticValues.algorithms['jaro_winkler'](a[::-1], b[::-1])
            sim11 = StaticValues.algorithms['monge_elkan'](a, b)
            sim7 = StaticValues.algorithms['cosine'](a, b)
            sim9 = StaticValues.algorithms['strike_a_match'](a, b)
            sim12 = StaticValues.algorithms['soft_jaccard'](a, b)
            if not status: sim5 = StaticValues.algorithms['sorted_winkler'](a, b)
            sim10 = StaticValues.algorithms['skipgram'](a, b)
            sim13 = StaticValues.algorithms['davies'](a, b)
            if status:
                sim14 = StaticValues.algorithms['l_jaro_winkler'](a, b)
                sim15 = StaticValues.algorithms['l_jaro_winkler'](a[::-1], b[::-1])

            if status: f.append([sim1, sim2, sim3, sim4, sim7, sim8, sim9, sim10, sim11, sim12, sim13, sim14, sim15])
            else: f.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])

        if all_features:
            a, b = transform(s1, s2, sorting=True, canonical=True)

            sim1 = self._compute_lsimilarity(a, b, 'damerau_levenshtein')
            sim2 = self._compute_lsimilarity(a, b, 'davies')
            sim3 = self._compute_lsimilarity(a, b, 'skipgram')
            sim4 = self._compute_lsimilarity(a, b, 'soft_jaccard')
            sim5 = self._compute_lsimilarity(a, b, 'strike_a_match')
            sim6 = self._compute_lsimilarity(a, b, 'cosine')
            sim7 = self._compute_lsimilarity(a, b, 'jaccard')
            sim8 = self._compute_lsimilarity(a, b, 'monge_elkan')
            sim9 = self._compute_lsimilarity(a, b, 'jaro_winkler')
            sim10 = self._compute_lsimilarity(a, b, 'jaro')
            sim11 = self._compute_lsimilarity(a, b, 'jaro_winkler_r')
            sim12 = self._compute_lsimilarity(a, b, 'l_jaro_winkler')
            sim13 = self._compute_lsimilarity(a, b, 'l_jaro_winkler_r')
            sim14, sim15, sim16 = list(self._compute_lsimilarity_base_scores(a, b, 'damerau_levenshtein'))

            f.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13, sim14, sim15, sim16])

        f = list(chain.from_iterable(f))

        return f

    def _compute_sorted_features(self, s1, s2):
        return self._compute_all_features(s1, s2, True, False)

    def _compute_basic_features(self, s1, s2):
        return self._compute_all_features(s1, s2, False, False)

    @staticmethod
    def _compute_lsimilarity(s1, s2, metric, w_type='avg'):
        baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
            s1, s2, LSimilarityVars.per_metric_optimal_values[metric][w_type][0])

        if metric in ['jaro_winkler_r', 'l_jaro_winkler_r']:
            return weighted_terms(
                {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']],
                 'len': baseTerms['len'], 'char_len': baseTerms['char_len']},
                {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']],
                 'len': mismatchTerms['len'], 'char_len': mismatchTerms['char_len']},
                {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']],
                 'len': specialTerms['len'], 'char_len': specialTerms['char_len']},
                metric[:-2], True if w_type == 'avg' else False
            )
        else:
            return weighted_terms(baseTerms, mismatchTerms, specialTerms, metric, True if w_type == 'avg' else False)

    @staticmethod
    def _compute_lsimilarity_base_scores(s1, s2, metric, w_type='avg'):
        baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
            s1, s2, LSimilarityVars.per_metric_optimal_values[metric][w_type][0])
        return score_per_term(baseTerms, mismatchTerms, specialTerms, metric)

    def _get_freqterms(self, encoding):
        print("Resetting any previously assigned frequent terms ...")
        LSimilarityVars.freq_ngrams['tokens'].clear()
        LSimilarityVars.freq_ngrams['chars'].clear()

        input_path = (True, os.path.join(getBasePath(), 'data/input/')) \
            if os.path.isdir(os.path.join(getBasePath(), 'data/input/')) \
            else (os.path.isdir(os.path.join(getBasePath(), '../data/input/')), os.path.join(getBasePath(), '../input/'))
        if input_path[0]:
            for f in glob.iglob(os.path.join(input_path[1], '*gram*{}{}.csv'.format('_', encoding))):
                gram_type = 'tokens' if 'token' in os.path.basename(os.path.normpath(f)) else 'chars'

                print("Loading frequent terms from file {} ...".format(f))
                df = pd.read_csv(f, sep='\t', header=0, names=['term', 'no'], nrows=self.max_freq_terms)
                LSimilarityVars.freq_ngrams[gram_type].update(df['term'].values.tolist())
            print('Frequent terms loaded.')
        else: print("Folder 'input' does not exist")
