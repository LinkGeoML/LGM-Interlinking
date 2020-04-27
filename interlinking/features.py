# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import pandas as pd
import numpy as np
from interlinking import config
from itertools import chain
import os
import glob

from interlinking.helpers import transform, getBasePath
from interlinking.sim_measures import LGMSimVars, lgm_sim_split, score_per_term, weighted_sim, sim_measures


class Features:
    """
    This class loads the dataset, frequent terms and builds features that are used as input to supported classification
    groups:

    * *basic*: similarity features based on basic similarity measures.
    * *basic_sorted*: similarity features based on sorted version of the basic similarity measures used in *basic* group.
    * *lgm*: similarity features based on variations of LGM-Sim similarity measures.

    See Also
    --------
    :func:`compute_features`: Details on the metrics each classification group implements.
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
        'alphabet1': str, 'alphabet2': str,
        'alpha2_cc1': str, 'alpha2_cc2': str
    }

    d = {
        'TRUE': True,
        'FALSE': False
    }

    def __init__(self):
        self.clf_method = config.MLConf.classification_method
        self.data_df = None

    def load_data(self, fname, encoding):
        self.data_df = pd.read_csv(fname, sep=config.delimiter, names=config.fieldnames, dtype=self.dtypes,
                                   na_filter=False, encoding='utf8')
        LGMSimVars().load_freq_terms(encoding)

    def build(self):
        """Build features depending on the assignment of parameter :py:attr:`~interlinking.config.MLConf.classification_method`
        and return values (fX, y) as ndarray of floats.

        Returns
        -------
        fX: ndarray
            The computed features that will be used as input to ML classifiers.
        y: ndarray
            Binary labels {True, False} to train the classifiers.
        """
        y = self.data_df['status'].str.upper().map(self.d).values

        fX = None
        if self.clf_method.lower() == 'basic':
            fX = np.asarray(
                list(map(self._compute_basic_features, self.data_df['s1'], self.data_df['s2'])), dtype=float)
        elif self.clf_method.lower() == 'basic_sorted':
            fX = np.asarray(
                list(map(self._compute_sorted_features, self.data_df['s1'], self.data_df['s2'])), dtype=float)
        else:  # lgm
            fX = np.asarray(list(map(self.compute_features, self.data_df['s1'], self.data_df['s2'])), dtype=float)

        return fX, y

    def compute_features(self, s1, s2, sorted=True, lgm_sims=True):
        """
        Depending on the group assigned to parameter :py:attr:`~interlinking.config.MLConf.classification_method`,
        this method builds an ndarray of the following groups of features:

        * *basic*: various similarity measures, i.e.,
          :func:`~interlinking.sim_measures.damerau_levenshtein`,
          :func:`~interlinking.sim_measures.jaro`,
          :func:`~interlinking.sim_measures.jaro_winkler` and the reversed one,
          :func:`~interlinking.sim_measures.sorted_winkler`,
          :func:`~interlinking.sim_measures.cosine`,
          :func:`~interlinking.sim_measures.jaccard`,
          :func:`~interlinking.sim_measures.strike_a_match`,
          :func:`~interlinking.sim_measures.monge_elkan`,
          :func:`~interlinking.sim_measures.soft_jaccard`,
          :func:`~interlinking.sim_measures.davies`,
          :func:`~interlinking.sim_measures.tuned_jaro_winkler` and the reversed one,
          :func:`~interlinking.sim_measures.skipgrams`.
        * *basic_sorted*: sorted versions of similarity measures utilized in *basic* group, except for the
          :func:`~interlinking.sim_measures.sorted_winkler`.
        * *lgm*: LGM-Sim variations that integrate, as internal, the similarity measures utilized in *basic* group,
          except for the :func:`~interlinking.sim_measures.sorted_winkler`.

        Parameters
        ----------
        s1, s2: str
            Input toponyms.
        sorted: bool, optional
            Value of True indicate to build features for groups *basic* and *basic_sorted*, value of False only for *basic* group.
        lgm_sims: bool, optional
            Values of True or False indicate whether to build or not features for group *lgm*.

        Returns
        -------
        :obj:`list`
            It returns a list (vector) of features.
        """
        f = []
        for status in list({False, sorted}):
            a, b = transform(s1, s2, sorting=status, canonical=status)

            sim1 = sim_measures['damerau_levenshtein'](a, b)
            sim8 = sim_measures['jaccard'](a, b)
            sim2 = sim_measures['jaro'](a, b)
            sim3 = sim_measures['jaro_winkler'](a, b)
            sim4 = sim_measures['jaro_winkler'](a[::-1], b[::-1])
            sim11 = sim_measures['monge_elkan'](a, b)
            sim7 = sim_measures['cosine'](a, b)
            sim9 = sim_measures['strike_a_match'](a, b)
            sim12 = sim_measures['soft_jaccard'](a, b)
            if not status: sim5 = sim_measures['sorted_winkler'](a, b)
            sim10 = sim_measures['skipgram'](a, b)
            sim13 = sim_measures['davies'](a, b)
            if status:
                sim14 = sim_measures['tuned_jaro_winkler'](a, b)
                sim15 = sim_measures['tuned_jaro_winkler'](a[::-1], b[::-1])

            if status: f.append([sim1, sim2, sim3, sim4, sim7, sim8, sim9, sim10, sim11, sim12, sim13, sim14, sim15])
            else: f.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])

        if lgm_sims:
            a, b = transform(s1, s2, sorting=True, canonical=True)

            sim1 = self._compute_lgm_sim(a, b, 'damerau_levenshtein')
            sim2 = self._compute_lgm_sim(a, b, 'davies')
            sim3 = self._compute_lgm_sim(a, b, 'skipgram')
            sim4 = self._compute_lgm_sim(a, b, 'soft_jaccard')
            sim5 = self._compute_lgm_sim(a, b, 'strike_a_match')
            sim6 = self._compute_lgm_sim(a, b, 'cosine')
            sim7 = self._compute_lgm_sim(a, b, 'jaccard')
            sim8 = self._compute_lgm_sim(a, b, 'monge_elkan')
            sim9 = self._compute_lgm_sim(a, b, 'jaro_winkler')
            sim10 = self._compute_lgm_sim(a, b, 'jaro')
            sim11 = self._compute_lgm_sim(a, b, 'jaro_winkler_r')
            sim12 = self._compute_lgm_sim(a, b, 'tuned_jaro_winkler')
            sim13 = self._compute_lgm_sim(a, b, 'tuned_jaro_winkler_r')
            sim14, sim15, sim16 = self._compute_lgm_sim_base_scores(a, b, 'damerau_levenshtein')

            f.append([
                sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13, sim14, sim15, sim16
            ])

        f = list(chain.from_iterable(f))

        return f

    def _compute_sorted_features(self, s1, s2):
        return self.compute_features(s1, s2, True, False)

    def _compute_basic_features(self, s1, s2):
        return self.compute_features(s1, s2, False, False)

    @staticmethod
    def _compute_lgm_sim(s1, s2, metric, w_type='avg'):
        baseTerms, mismatchTerms, specialTerms = lgm_sim_split(
            s1, s2, LGMSimVars.per_metric_optValues[metric][w_type][0])

        if metric in ['jaro_winkler_r', 'tuned_jaro_winkler_r']:
            return weighted_sim(
                {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']],
                 'len': baseTerms['len'], 'char_len': baseTerms['char_len']},
                {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']],
                 'len': mismatchTerms['len'], 'char_len': mismatchTerms['char_len']},
                {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']],
                 'len': specialTerms['len'], 'char_len': specialTerms['char_len']},
                metric[:-2], True if w_type == 'avg' else False
            )
        else:
            return weighted_sim(baseTerms, mismatchTerms, specialTerms, metric, True if w_type == 'avg' else False)

    @staticmethod
    def _compute_lgm_sim_base_scores(s1, s2, metric, w_type='avg'):
        base_t, mis_t, special_t = lgm_sim_split(s1, s2, LGMSimVars.per_metric_optValues[metric][w_type][0])
        return score_per_term(base_t, mis_t, special_t, metric)