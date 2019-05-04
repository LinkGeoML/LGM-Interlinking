# -*- coding: utf-8 -*-

import os
import re
from text_unidecode import unidecode

from sim_measures import damerau_levenshtein, jaccard, jaro, jaro_winkler, monge_elkan, cosine, \
    strike_a_match, soft_jaccard, sorted_winkler, permuted_winkler, skipgram, davies, l_jaro_winkler, lsimilarity, \
    avg_lsimilarity, strip_accents


punctuation_regex = re.compile(u'[‘’“”\'"!?;/⧸⁄‹›«»`ʿ,.-]')


def ascii_transliteration_and_punctuation_strip(s):
    # NFKD: first applies a canonical decomposition, i.e., translates each character into its decomposed form.
    # and afterwards apply the compatibility decomposition, i.e. replace all compatibility characters with their
    # equivalents.

    s = unidecode(strip_accents(s.lower()))
    s = punctuation_regex.sub('', s)
    return s


def transform(strA, strB, sorting=False, canonical=False, delimiter=' ', thres=0.55, only_sorting=False):
    a = strA.decode('utf8') #.lower()
    b = strB.decode('utf8') #.lower()

    if canonical:
        a = ascii_transliteration_and_punctuation_strip(a)
        b = ascii_transliteration_and_punctuation_strip(b)

    if sorting:
        tmp_a = a.replace(' ', '')
        tmp_b = b.replace(' ', '')

        if StaticValues.algorithms['damerau_levenshtein'](tmp_a, tmp_b) < thres:
            a = " ".join(sorted_nicely(a.split(delimiter)))
            b = " ".join(sorted_nicely(b.split(delimiter)))
        elif StaticValues.algorithms['damerau_levenshtein'](tmp_a, tmp_b) > StaticValues.algorithms['damerau_levenshtein'](a, b):
            a = tmp_a
            b = tmp_b
    elif only_sorting:
        a = " ".join(sorted_nicely(a.split(delimiter)))
        b = " ".join(sorted_nicely(b.split(delimiter)))

    return a, b


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def getTMabsPath(ds):
    return os.path.join(os.path.abspath('../Toponym-Matching'), 'dataset', ds)


def getRelativePathtoWorking(ds):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ds)


class StaticValues:
    algorithms = {
        'damerau_levenshtein': damerau_levenshtein,
        'davies': davies,
        'skipgram': skipgram,
        'permuted_winkler': permuted_winkler,
        'sorted_winkler': sorted_winkler,
        'soft_jaccard': soft_jaccard,
        'strike_a_match': strike_a_match,
        'cosine': cosine,
        'monge_elkan': monge_elkan,
        'jaro_winkler': jaro_winkler,
        'jaro': jaro,
        'jaccard': jaccard,
        'l_jaro_winkler': l_jaro_winkler,
        'lsimilarity': lsimilarity,
        'avg_lsimilarity': avg_lsimilarity,
    }

    methods = [
        ["Damerau-Levenshtein", {
            'orig': 0.55, 'sorted': 0.60, 'orig_onlylatin': 0.4, 'sorted_onlylatin': 0.55,
            'orig_latin_EU/NA': 0.45, 'sorted_latin_EU/NA': 0.55, 'orig_global': 0.55, 'sorted_global': 0.55
        }],
        ["Jaro", {
            'orig': 0.75, 'sorted': 0.8, 'orig_onlylatin': 0.7, 'sorted_onlylatin': 0.75,
            'orig_latin_EU/NA': 0.7, 'sorted_latin_EU/NA': 0.75, 'orig_global': 0.75, 'sorted_global': 0.75
        }],
        ["Jaro-Winkler", {
            'orig': 0.7, 'sorted': 0.85, 'orig_onlylatin': 0.7, 'sorted_onlylatin': 0.7,
            'orig_latin_EU/NA': 0.7, 'sorted_latin_EU/NA': 0.7, 'orig_global': 0.7, 'sorted_global': 0.7
        }],
        ["Jaro-Winkler reversed", {
            'orig': 0.75, 'sorted': 0.75, 'orig_onlylatin': 0.7, 'sorted_onlylatin': 0.75,
            'orig_latin_EU/NA': 0.7, 'sorted_latin_EU/NA': 0.75, 'orig_global': 0.75, 'sorted_global': 0.75
        }],
        ["Sorted Jaro-Winkler", {
            'orig': 0.7, 'sorted': 0.85, 'orig_onlylatin': 0.70, 'sorted_onlylatin': 0.7,
            'orig_latin_EU/NA': 0.7, 'sorted_latin_EU/NA': 0.7, 'orig_global': 0.7, 'sorted_global': 0.7
        }],
        ["Permuted Jaro-Winkler", {
            'orig': 0.7, 'sorted': 0.7, 'orig_onlylatin': 0.60, 'sorted_onlylatin': 0.6,
            'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.55}],
        ["Cosine N-grams", {
            'orig': 0.4, 'sorted': 0.7, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.45,
            'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.45, 'orig_global': 0.4,
            'sorted_global': 0.45
        }],
        ["Jaccard N-grams", {
            'orig': 0.25, 'sorted': 0.35, 'orig_onlylatin': 0.30, 'sorted_onlylatin': 0.3,
            'orig_latin_EU/NA': 0.3, 'sorted_latin_EU/NA': 0.3, 'orig_global': 0.3, 'sorted_global': 0.3
        }],
        ["Dice bigrams", {
            'orig': 0.5, 'sorted': 0.55, 'orig_onlylatin': 0.4, 'sorted_onlylatin': 0.45,
            'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.5, 'orig_global': 0.5, 'sorted_global': 0.4
        }],
        ["Jaccard skipgrams", {
            'orig': 0.45, 'sorted': 0.55, 'orig_onlylatin': 0.4, 'sorted_onlylatin': 0.45,
            'orig_latin_EU/NA': 0.45, 'sorted_latin_EU/NA': 0.55, 'orig_global': 0.45, 'sorted_global': 0.45
        }],
        ["Monge-Elkan", {
            'orig': 0.7, 'sorted': 0.85, 'orig_onlylatin': 0.7, 'sorted_onlylatin': 0.75,
            'orig_latin_EU/NA': 0.7, 'sorted_latin_EU/NA': 0.75, 'orig_global': 0.7, 'sorted_global': 0.75
        }],
        ["Soft-Jaccard", {
            'orig': 0.6, 'sorted': 0.7, 'orig_onlylatin': 0.55, 'sorted_onlylatin': 0.6,
            'orig_latin_EU/NA': 0.55, 'sorted_latin_EU/NA': 0.6, 'orig_global': 0.6, 'sorted_global': 0.6
        }],
        ["Davis and De Salles", {
            'orig': 0.65, 'sorted': 0.7, 'orig_onlylatin': 0.6, 'sorted_onlylatin': 0.65,
            'orig_latin_EU/NA': 0.65, 'sorted_latin_EU/NA': 0.65, 'orig_global': 0.65, 'sorted_global': 0.65
        }],
        ["LinkGeoML Jaro-Winkler", {
            'orig': 0.7, 'sorted': 0.85, 'orig_onlylatin': 0.70, 'sorted_onlylatin': 0.7,
            'orig_latin_EU/NA': 0.7, 'sorted_latin_EU/NA': 0.7, 'orig_global': 0.7, 'sorted_global': 0.75
        }],
        ["LinkGeoML Jaro-Winkler reversed", {
            'orig': 0.75, 'sorted': 0.75, 'orig_onlylatin': 0.75, 'sorted_onlylatin': 0.75,
            'orig_latin_EU/NA': 0.75, 'sorted_latin_EU/NA': 0.8, 'orig_global': 0.8, 'sorted_global': 0.8
        }],
        ["LinkGeoML Similarity", {
            'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
            'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.55, 'orig_global': 0.55, 'sorted_global': 0.55
        }],
        ["Avg LinkGeoML Similarity", {
            'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
            'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.55, 'orig_global': 0.55, 'sorted_global': 0.55
        }],

        ["Jaro LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
           'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.75, 'orig_global': 0.55, 'sorted_global': 0.75
          }],
        ["Jaro Avg LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.75, 'orig_global': 0.55, 'sorted_global': 0.75
          }],
        ["Jaro-Winkler LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.7, 'orig_global': 0.55, 'sorted_global': 0.7
          }],
        ["Jaro-Winkler Avg LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.7, 'orig_global': 0.55, 'sorted_global': 0.7
          }],
        ["Jaro-Winkler reversed LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.7, 'orig_global': 0.55, 'sorted_global': 0.7
          }],
        ["Jaro-Winkler reversed Avg LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.75, 'orig_global': 0.55, 'sorted_global': 0.75
          }],
        ["Cosine NGrams LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.45, 'orig_global': 0.55, 'sorted_global': 0.45
          }],
        ["Cosine NGrams Avg LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.45, 'orig_global': 0.55, 'sorted_global': 0.45
          }],
        ["Jaccard NGrams LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.35, 'orig_global': 0.55, 'sorted_global': 0.35
          }],
        ["Jaccard NGrams Avg LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.35, 'orig_global': 0.55, 'sorted_global': 0.35
          }],
        ["Dice BiGrams LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.5, 'orig_global': 0.55, 'sorted_global': 0.5
          }],
        ["Dice BiGrams Avg LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.5, 'orig_global': 0.55, 'sorted_global': 0.5
          }],
        ["Jaccard Skipgrams LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.45, 'orig_global': 0.55, 'sorted_global': 0.45
          }],
        ["Jaccard Skipgrams Avg LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.45, 'orig_global': 0.55, 'sorted_global': 0.45
          }],
        ["Monge–Elkan LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.7, 'orig_global': 0.55, 'sorted_global': 0.7
          }],
        ["Monge–Elkan Avg LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.7, 'orig_global': 0.55, 'sorted_global': 0.7
          }],
        ["Soft–Jaccard LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.55, 'orig_global': 0.55, 'sorted_global': 0.55
          }],
        ["Soft–Jaccard Avg LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.6, 'orig_global': 0.55, 'sorted_global': 0.6
          }],
        ["Davis and De Salles LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.65, 'orig_global': 0.55, 'sorted_global': 0.65
          }],
        ["Davis and De Salles Avg LinkGeoML Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.65, 'orig_global': 0.55, 'sorted_global': 0.65
          }],
        ["LinkGeoML Jaro - Winkler Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.7, 'orig_global': 0.55, 'sorted_global': 0.7
          }],
        ["LinkGeoML Jaro - Winkler Avg Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.7, 'orig_global': 0.55, 'sorted_global': 0.7
          }],
        ["LinkGeoML Jaro - Winkler reversed Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.7, 'orig_global': 0.55, 'sorted_global': 0.7
          }],
        ["LinkGeoML Jaro - Winkler reversed Avg Similarity",
         {'orig': 0.4, 'sorted': 0.35, 'orig_onlylatin': 0.35, 'sorted_onlylatin': 0.35,
          'orig_latin_EU/NA': 0.4, 'sorted_latin_EU/NA': 0.75, 'orig_global': 0.55, 'sorted_global': 0.75
          }],
    ]

    nameIDs = {
        'damerau_levenshtein': 0,
        'davies': 12,
        'skipgram': 9,
        'permuted_winkler': 5,
        'sorted_winkler': 4,
        'soft_jaccard': 11,
        'strike_a_match': 8,
        'cosine': 6,
        'monge_elkan': 10,
        'jaro_winkler': 2,
        'jaro': 1,
        'jaccard': 7,
        'l_jaro_winkler': 13,
        'lsimilarity': 15,
        'avg_lsimilarity': 16,
    }

    methods_as_saved = [
        "damerau_levenshtein",
        "jaccard",
        "jaro",
        "jaro_winkler",
        "jaro_winkler_reversed",
        "monge_elkan",
        "cosine",
        "strike_a_match",
        "soft_jaccard",
        "sorted_winkler",
        "skipgram",
        "davies",
        "l_jaro_winkler",
        "l_jaro_winkler_reversed",
        "lsimilarity",
    ]

    classifiers_abbr = {
        'lsvm': 0,
        'dt': 1,
        'rf': 2,
        'nn': 3,
        'nb': 4,
        'et': 5,
        'xgboost': 6,
    }

    classifiers = [
        "Linear SVM",
        "Decision Tree", "Random Forest", "Neural Net", "Naive Bayes",
        "ExtraTreeClassifier", "XGBOOST"
    ]

    featureColumns = [
        "Damerau-Levenshtein",
        "Jaro",
        "Jaro-Winkler",
        "Jaro-Winkler reversed",
        "Sorted Jaro-Winkler",
        # "Permuted Jaro-Winkler",
        "Cosine N-grams",
        "Jaccard N-grams",
        "Dice bigrams",
        "Jaccard skipgrams",
        "Monge-Elkan",
        "Soft-Jaccard",
        "Davis and De Salles",
        "Damerau-Levenshtein Sorted",
        "Jaro Sorted",
        "Jaro-Winkler Sorted",
        "Jaro-Winkler reversed Sorted",
        # "Sorted Jaro-Winkler Sorted",
        # "Permuted Jaro-Winkler Sorted",
        "Cosine N-grams Sorted",
        "Jaccard N-grams Sorted",
        "Dice bigrams Sorted",
        "Jaccard skipgrams Sorted",
        "Monge-Elkan Sorted",
        "Soft-Jaccard Sorted",
        "Davis and De Salles Sorted",
        "LinkGeoML Jaro-Winkler",
        "LinkGeoML Jaro-Winkler reversed",
        # "LSimilarity",
        "LSimilarity_wavg",
        # "LSimilarity_davies",
        # "LSimilarity_skipgram",
        # "LSimilarity_soft_jaccard",
        # "LSimilarity_strike_a_match",
        # "LSimilarity_cosine",
        # "LSimilarity_monge_elkan",
        # "LSimilarity_jaro_winkler",
        # "LSimilarity_jaro",
        # "LSimilarity_jaro_winkler_reversed",
        "LSimilarity_davies_wavg",
        "LSimilarity_skipgram_wavg",
        "LSimilarity_soft_jaccard_wavg",
        "LSimilarity_strike_a_match_wavg",
        "LSimilarity_cosine_wavg",
        "LSimilarity_jaccard_wavg",
        "LSimilarity_monge_elkan_wavg",
        "LSimilarity_jaro_winkler_wavg",
        "LSimilarity_jaro_wavg",
        "LSimilarity_jaro_winkler_reversed_wavg",
        "LSimilarity_l_jaro_winkler_wavg",
        "LSimilarity_l_jaro_winkler_reversed_wavg",
        # "LSimilarity_baseScore",
        # "LSimilarity_mismatchScore",
        # "LSimilarity_specialScore",
        "Avg LSimilarity_baseScore",
        "Avg LSimilarity_mismatchScore",
        "Avg LSimilarity_specialScore",
        # non metric features
        # "contains_str1",
        # "contains_str2",
        # "WordsNo_str1",
        # "WordsNo_str2",
        # "dashed_str1",
        # "dashed_str2",
        # "hasFreqTerm_str1",
        # "hasFreqTerm_str2",
        # "posOfHigherSim_str1_start",
        # "posOfHigherSim_str1_middle",
        # "posOfHigherSim_str1_end",
        # "posOfHigherSim_str2_start",
        # "posOfHigherSim_str2_middle",
        # "posOfHigherSim_str2_end",
    ]

    MetricOptimalValues = {
        'latin': {
            # Only latin dataset 100k lines
            'damerau_levenshtein': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.5, 0.1, 0.4]]},
            'jaro': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.7, 0.1, 0.2]]},
            'jaro_winkler': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.6, 0.1, 0.3]]},
            'jaro_winkler_r': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.7, 0.1, 0.2]]},
            'permuted_winkler': [],
            'sorted_winkler': [],
            'cosine': {'simple': [0.6, [0.6, 0.2, 0.2]], 'avg': [0.8, [0.4, 0.2, 0.4]]},
            'jaccard': {'simple': [0.6, [0.6, 0.1, 0.3]], 'avg': [0.8, [0.334, 0.333, 0.333]]},
            'strike_a_match': {'simple': [0.6, [0.6, 0.1, 0.3]], 'avg': [0.8, [0.4, 0.2, 0.4]]},
            'skipgram': {'simple': [0.6, [0.6, 0.2, 0.2]], 'avg': [0.8, [0.334, 0.333, 0.333]]},
            'monge_elkan': {'simple': [0.6, [0.7, 0.2, 0.1]], 'avg': [0.8, [0.6, 0.1, 0.3]]},
            'soft_jaccard': {'simple': [0.8, [0.6, 0.1, 0.3]], 'avg': [0.8, [0.5, 0.1, 0.4]]},
            'davies': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.6, 0.1, 0.3]]},
            'l_jaro_winkler': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.6, 0.1, 0.3]]},
            'l_jaro_winkler_r': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.7, 0.1, 0.2]]},
        },
        'global': {
            'damerau_levenshtein': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
            'jaro': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
            'jaro_winkler': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'jaro_winkler_r': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
            'permuted_winkler': [],
            'sorted_winkler': [],
            'cosine': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'jaccard': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'strike_a_match': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.65, [0.4, 0.5, 0.1]]},
            'skipgram': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'monge_elkan': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'soft_jaccard': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.7, [0.4, 0.5, 0.1]]},
            'davies': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.7, [0.4, 0.5, 0.1]]},
            'l_jaro_winkler': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'l_jaro_winkler_r': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
        }
    }
