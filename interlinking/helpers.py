# coding= utf-8
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import os
import re
from text_unidecode import unidecode
import __main__
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pycountry
from langdetect import detect, lang_detect_exception

from interlinking.sim_measures import strip_accents, sim_measures
from interlinking import config


punctuation_regex = re.compile(u'[‘’“”\'"!?;/⧸⁄‹›«»`ʿ,.-]')


def ascii_transliteration_and_punctuation_strip(s):
    # NFKD: first applies a canonical decomposition, i.e., translates each character into its decomposed form.
    # and afterwards apply the compatibility decomposition, i.e. replace all compatibility characters with their
    # equivalents.

    s = unidecode(strip_accents(s.lower()))
    s = punctuation_regex.sub('', s)
    return s


def transform(s1, s2, sorting=False, canonical=False, delimiter=' ', thres=config.sort_thres, simple_sorting=False):
    # a = six.text_type(s1) #.lower()
    a = s1
    b = s2

    if canonical:
        a = ascii_transliteration_and_punctuation_strip(a)
        b = ascii_transliteration_and_punctuation_strip(b)

    if simple_sorting:
        a = " ".join(sorted_nicely(a.split(delimiter)))
        b = " ".join(sorted_nicely(b.split(delimiter)))
    elif sorting:
        tmp_a = a.replace(' ', '')
        tmp_b = b.replace(' ', '')

        if sim_measures['damerau_levenshtein'](tmp_a, tmp_b) < thres:
            a = " ".join(sorted_nicely(a.split(delimiter)))
            b = " ".join(sorted_nicely(b.split(delimiter)))
        elif sim_measures['damerau_levenshtein'](tmp_a, tmp_b) > sim_measures['damerau_levenshtein'](a, b):
            a = tmp_a
            b = tmp_b

    return a, b


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_langnm(s, lang_detect=False):
    lname = 'english'
    try:
        lname = pycountry.languages.get(alpha_2=detect(s)).name.lower() if lang_detect else 'english'
    except lang_detect_exception.LangDetectException as e:
        print(e)

    return lname


# Clean the string from stopwords based on language detections feature
# Returned values #1: non-stopped words, #2: stopped words
def normalize_str(s, lang_detect=False):
    lname = get_langnm(s, lang_detect)
    stemmer = SnowballStemmer(lname)
    tokens = word_tokenize(s)
    # words = [word.lower() for word in tokens if word.isalpha()]
    stopwords_set = set(stopwords.words(lname))

    stopped_words = set(filter(lambda token: token in stopwords_set, tokens))
    filtered_words = list(filter(lambda token: token not in stopped_words, tokens))
    filtered_stemmed_words = list(map(lambda token: stemmer.stem(token), filtered_words))

    return filtered_words, filtered_stemmed_words, stopped_words


def getBasePath():
    return os.path.abspath(os.path.dirname(__main__.__file__))


def getRelativePathtoWorking(ds):
    return os.path.join(getBasePath(), 'data', ds)


class StaticValues:
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

    opt_values = {
        'latin': {
            # Only latin dataset 100k lines
            'damerau_levenshtein': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.5, 0.1, 0.4]]},
            'jaro': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.7, 0.1, 0.2]]},
            'jaro_winkler': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.6, 0.1, 0.3]]},
            'jaro_winkler_r': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.7, 0.1, 0.2]]},
            # 'permuted_winkler': [],
            # 'sorted_winkler': [],
            'cosine': {'simple': [0.6, [0.6, 0.2, 0.2]], 'avg': [0.8, [0.4, 0.2, 0.4]]},
            'jaccard': {'simple': [0.6, [0.6, 0.1, 0.3]], 'avg': [0.8, [0.334, 0.333, 0.333]]},
            'strike_a_match': {'simple': [0.6, [0.6, 0.1, 0.3]], 'avg': [0.8, [0.4, 0.2, 0.4]]},
            'skipgram': {'simple': [0.6, [0.6, 0.2, 0.2]], 'avg': [0.8, [0.334, 0.333, 0.333]]},
            'monge_elkan': {'simple': [0.6, [0.7, 0.2, 0.1]], 'avg': [0.8, [0.6, 0.1, 0.3]]},
            'soft_jaccard': {'simple': [0.8, [0.6, 0.1, 0.3]], 'avg': [0.8, [0.5, 0.1, 0.4]]},
            'davies': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.6, 0.1, 0.3]]},
            'tuned_jaro_winkler': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.6, 0.1, 0.3]]},
            'tuned_jaro_winkler_r': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.7, 0.1, 0.2]]},
        },
        'global': {
            'damerau_levenshtein': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
            'jaro': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
            'jaro_winkler': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'jaro_winkler_r': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
            # 'permuted_winkler': [],
            # 'sorted_winkler': [],
            'cosine': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'jaccard': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'strike_a_match': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.65, [0.4, 0.5, 0.1]]},
            'skipgram': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'monge_elkan': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'soft_jaccard': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.7, [0.4, 0.5, 0.1]]},
            'davies': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.7, [0.4, 0.5, 0.1]]},
            'tuned_jaro_winkler': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'tuned_jaro_winkler_r': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
        }
    }

    sim_metrics = {
        'damerau_levenshtein': ['basic', 'sorted', 'lgm'],
        'jaro': ['basic', 'sorted', 'lgm'],
        'jaro_winkler': ['basic', 'sorted', 'lgm'],
        'jaro_winkler_reversed': ['basic', 'sorted', 'lgm'],
        'sorted_winkler': ['basic'],
        'permuted_winkler': [],
        'cosine': ['basic', 'sorted', 'lgm'],
        'jaccard': ['basic', 'sorted', 'lgm'],
        'strike_a_match': ['basic', 'sorted', 'lgm'],
        'skipgram': ['basic', 'sorted', 'lgm'],
        'monge_elkan': ['basic', 'sorted', 'lgm'],
        'soft_jaccard': ['basic', 'sorted', 'lgm'],
        'davies': ['basic', 'sorted', 'lgm'],
        'tuned_jaro_winkler': ['basic', 'sorted', 'lgm'],
        'tuned_jaro_winkler_reversed': ['basic', 'sorted', 'lgm'],
        # 'lgm_sim': ['lgm'],
        # 'avg_lgm_sim': ['lgm'],
    }
