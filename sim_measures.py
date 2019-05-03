#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import csv
import os.path
import sys
import math
import random
import jellyfish
import pyxdameraulevenshtein
import numpy as np
import itertools
import unicodedata
from alphabet_detector import AlphabetDetector
import re
import pycountry_convert


fields = ["geonameid",
          "name",
          "asciiname",
          "alternatenames",
          "latitude",
          "longitude",
          "feature class",
          "feature_code",
          "country_code",
          "cc2",
          "admin1_code",
          "admin2_code",
          "admin3_code",
          "admin4_code",
          "population",
          "elevation",
          "dem",
          "timezone",
          "modification_date"]


def check_alphabet(str, alphabet, only=True):
    ad = AlphabetDetector()
    uni_string = unicode(str, "utf-8")
    if only:
        return ad.only_alphabet_chars(uni_string, alphabet.upper())
    else:
        for i in uni_string:
            if ad.is_in_alphabet(i, alphabet.upper()): return True
        return False


def detect_alphabet(str):
    ad = AlphabetDetector()
    uni_string = unicode(str, "utf-8")
    ab = ad.detect_alphabet(uni_string)
    if "CYRILLIC" in ab:
        return "CYRILLIC"
    return ab.pop() if len(ab) != 0 else 'UND'


# The geonames dataset can be obtained from http://download.geonames.org/export/dump/allCountries.zip
def build_dataset_from_geonames(output='dataset-unfiltered.txt', only_latin=False):
    # remove dupls after running this script
    # cat -n dataset-string-similarity.txt | sort -k2 -k1n | uniq -f1 | sort -nk1,1 | cut -f2-
    datasets = ['allCountries.txt', 'cities5000.txt', 'cities500.txt']

    csv.field_size_limit(sys.maxsize)
    lastname = None
    lastname2 = None
    lastid = None
    # country = None
    skip = random.randint(10, 100)
    file = open(output, "w+")
    max_no_attempts = 1000

    for input in datasets:
        if not os.path.isfile(input):
            print("File {0} does not exist".format(input))
            continue

        print("Working on dataset {}...".format(input))
        with open(input) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=fields, delimiter='\t')
            for row in reader:
                skip = skip - 1
                if skip > 0: continue
                names = set([name.strip() for name in ("" + row['alternatenames']).split(",") if len(name.strip()) > 2])

                # remove non LATIN names
                if only_latin:
                    for n in list(names):
                        if not check_alphabet(n, 'LATIN'): names.remove(n)
                        else:
                            try:
                                if pycountry_convert.country_alpha2_to_continent_code(row['country_code']) not in \
                                        ['EU', 'NA'] or row['country_code'] in ['RU']:
                                    names.remove(n)
                            except KeyError as e:
                                names.remove(n)
                                print(e.message)

                if len(names) < 5: continue
                lastid = row['geonameid']
                firstcountry = row['country_code']
                lastname = random.sample(names, 1)[0]
                lastname2 = random.sample(names, 1)[0]
                while True:
                    lastname2 = random.sample(names, 1)[0]
                    if not (lastname2.lower() == lastname.lower()): break
        with open(input) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=fields, delimiter='\t')
            for row in reader:
                names = set([name.strip() for name in ("" + row['alternatenames']).split(",") if len(name.strip()) > 2])
                if len(row['name'].strip()) > 2: names.add(row['name'].strip())
                if len(unicode(row['asciiname'], "utf-8").strip()) > 2: names.add(row['asciiname'].strip())

                # nonLATIN = False
                if only_latin:
                    for n in list(names):
                        if not check_alphabet(n, 'LATIN'): names.remove(n)
                        else:
                            try:
                                if pycountry_convert.country_alpha2_to_continent_code(row['country_code']) not in \
                                        ['EU', 'NA'] or row['country_code'] in ['RU']:
                                    names.remove(n)
                            except KeyError as e:
                                names.remove(n)
                                print(e.message)

                if len(names) < 3: continue
                id = row['geonameid']
                country = row['country_code']
                randomname1 = random.sample(names, 1)[0]
                randomname3 = random.sample(names, 1)[0]
                randomname5 = random.sample(names, 1)[0]
                while True:
                    randomname2 = random.sample(names, 1)[0]
                    if not (randomname1.lower() == randomname2.lower()): break
                attempts = max_no_attempts
                while attempts > 0:
                    attempts = attempts - 1
                    randomname3 = random.sample(names, 1)[0]
                    if lastname is None or (
                            jaccard(randomname3, lastname) > 0.0 and not (randomname3.lower() == lastname.lower())): break
                    if damerau_levenshtein(randomname3, lastname) == 0.0 and random.random() < 0.5: break
                if attempts <= 0:
                    auxl = lastname
                    lastname = lastname2
                    lastname2 = auxl
                    attempts = max_no_attempts
                    while attempts > 0:
                        attempts = attempts - 1
                        randomname3 = random.sample(names, 1)[0]
                        if lastname is None or (jaccard(randomname3, lastname) > 0.0 and not (
                                randomname3.lower() == lastname.lower())): break
                        if damerau_levenshtein(randomname3, lastname) == 0.0 and random.random() < 0.5: break
                if attempts <= 0:
                    lastid = id
                    lastname = randomname1
                    lastname2 = randomname2
                    firstcountry = row['country_code']
                    continue
                if randomname1 is None or randomname2 is None or id is None or country is None:
                    continue
                # print randomname1 + "\t" + randomname2 + "\tTRUE\t" + id + "\t" + id + "\t" + detect_alphabet(
                #     randomname1) + "\t" + detect_alphabet(randomname2) + "\t" + country + "\t" + country
                file.write(randomname1 + "\t" + randomname2 + "\tTRUE\t" + id + "\t" + id + "\t" + detect_alphabet(
                    randomname1) + "\t" + detect_alphabet(randomname2) + "\t" + country + "\t" + country + "\n")
                if not (lastid is None):
                    # print lastname + "\t" + randomname3 + "\tFALSE\t" + lastid + "\t" + id + "\t" + detect_alphabet(
                    # lastname) + "\t" + detect_alphabet(randomname3) + "\t" + firstcountry + "\t" + country
                    file.write(lastname + "\t" + randomname3 + "\tFALSE\t" + lastid + "\t" + id + "\t" + detect_alphabet(
                    lastname) + "\t" + detect_alphabet(randomname3) + "\t" + firstcountry + "\t" + country + "\n")
                lastname = randomname1
                if len(names) < 5:
                    lastid = id
                    lastname2 = randomname2
                    firstcountry = country
                    continue
                curr_attempt = 0
                while True:
                    randomname4 = random.sample(names, 1)[0]
                    if not (randomname4.lower() == randomname1.lower()) and not (
                            randomname4.lower() == randomname2.lower()): break
                    curr_attempt += 1
                    if curr_attempt > max_no_attempts: break
                if curr_attempt > max_no_attempts:
                    print("Failed to find alternative names...")
                    lastid = id
                    lastname2 = randomname2
                    firstcountry = country
                    continue

                attempts = max_no_attempts
                while attempts > 0:
                    attempts = attempts - 1
                    randomname5 = random.sample(names, 1)[0]
                    if lastname2 is None or (jaccard(randomname5, lastname2) > 0.0 and not (
                            randomname5.lower() == lastname2.lower()) and not (
                            randomname5.lower() == randomname3.lower())): break
                    if damerau_levenshtein(randomname5, lastname2) == 0.0 and random.random() < 0.5: break
                if attempts > 0:
                    aux = random.sample([randomname1, randomname2], 1)[0]
                    # print randomname4 + "\t" + aux + "\tTRUE\t" + id + "\t" + id + "\t" + detect_alphabet(
                    #     randomname4) + "\t" + detect_alphabet(aux) + "\t" + country + "\t" + country
                    file.write(randomname4 + "\t" + aux + "\tTRUE\t" + id + "\t" + id + "\t" + detect_alphabet(
                        randomname4) + "\t" + detect_alphabet(aux) + "\t" + country + "\t" + country + "\n")
                    if not (lastid is None):
                        # print lastname2 + "\t" + randomname5 + "\tFALSE\t" + lastid + "\t" + id + "\t" + detect_alphabet(
                        # lastname2) + "\t" + detect_alphabet(randomname5) + "\t" + firstcountry + "\t" + country
                        file.write(lastname2 + "\t" + randomname5 + "\tFALSE\t" + lastid + "\t" + id + "\t" + detect_alphabet(
                            lastname2) + "\t" + detect_alphabet(randomname5) + "\t" + firstcountry + "\t" + country + "\n")
                lastname2 = random.sample([randomname2, randomname4], 1)[0]
                lastid = id
    file.close()


def filter_dataset(input='dataset-unfiltered.txt', num_instances=2500000):
    pos = []
    neg = []
    file = open("dataset-string-similarity.txt", "w+")
    print("Filtering for {0}...".format(num_instances * 2))
    for line in open(input):
        splitted = line.split('\t')
        if not (splitted[2] == "TRUE" or splitted[2] == "FALSE") or \
                not (len(unicode(splitted[7], "utf-8")) == 2 and len(unicode(splitted[8], "utf-8")) == 3) or \
                not (splitted[5] != "UND" and splitted[6] != "UND") or \
                not (splitted[3].isdigit() and splitted[4].isdigit()) or \
                len(splitted) != 9 or \
                len(unicode(splitted[1], "utf-8")) < 3 or \
                len(unicode(splitted[0], "utf-8")) < 3:
            continue
        if '\tTRUE\t' in line:
            pos.append(line)
        else:
            neg.append(line)
    pos = random.sample(pos, len(pos))
    neg = random.sample(neg, len(neg))
    for i in range(min(num_instances, len(pos), len(neg))):
        file.write(pos[i])
        file.write(neg[i])
    print("Filtering ended with {0}.".format(min(num_instances * 2, len(pos) + len(neg))))
    file.close()


def skipgrams(sequence, n, k):
    sequence = " " + sequence + " "
    res = []
    for ngram in {sequence[i:i + n + k] for i in range(len(sequence) - (n + k - 1))}:
        if k == 0:
            res.append(ngram)
        else:
            res.append(ngram[0:1] + ngram[k + 1:len(ngram)])
    return res


def skipgram(str1, str2):
    a1 = set(skipgrams(str1, 2, 0))
    a2 = set(skipgrams(str1, 2, 1) + skipgrams(str1, 2, 2))
    b1 = set(skipgrams(str2, 2, 0))
    b2 = set(skipgrams(str2, 2, 1) + skipgrams(str1, 2, 2))
    c1 = a1.intersection(b1)
    c2 = a2.intersection(b2)
    d1 = a1.union(b1)
    d2 = a2.union(b2)
    try:
        return float(len(c1) + len(c2)) / float(len(d1) + len(d2))
    except:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def davies(str1, str2):
    a = strip_accents(str1.lower()).replace(u'-', u' ').split(' ')
    b = strip_accents(str2.lower()).replace(u'-', u' ').split(' ')
    for i in range(len(a)):
        if len(a[i]) > 1 or not (a[i].endswith(u'.')): continue
        replacement = len(str2)
        for j in range(len(b)):
            if b[j].startswith(a[i].replace(u'.', '')):
                if len(b[j]) < replacement:
                    a[i] = b[j]
                    replacement = len(b[j])
    for i in range(len(b)):
        if len(b[i]) > 1 or not (b[i].endswith(u'.')): continue
        replacement = len(str1)
        for j in range(len(a)):
            if a[j].startswith(b[i].replace(u'.', '')):
                if len(a[j]) < replacement:
                    b[i] = a[j]
                    replacement = len(a[j])
    a = set(a)
    b = set(b)
    aux1 = sorted_winkler(str1, str2)
    intersection_length = (sum(max(jaro_winkler(i, j) for j in b) for i in a) + sum(
        max(jaro_winkler(i, j) for j in a) for i in b)) / 2.0
    aux2 = float(intersection_length) / (len(a) + len(b) - intersection_length)
    return (aux1 + aux2) / 2.0


def cosine(str1, str2):
    str1 = " " + str1 + " "
    str2 = " " + str2 + " "
    x = list(itertools.chain.from_iterable([[str1[i:i + n] for i in range(len(str1) - (n - 1))] for n in [2, 3]]))
    y = list(itertools.chain.from_iterable([[str2[i:i + n] for i in range(len(str2) - (n - 1))] for n in [2, 3]]))
    vectorIndex = {}
    offset = 0
    for offset, word in enumerate(set(x + y)): vectorIndex[word] = offset
    vector = np.zeros(len(vectorIndex))
    for word in x: vector[vectorIndex[word]] += 1
    x = vector
    vector = np.zeros(len(vectorIndex))
    for word in y: vector[vectorIndex[word]] += 1
    y = vector
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = math.sqrt(sum([a * a for a in x])) * math.sqrt(sum([a * a for a in y]))
    try:
        return numerator / denominator
    except:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def damerau_levenshtein(str1, str2):
    aux = pyxdameraulevenshtein.normalized_damerau_levenshtein_distance(str1, str2)
    return 1.0 - aux


def jaro(str1, str2):
    aux = jellyfish.jaro_distance(str1, str2)
    return aux


def jaro_winkler(str1, str2):
    aux = jellyfish.jaro_winkler(str1, str2)
    return aux


def monge_elkan_aux(str1, str2):
    cummax = 0
    for ws in str1.split(" "):
        maxscore = 0
        for wt in str2.split(" "):
            maxscore = max(maxscore, jaro_winkler(ws, wt))
        cummax += maxscore
    return cummax / len(str1.split(" "))


def monge_elkan(str1, str2):
    return (monge_elkan_aux(str1, str2) + monge_elkan_aux(str2, str1)) / 2.0


# http://www.catalysoft.com/articles/StrikeAMatch.html
def strike_a_match(str1, str2):
    pairs1 = {str1[i:i + 2] for i in xrange(len(str1) - 1)}
    pairs2 = {str2[i:i + 2] for i in xrange(len(str2) - 1)}
    union = len(pairs1) + len(pairs2)
    hit_count = 0
    for x in pairs1:
        for y in pairs2:
            if x == y:
                hit_count += 1
                break
    try:
        return (2.0 * hit_count) / union
    except:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def jaccard(str1, str2):
    str1 = " " + str1 + " "
    str2 = " " + str2 + " "
    a = list(itertools.chain.from_iterable([[str1[i:i + n] for i in range(len(str1) - (n - 1))] for n in [2, 3]]))
    b = list(itertools.chain.from_iterable([[str2[i:i + n] for i in range(len(str2) - (n - 1))] for n in [2, 3]]))
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    try:
        return float(len(c)) / (float((len(a) + len(b) - len(c))))
    except:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def soft_jaccard(str1, str2):
    a = set(str1.split(" "))
    b = set(str2.split(" "))
    intersection_length = (sum(max(jaro_winkler(i, j) for j in b) for i in a) + sum(
        max(jaro_winkler(i, j) for j in a) for i in b)) / 2.0
    return float(intersection_length) / (len(a) + len(b) - intersection_length)


def sorted_winkler(str1, str2):
    a = sorted(str1.split(" "))
    b = sorted(str2.split(" "))
    a = " ".join(a)
    b = " ".join(b)
    return jaro_winkler(a, b)


def permuted_winkler(str1, str2):
    a = str1.split(" ")
    b = str2.split(" ")
    if len(a) > 5: a = a[0:5] + [u''.join(a[5:])]
    if len(b) > 5: b = b[0:5] + [u''.join(b[5:])]
    lastscore = 0.0
    for a in itertools.permutations(a):
        for b in itertools.permutations(b):
            sa = u' '.join(a)
            sb = u' '.join(b)
            score = jaro_winkler(sa, sb)
            if score > lastscore: lastscore = score
    return lastscore


def _check_type(s):
    if not isinstance(s, unicode):
        raise TypeError('expected unicode, got %s' % type(s).__name__)


def _jaro_winkler(ying, yang, long_tolerance, winklerize):
    _check_type(ying)
    _check_type(yang)

    ying_len = len(ying)
    yang_len = len(yang)

    if not ying_len or not yang_len:
        return 0.0

    min_len = max(ying_len, yang_len)
    search_range = (min_len // 2) - 1
    if search_range < 0:
        search_range = 0

    ying_flags = [False]*ying_len
    yang_flags = [False]*yang_len

    # looking only within search range, count & flag matched pairs
    common_chars = 0
    for i, ying_ch in enumerate(ying):
        low = i - search_range if i > search_range else 0
        hi = i + search_range if i + search_range < yang_len else yang_len - 1
        for j in range(low, hi+1):
            if not yang_flags[j] and yang[j] == ying_ch:
                ying_flags[i] = yang_flags[j] = True
                common_chars += 1
                break

    # short circuit if no characters match
    if not common_chars:
        return 0.0

    # count transpositions
    k = trans_count = 0
    for i, ying_f in enumerate(ying_flags):
        if ying_f:
            for j in range(k, yang_len):
                if yang_flags[j]:
                    k = j + 1
                    break
            if ying[i] != yang[j]:
                trans_count += 1
    trans_count /= 2

    # adjust for similarities in nonmatched characters
    common_chars = float(common_chars)
    weight = ((common_chars/ying_len + common_chars/yang_len +
              (common_chars-trans_count) / common_chars)) / 3

    # winkler modification: continue to boost if strings are similar
    if winklerize and weight > 0.7 and ying_len > 3 and yang_len > 3:
        # adjust for up to first 4 chars in common
        j = min(min_len, 4)
        i = 0
        k = 0
        mismatch_is_allowed = False
        if ying_len > 4 and yang_len > 4: mismatch_is_allowed = True
        mismatch_is_checked = False
        while i < j and k < j:
            if ying[i] == yang[k] and ying[i]:
                i += 1
                k += 1
            elif mismatch_is_allowed and not mismatch_is_checked:
                if ying[i] == yang[k+1]:
                    i += 1
                    k += 2
                    j = min(j + 1, min_len)
                elif ying[i+1] == yang[k]:
                    i += 2
                    k += 1
                    j = min(j + 1, min_len)
                elif ying[i+1] == yang[k+1]:
                    i += 2
                    k += 2
                    j = min(j + 1, min_len)

                mismatch_is_checked = True
            else: break

        if i or k:
            weight += min(i, k) * 0.1 * (1.0 - weight)

        # optionally adjust for long strings
        # after agreeing beginning chars, at least two or more must agree and
        # agreed characters must be > half of remaining characters
        if (long_tolerance and min_len > 4 and common_chars > i+1 and
                2 * common_chars >= min_len + i):
            weight += ((1.0 - weight) * (float(common_chars-i-1) / float(ying_len+yang_len-i*2+2)))

    return weight


def l_jaro_winkler(s1, s2, long_tolerance=False):
    return _jaro_winkler(s1, s2, long_tolerance, True)


class LSimilarityVars:
    freq_ngrams = {'tokens': set(), 'chars': set()}
    lsimilarity_weights = []

    per_metric_optimal_values = {
        # # Only latin dataset 100k lines
        # 'damerau_levenshtein': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.5, 0.1, 0.4]]},
        # 'jaro': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.7, 0.1, 0.2]]},
        # 'jaro_winkler': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.6, 0.1, 0.3]]},
        # 'jaro_winkler_r': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.7, 0.1, 0.2]]},
        # 'permuted_winkler': [],
        # 'sorted_winkler': [],
        # 'cosine': {'simple': [0.6, [0.6, 0.2, 0.2]], 'avg': [0.8, [0.4, 0.2, 0.4]]},
        # 'jaccard': {'simple': [0.6, [0.6, 0.1, 0.3]], 'avg': [0.8, [0.334, 0.333, 0.333]]},
        # 'strike_a_match': {'simple': [0.6, [0.6, 0.1, 0.3]], 'avg': [0.8, [0.4, 0.2, 0.4]]},
        # 'skipgram': {'simple': [0.6, [0.6, 0.2, 0.2]], 'avg': [0.8, [0.334, 0.333, 0.333]]},
        # 'monge_elkan': {'simple': [0.6, [0.7, 0.2, 0.1]], 'avg': [0.8, [0.6, 0.1, 0.3]]},
        # 'soft_jaccard': {'simple': [0.8, [0.6, 0.1, 0.3]], 'avg': [0.8, [0.5, 0.1, 0.4]]},
        # 'davies': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.6, 0.1, 0.3]]},
        # 'l_jaro_winkler': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.6, 0.1, 0.3]]},
        # 'l_jaro_winkler_r': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.7, 0.1, 0.2]]},

        # 'damerau_levenshtein': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
        # 'jaro': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
        # 'jaro_winkler': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
        # 'jaro_winkler_r': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
        # 'permuted_winkler': [],
        # 'sorted_winkler': [],
        # 'cosine': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
        # 'jaccard': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
        # 'strike_a_match': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.65, [0.4, 0.5, 0.1]]},
        # 'skipgram': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
        # 'monge_elkan': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
        # 'soft_jaccard': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.7, [0.4, 0.5, 0.1]]},
        # 'davies': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.7, [0.4, 0.5, 0.1]]},
        # 'l_jaro_winkler': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
        # 'l_jaro_winkler_r': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
    }


def termsim_split(a, b, thres):
    base = {'a': [], 'b': [], 'len': 0}
    mis = {'a': [], 'b': [], 'len': 0}

    ls1, ls2 = a.split(), b.split()
    while ls1 and ls2:
        str1, str2 = ls1[0], ls2[0]
        if jaro_winkler(str1[::-1], str2[::-1]) >= thres:
            base['a'].append(str1)
            ls1.pop(0)

            base['b'].append(str2)
            ls2.pop(0)
        else:
            if str1 < str2:
                mis['a'].append(str1)
                ls1.pop(0)
            else:
                mis['b'].append(str2)
                ls2.pop(0)

    mis['a'].extend(ls1)
    mis['b'].extend(ls2)

    base['len'] = len(base['a']) + len(base['b'])
    base['char_len'] = sum(len(s) for s in base['a']) + sum(len(s) for s in base['b'])
    mis['len'] = len(mis['a']) + len(mis['b'])
    mis['char_len'] = sum(len(s) for s in mis['a']) + sum(len(s) for s in mis['b'])

    return base, mis


def lsimilarity_terms(str1, str2, term_split_thres):
    specialTerms = dict(a=[], b=[], len=0)
    # specialTerms['a'] = filter(lambda x: x in a, freq_terms)
    # specialTerms['b'] = filter(lambda x: x in b, freq_terms)
    # for x in LSimilarityVars.freq_ngrams['tokens']:
    #     if len(x) > 1:
    #         if x in str1: specialTerms['a'].append(x)
    #         if x in str2: specialTerms['b'].append(x)
    specialTerms['a'] = list(set(str1.split()) & LSimilarityVars.freq_ngrams['tokens'])
    specialTerms['b'] = list(set(str2.split()) & LSimilarityVars.freq_ngrams['tokens'])
    specialTerms['len'] = len(specialTerms['a']) + len(specialTerms['b'])
    specialTerms['char_len'] = sum(len(s) for s in specialTerms['a']) + sum(len(s) for s in specialTerms['b'])

    if specialTerms['a']:  # check if list is empty
        str1 = re.sub("|".join(specialTerms['a']), ' ', str1).strip()
    if specialTerms['b']:
        str2 = re.sub("|".join(specialTerms['b']), ' ', str2).strip()

    baseTerms, mismatchTerms = termsim_split(str1, str2, term_split_thres)

    return baseTerms, mismatchTerms, specialTerms


def score_per_term(baseTerms, mismatchTerms, specialTerms, method):
    baseScore, misScore, specialScore = 0, 0, 0

    if method == 'damerau_levenshtein':
        if baseTerms['a'] or baseTerms['b']:
            baseScore = damerau_levenshtein(' '.join(baseTerms['a']) + u'', ' '.join(baseTerms['b']) + u'')
        if mismatchTerms['a'] or mismatchTerms['b']:
            misScore = damerau_levenshtein(' '.join(mismatchTerms['a']) + u'', ' '.join(mismatchTerms['b']) + u'')
        if specialTerms['a'] or specialTerms['b']:
            specialScore = damerau_levenshtein(' '.join(specialTerms['a']) + u'', ' '.join(specialTerms['b']) + u'')
    elif method == 'davies':
        baseScore, misScore, specialScore = davies(' '.join(baseTerms['a']) + u'', ' '.join(baseTerms['b']) + u''), \
                                            davies(' '.join(mismatchTerms['a']) + u'',
                                                   ' '.join(mismatchTerms['b']) + u''), \
                                            davies(' '.join(specialTerms['a']) + u'', ' '.join(specialTerms['b']) + u'')
    elif method == 'skipgram':
        baseScore, misScore, specialScore = skipgram(' '.join(baseTerms['a']) + u'', ' '.join(baseTerms['b']) + u''), \
                                            skipgram(' '.join(mismatchTerms['a']) + u'',
                                                     ' '.join(mismatchTerms['b']) + u''), \
                                            skipgram(' '.join(specialTerms['a']) + u'',
                                                     ' '.join(specialTerms['b']) + u'')
    elif method == 'soft_jaccard':
        baseScore, misScore, specialScore = soft_jaccard(' '.join(baseTerms['a']) + u'',
                                                         ' '.join(baseTerms['b']) + u''), \
                                            soft_jaccard(' '.join(mismatchTerms['a']) + u'',
                                                         ' '.join(mismatchTerms['b']) + u''), \
                                            soft_jaccard(' '.join(specialTerms['a']) + u'',
                                                         ' '.join(specialTerms['b']) + u'')
    elif method == 'strike_a_match':
        baseScore, misScore, specialScore = strike_a_match(' '.join(baseTerms['a']) + u'',
                                                           ' '.join(baseTerms['b']) + u''), \
                                            strike_a_match(' '.join(mismatchTerms['a']) + u'',
                                                           ' '.join(mismatchTerms['b']) + u''), \
                                            strike_a_match(' '.join(specialTerms['a']) + u'',
                                                           ' '.join(specialTerms['b']) + u'')
    elif method == 'cosine':
        baseScore, misScore, specialScore = cosine(' '.join(baseTerms['a']) + u'', ' '.join(baseTerms['b']) + u''), \
                                            cosine(' '.join(mismatchTerms['a']) + u'',
                                                   ' '.join(mismatchTerms['b']) + u''), \
                                            cosine(' '.join(specialTerms['a']) + u'', ' '.join(specialTerms['b']) + u'')
    elif method == 'monge_elkan':
        baseScore, misScore, specialScore = monge_elkan(' '.join(baseTerms['a']) + u'', ' '.join(baseTerms['b']) + u''), \
                                            monge_elkan(' '.join(mismatchTerms['a']) + u'',
                                                        ' '.join(mismatchTerms['b']) + u''), \
                                            monge_elkan(' '.join(specialTerms['a']) + u'',
                                                        ' '.join(specialTerms['b']) + u'')
    elif method == 'jaro_winkler':
        baseScore, misScore, specialScore = jaro_winkler(' '.join(baseTerms['a']) + u'',
                                                         ' '.join(baseTerms['b']) + u''), \
                                            jaro_winkler(' '.join(mismatchTerms['a']) + u'',
                                                         ' '.join(mismatchTerms['b']) + u''), \
                                            jaro_winkler(' '.join(specialTerms['a']) + u'',
                                                         ' '.join(specialTerms['b']) + u'')
    elif method == 'jaro':
        baseScore, misScore, specialScore = jaro(' '.join(baseTerms['a']) + u'', ' '.join(baseTerms['b']) + u''), \
                                            jaro(' '.join(mismatchTerms['a']) + u'',
                                                 ' '.join(mismatchTerms['b']) + u''), \
                                            jaro(' '.join(specialTerms['a']) + u'', ' '.join(specialTerms['b']) + u'')
    elif method == 'jaccard':
        baseScore, misScore, specialScore = jaccard(' '.join(baseTerms['a']) + u'', ' '.join(baseTerms['b']) + u''), \
                                            jaccard(' '.join(mismatchTerms['a']) + u'',
                                                    ' '.join(mismatchTerms['b']) + u''), \
                                            jaccard(' '.join(specialTerms['a']) + u'',
                                                    ' '.join(specialTerms['b']) + u'')
    elif method == 'l_jaro_winkler':
        baseScore, misScore, specialScore = l_jaro_winkler(' '.join(baseTerms['a']) + u'',
                                                           ' '.join(baseTerms['b']) + u''), \
                                            l_jaro_winkler(' '.join(mismatchTerms['a']) + u'',
                                                           ' '.join(mismatchTerms['b']) + u''), \
                                            l_jaro_winkler(' '.join(specialTerms['a']) + u'',
                                                           ' '.join(specialTerms['b']) + u'')

    return baseScore, misScore, specialScore


def calibrate_weights(baseTerms, mismatchTerms, specialTerms, method, averaged=False, tmode=False):
    lsim_variance = 'avg' if averaged else 'simple'
    weights = LSimilarityVars.lsimilarity_weights[:] if tmode else LSimilarityVars.per_metric_optimal_values[method][lsim_variance][1][:]

    if baseTerms['len'] == 0:
        weights[1] += weights[0] * (float(mismatchTerms['len']) / (mismatchTerms['len'] + specialTerms['len']))
        weights[2] += weights[0] * (1 - float(mismatchTerms['len']) / (mismatchTerms['len'] + specialTerms['len']))
        weights[0] = 0
    if mismatchTerms['len'] == 0:
        weights[0] += weights[1] * (float(baseTerms['len']) / (baseTerms['len'] + specialTerms['len']))
        weights[2] += weights[1] * (1 - float(baseTerms['len']) / (baseTerms['len'] + specialTerms['len']))
        weights[1] = 0
    if specialTerms['len'] == 0:
        weights[0] += weights[2] * (float(baseTerms['len']) / (baseTerms['len'] + mismatchTerms['len']))
        weights[1] += weights[2] * (1 - float(baseTerms['len']) / (baseTerms['len'] + mismatchTerms['len']))
        weights[2] = 0

    if averaged:
        weights[0] = weights[0] * baseTerms['char_len'] / 2
        weights[1] = weights[1] * mismatchTerms['char_len'] / 2
        weights[2] = weights[2] * specialTerms['char_len'] / 2
    denominator = weights[0] + weights[1] + weights[2]

    return [w / denominator for w in weights]


def weighted_terms(baseTerms, mismatchTerms, specialTerms, method, averaged, test_mode=False):
    baseTerms_val, mismatchTerms_val, specialTerms_val = score_per_term(baseTerms, mismatchTerms, specialTerms, method)
    lweights = calibrate_weights(baseTerms, mismatchTerms, specialTerms, method, averaged, tmode=test_mode)

    return baseTerms_val * lweights[0] + mismatchTerms_val * lweights[1] + specialTerms_val * lweights[2]


def lsimilarity(str1, str2, method='damerau_levenshtein', averaged=False):
    lsim_variance = 'avg' if averaged else 'simple'
    split_thres = LSimilarityVars.per_metric_optimal_values[method][lsim_variance][0]

    baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(str1, str2, split_thres)
    thres = weighted_terms(baseTerms, mismatchTerms, specialTerms, method, averaged)

    return thres


def avg_lsimilarity(str1, str2, method='damerau_levenshtein'):
    return lsimilarity(str1, str2, method, True)
