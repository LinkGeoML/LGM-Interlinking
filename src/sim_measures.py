#!/usr/bin/env python
# encoding=utf8
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

"""
This module implements various similarity metrics used across different scenarios. Many of these functions were
developed by Rui Santos and Alexandre Marinho for their work in
`Toponym-Matching <https://github.com/ruipds/Toponym-Matching/blob/master/datasetcreator.py>`_.
"""

# Python 2 and 3
from builtins import range
from io import open
import six

import csv
import os.path
import sys
import math
import random
import itertools
import re
import __main__

import numpy as np
import unicodedata
from alphabet_detector import AlphabetDetector
import pycountry_convert
import jellyfish
import pyxdameraulevenshtein
from tqdm import tqdm


def getBasePath():
    return os.path.abspath(os.path.dirname(__main__.__file__))


def getRelativePathtoWorking(ds):
    return os.path.join(getBasePath(), ds)


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
    uni_string = six.text_type(str)
    if only:
        return ad.only_alphabet_chars(uni_string, alphabet.upper())
    else:
        for i in uni_string:
            if ad.is_in_alphabet(i, alphabet.upper()): return True
        return False


def detect_alphabet(str):
    ad = AlphabetDetector()
    uni_string = six.text_type(str)
    ab = ad.detect_alphabet(uni_string)
    if "CYRILLIC" in ab:
        return "CYRILLIC"
    return ab.pop() if len(ab) != 0 else 'UND'


# The geonames dataset can be obtained from http://download.geonames.org/export/dump/allCountries.zip
def build_dataset_from_geonames(dataset='allCountries.txt', output='dataset-unfiltered.txt', only_latin=False):
    # remove dupls after running this script
    # cat -n dataset-string-similarity.txt | sort -k2 -k1n | uniq -f1 | sort -nk1,1 | cut -f2-

    csv.field_size_limit(sys.maxsize)
    lastname = None
    lastname2 = None
    lastid = None
    # country = None
    skip = random.randint(10, 10000)
    file = open(getRelativePathtoWorking(os.path.join('data', output)), "w+")
    max_no_attempts = 300
    totalrows = 0
    min_altnames = 3
    max_altnames = 4

    input = getRelativePathtoWorking(os.path.join('data', dataset))
    if not os.path.isfile(input):
        print("File {0} does not exist".format(input))
        exit()

    print("Working on dataset {}...".format(input))
    with open(input) as csvfile:
        # reader = csv.DictReader(csvfile, fieldnames=fields, delimiter='\t')
        first_line = csvfile.readline()
        has_header = csv.Sniffer().has_header(first_line)
        file.seek(0)  # Rewind.
        reader = csv.DictReader(csvfile, fieldnames=first_line.rstrip('\n').split(',') if has_header else fields)
        if has_header:
            next(reader)  # Skip header row.
        for row in reader:
            totalrows += 1
            skip = skip - 1
            if skip > 0: continue
            names = set([name.strip() for name in ("" + row['alternate_names']).split(",") if len(name.strip()) > 3])

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

            if len(names) < max_altnames: continue
            lastid = row['geonameid']
            firstcountry = row['country_code'] if 'country_code' in row else 'unknown'
            lastname = random.sample(names, 1)[0]
            lastname2 = random.sample(names, 1)[0]
            while True:
                lastname2 = random.sample(names, 1)[0]
                if not (lastname2.lower() == lastname.lower()): break
    with open(input) as csvfile:
        # reader = csv.DictReader(csvfile, fieldnames=fields, delimiter='\t')
        first_line = csvfile.readline()
        has_header = csv.Sniffer().has_header(first_line)
        file.seek(0)  # Rewind.
        reader = csv.DictReader(csvfile, fieldnames=first_line.rstrip('\n').split(',') if has_header else fields)
        if has_header:
            next(reader)  # Skip header row.

        with tqdm(total=totalrows) as pbar:
            for row in reader:
                pbar.update(1)

                names = set([name.strip() for name in ("" + row['alternate_names']).split(",") if len(name.strip()) > 3])
                if len(row['name'].strip()) > 2: names.add(row['name'].strip())
                if len(six.text_type(row['asciiname']).strip()) > 2: names.add(six.text_type(row['asciiname']).strip())

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

                if len(names) < min_altnames: continue
                id = row['geonameid']
                country = row['country_code'] if 'country_code' in row else 'unknown'
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
                    firstcountry = row['country_code'] if 'country_code' in row else 'unknown'
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
                if len(names) < max_altnames:
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
                    print("Failed to find alternative names for {}...".format(id))
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


# The geonames dataset can be obtained from http://download.geonames.org/export/dump/allCountries.zip
def build_dataset_from_source(dataset='allCountries.txt', n_alternates=3, output='dataset-unfiltered.txt'):
    # remove dupls after running this script
    # cat -n dataset-string-similarity.txt | sort -k2 -k1n | uniq -f1 | sort -nk1,1 | cut -f2-

    csv.field_size_limit(sys.maxsize)
    lastnames = []

    skip = random.randint(10, 10000)
    file = open(getRelativePathtoWorking(os.path.join('data', output)), "w+")
    max_no_attempts = 300
    totalrows = 0
    str_length = 2
    negative_list_size = 10000

    input = getRelativePathtoWorking(os.path.join('data', dataset))
    if not os.path.isfile(input):
        print("File {0} does not exist".format(input))
        exit()

    print("Working on dataset {}...".format(input))
    with open(input) as csvfile:
        # reader = csv.DictReader(csvfile, fieldnames=fields, delimiter='\t')
        first_line = csvfile.readline()
        has_header = csv.Sniffer().has_header(first_line)
        file.seek(0)  # Rewind.
        reader = csv.DictReader(csvfile, fieldnames=first_line.rstrip('\n').split(',') if has_header else fields)
        if has_header:
            next(reader)  # Skip header row.
        for row in reader:
            skip = skip - 1
            if skip > 0: continue
            names = set([name.strip() for name in ("" + row['alternate_names']).split(",") if len(name.strip()) > str_length])

            if len(names) < n_alternates: continue
            lastid = row['geonameid']
            firstcountry = row['country_code'] if 'country_code' in row else 'unknown'
            for n in names:
                lastnames.append([n, lastid, firstcountry])

            if len(lastnames) >= negative_list_size: break

    with open(input) as csvfile:
        # reader = csv.DictReader(csvfile, fieldnames=fields, delimiter='\t')
        first_line = csvfile.readline()
        has_header = csv.Sniffer().has_header(first_line)
        file.seek(0)  # Rewind.
        reader = csv.DictReader(csvfile, fieldnames=first_line.rstrip('\n').split(',') if has_header else fields)
        if has_header:
            next(reader)  # Skip header row.

        with tqdm(total=totalrows) as pbar:
            for row in reader:
                pbar.update(1)

                names = set([name.strip() for name in ("" + row['alternate_names']).split(",") if len(name.strip()) > str_length])
                if len(row['name'].strip()) > str_length: names.add(row['name'].strip())
                if len(six.text_type(row['asciiname']).strip()) > str_length: names.add(six.text_type(row['asciiname']).strip())

                if len(names) < n_alternates: continue
                id = row['geonameid']
                country = row['country_code'] if 'country_code' in row else 'unknown'

                for n1, n2 in itertools.combinations(names, 2):
                    if jaccard(n1.lower(), n2.lower()) == 1.0:
                        continue

                    file.write(n1 + "\t" + n2 + "\tTRUE\t" + id + "\t" + id + "\t" + detect_alphabet(
                        n1) + "\t" + detect_alphabet(n2) + "\t" + country + "\t" + country + "\n")
                    attempts = max_no_attempts
                    while attempts > 0:
                        attempts = attempts - 1
                        randomname = random.sample(lastnames, 1)[0]
                        if randomname[1] != id: break
                    if attempts > 0:
                        file.write(
                            n1 + "\t" + randomname[0] + "\tFALSE\t" + id + "\t" + randomname[1] + "\t" +
                            detect_alphabet(n1) + "\t" + detect_alphabet(randomname[0]) + "\t" +
                            country + "\t" + randomname[2] + "\n")

                for n in names:
                    lastnames.append([n, id, country])
                lastnames = lastnames[(len(lastnames) - negative_list_size):]
    file.close()


def filter_dataset(input='dataset-unfiltered.txt', num_instances=2500000):
    pos = []
    neg = []
    file = open(getRelativePathtoWorking(os.path.join("data", "dataset-string-similarity.txt")), "w+")
    print("Filtering for {0}...".format(num_instances * 2))
    for line in open(getRelativePathtoWorking(os.path.join("data", input))):
        splitted = line.split('\t')
        # if not (splitted[2] == "TRUE" or splitted[2] == "FALSE") or \
        #         not (len(six.text_type(splitted[7])) == 2 and len(six.text_type(splitted[8])) == 3) or \
        #         not (splitted[5] != "UND" and splitted[6] != "UND") or \
        #         not (splitted[3].isdigit() and splitted[4].isdigit()) or \
        #         len(splitted) != 9 or \
        #         len(six.text_type(splitted[1])) < 3 or \
        #         len(six.text_type(splitted[0])) < 3:
        #     continue
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


def build_dataset(dataset='allCountries.txt', n_alternates=3, num_instances=2500000, encoding='global'):
    # build_dataset_from_geonames(dataset=dataset, only_latin=True if encoding.lower() == 'latin' else False)
    build_dataset_from_source(dataset, n_alternates)
    filter_dataset(num_instances=num_instances)


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
    """Implements Jaccard-skipgram metric.

    Parameters
    ----------
    str1, str2: str
        Input values in unicode.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
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
    """Implements Davies de Salles metric.

    str1, str2: str
        Input values in unicode.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
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
    """Implements Cosine N-Grams metric for n=[2,3].

    str1, str2: str
        Input values in unicode.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
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
    """Implements Damerau-Levenshtein metric.

    str1, str2: str
        Input values in unicode.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
    aux = pyxdameraulevenshtein.normalized_damerau_levenshtein_distance(str1, str2)
    return 1.0 - aux


def jaro(str1, str2):
    """Implements Jaro metric.

    str1, str2: str
        Input values in unicode.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
    aux = jellyfish.jaro_distance(str1, str2)
    return aux


def jaro_winkler(str1, str2):
    """Implements Jaro-Winkler metric.

    str1, str2: str
        Input values in unicode.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
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
    """Implements Monge-Elkan metric.

    str1, str2: str
        Input values in unicode.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
    return (monge_elkan_aux(str1, str2) + monge_elkan_aux(str2, str1)) / 2.0


# http://www.catalysoft.com/articles/StrikeAMatch.html
def strike_a_match(str1, str2):
    """Implements Dice Bi-Grams metric.

    str1, str2: str
        Input values in unicode.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
    pairs1 = {str1[i:i + 2] for i in range(len(str1) - 1)}
    pairs2 = {str2[i:i + 2] for i in range(len(str2) - 1)}
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
    """Implements Jaccard N-Grams metric for n=[2,3].

    str1, str2: str
        Input values in unicode.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
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
    """Implements Soft-Jaccard metric.

    str1, str2: str
        Input values in unicode.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
    a = set(str1.split(" "))
    b = set(str2.split(" "))
    intersection_length = (sum(max(jaro_winkler(i, j) for j in b) for i in a) + sum(
        max(jaro_winkler(i, j) for j in a) for i in b)) / 2.0
    return float(intersection_length) / (len(a) + len(b) - intersection_length)


def sorted_winkler(str1, str2):
    """Implements Sorted Jaro-Winkler metric.

    str1, str2: str
        Input values in unicode.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
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
    if not isinstance(s, six.text_type):
        raise TypeError('expected str or unicode, got %s' % type(s).__name__)


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


def lgm_jaro_winkler(s1, s2, long_tolerance=False):
    """Implements LGM Jaro-Winkler metric.

    str1, str2: str
        Input values in unicode.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
    return _jaro_winkler(s1, s2, long_tolerance, True)


class LGMSimVars:
    freq_ngrams = {'tokens': set(), 'chars': set()}
    weights = []
    per_metric_optValues = {}


def termsim_split(s1, s2, thres):
    base = {'a': [], 'b': [], 'len': 0}
    mis = {'a': [], 'b': [], 'len': 0}

    ls1, ls2 = s1.split(), s2.split()
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


def lgm_sim_lterms(s1, s2, split_thres):
    """Splits each toponym-string, i.e., s1, s2, to tokens, builds three distinct lists per toponym-string, i.e., base,
    mismatch and frequent, and assigns the produced tokens to these lists. The *base* lists contains the terms that are
    similar to one of the other toponym's tokens, The *mismatch* contains the terms that have no similar pair to the
    tokens of the other toponym and the *frequent* list contains the terms that are common for the specified dataset
    of toponyms.

    Parameters
    ----------
    s1, s2: str
        Input values in unicode.
    split_thres: float
        If the similarity score is above this threshold, the compared terms are identified as base terms,
        otherwise as mismatch ones.

    Returns
    -------
    tuple of (dict of list of :obj:`str`, dict of list of :obj:`str`, dict of list of :obj:`str`)
        Three lists of terms identified as base, mismatch or frequent respectively per toponym, i.e., *a* for s1 and
        *b* for s2.
    """
    specialTerms = dict(a=[], b=[], len=0)

    specialTerms['a'] = list(set(s1.split()) & LGMSimVars.freq_ngrams['tokens'])
    specialTerms['b'] = list(set(s2.split()) & LGMSimVars.freq_ngrams['tokens'])
    specialTerms['len'] = len(specialTerms['a']) + len(specialTerms['b'])
    specialTerms['char_len'] = sum(len(s) for s in specialTerms['a']) + sum(len(s) for s in specialTerms['b'])

    if specialTerms['a']:  # check if list is empty
        s1 = re.sub("|".join(specialTerms['a']), ' ', s1).strip()
    if specialTerms['b']:
        s2 = re.sub("|".join(specialTerms['b']), ' ', s2).strip()

    baseTerms, mismatchTerms = termsim_split(s1, s2, split_thres)

    return baseTerms, mismatchTerms, specialTerms


def score_per_term(base_t, mis_t, special_t, metric):
    """Computes three distinct similarity scores for each list of terms.

    Parameters
    ----------
    base_t, mismatch_t special_t: list of str
        Lists of toponym terms identified as base, mismatch or frequent (special) respectively.
    metric: str
        Indicates the metric to utilize in order to calculate the similarity score by comparing individually the
        three lists.
    Returns
    -------
    tuple of (float, float, float)
        A similarity score for every list of terms. Each score is normalized in range [0,1].
    """
    scores = [0, 0, 0]  # base, mis, special

    for idx, (term_a, term_b) in enumerate(zip(
            [base_t['a'], mis_t['a'], special_t['a']],
            [base_t['b'], mis_t['b'], special_t['b']]
    )):
        if term_a or term_b: scores[idx] = algnms_to_func[metric](u' '.join(term_a), u' '.join(term_b))

    return scores[0], scores[1], scores[2]


def recalculated_weights(base_t, mis_t, special_t, metric, avg=False, tmode=False):
    lsim_variance = 'avg' if avg else 'simple'
    weights = LGMSimVars.weights[:] if tmode \
        else LGMSimVars.per_metric_optValues[metric][lsim_variance][1][:]

    if base_t['len'] == 0:
        weights[1] += weights[0] * (float(mis_t['len']) / (mis_t['len'] + special_t['len']))
        weights[2] += weights[0] * (1 - float(mis_t['len']) / (mis_t['len'] + special_t['len']))
        weights[0] = 0
    if mis_t['len'] == 0:
        weights[0] += weights[1] * (float(base_t['len']) / (base_t['len'] + special_t['len']))
        weights[2] += weights[1] * (1 - float(base_t['len']) / (base_t['len'] + special_t['len']))
        weights[1] = 0
    if special_t['len'] == 0:
        weights[0] += weights[2] * (float(base_t['len']) / (base_t['len'] + mis_t['len']))
        weights[1] += weights[2] * (1 - float(base_t['len']) / (base_t['len'] + mis_t['len']))
        weights[2] = 0

    if avg:
        weights[0] = weights[0] * base_t['char_len'] / 2
        weights[1] = weights[1] * mis_t['char_len'] / 2
        weights[2] = weights[2] * special_t['char_len'] / 2
    denominator = weights[0] + weights[1] + weights[2]

    return [w / denominator for w in weights]


def weighted_sim(base_t, mis_t, special_t, metric, avg, test_mode=False):
    """Re-calculates the significance weights for each list of terms taking into account their lengths.

    Parameters
    ----------
    base_t, mis_t, special_t: list of str
        Lists of toponym terms identified as base, mismatch or frequent (special) respectively.
    metric: str
        Indicates the metric to utilize in order to calculate the similarity score by comparing individually the
        three lists.
    avg: bool
        If value is True, the three individual similarity scores (for each term list) are properly weighted, otherwise
        each term list' score is of equal significance to the final score.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
    baseTerms_val, mismatchTerms_val, specialTerms_val = score_per_term(base_t, mis_t, special_t, metric)
    lweights = recalculated_weights(base_t, mis_t, special_t, metric, avg, tmode=test_mode)

    return baseTerms_val * lweights[0] + mismatchTerms_val * lweights[1] + specialTerms_val * lweights[2]


def lgm_sim(str1, str2, metric='damerau_levenshtein', avg=False):
    """Implements LGM-Sim metric.

    Parameters
    ----------
    str1, str2: str
        Input values in unicode.
    metric: str, optional
        Similarity metric used, as internal one, to split toponyms in the two distinct lists that contains base and
        mismatch terms respectively. Each of the above supported metrics can be used as input.
        Default metric is :attr:`~src.sim_measures.damerau_levenshtein`.
    avg: bool, optional
        If value is True, the three individual similarity scores (for each term list) are properly weighted, otherwise
        each term list' score is of equal significance to the final score. Default value is False.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
    lsim_variance = 'avg' if avg else 'simple'
    split_thres = LGMSimVars.per_metric_optValues[metric][lsim_variance][0]

    baseTerms, mismatchTerms, specialTerms = lgm_sim_lterms(str1, str2, split_thres)
    thres = weighted_sim(baseTerms, mismatchTerms, specialTerms, metric, avg)

    return thres


def avg_lgm_sim(str1, str2, metric='damerau_levenshtein'):
    """Implements LGM-Sim metric where *avg* flag is True.

    Parameters
    ----------
    str1, str2: str
        Input values in unicode.
    metric: str, optional
        Similarity metric used, as internal one, to split toponyms in the two distinct lists that contains base and
        mismatch terms respectively. Each of the above supported metrics can be used as input.
        Default metric is :attr:`~src.sim_measures.damerau_levenshtein`.

    Returns
    -------
    float
        A similarity score normalized in range [0,1].
    """
    return lgm_sim(str1, str2, metric, True)

algnms_to_func = {
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
    'lgm_jaro_winkler': lgm_jaro_winkler,
    'lgm_sim': lgm_sim,
    'avg_lgm_sim': avg_lgm_sim,
}
