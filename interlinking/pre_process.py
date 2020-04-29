import csv
from collections import Counter, defaultdict
import itertools
import os
import re

from interlinking import config, helpers


def extract_freqterms(fname, encoding):
    pattern = re.compile("^[a-zA-Z]+")

    ngram_stats = {
        # '2gram': Counter(), '3gram': Counter(), '4gram': Counter(),
        'gram_token': Counter(),
        # '2gram_token': Counter(), '3gram_token': Counter()
    }

    dstemmed = defaultdict(set)
    with open(os.path.join(config.default_data_path, fname)) as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=config.fieldnames, delimiter=config.delimiter)

        for row in reader:
            a, b = helpers.transform(row[config.use_cols['s1']], row[config.use_cols['s2']], canonical=True)

            for s in [a, b]:
                ngram_tokens, ngram_tokens_stemmed, _ = helpers.normalize_str(s)

                for term, stem in zip(ngram_tokens, ngram_tokens_stemmed):
                    if len(term) < 3 or not pattern.match(term): continue

                    ngram_stats['gram_token'][stem] += 1
                    dstemmed[stem].add(term)
                # for gram in list(itertools.chain.from_iterable(
                #         [[ngram_tokens_stemmed[i:i + n] for i in range(len(ngram_tokens_stemmed) - (n - 1))]
                #          for n in [2, 3]])
                # ):
                #     if len(gram) == 2:
                #         ngram_stats['2gram_token'][' '.join(gram)] += 1
                #     else:
                #         ngram_stats['3gram_token'][' '.join(gram)] += 1

                # # ngrams chars
                # # ngrams = zip(*[''.join(strA_ngrams_tokens)[i:] for i in range(n) for n in [2, 3, 4]])
                # for gram in list(itertools.chain.from_iterable(
                #         [[''.join(ngram_tokens)[i:i + n] for i in range(len(''.join(ngram_tokens)) - (n - 1))]
                #          for n in [2, 3, 4]])
                # ):
                #     if len(gram) == 2:
                #         ngram_stats['2gram'][gram] += 1
                #     elif len(gram) == 3:
                #         ngram_stats['3gram'][gram] += 1
                #     elif len(gram) == 4:
                #         ngram_stats['4gram'][gram] += 1

    for gram in ngram_stats.keys():
        with open(os.path.join(config.default_data_path, "{0}s_{1}.csv".format(gram, encoding)), "w+") as f:
            f.write('gram\tcount\n')
            for value, count in ngram_stats[gram].most_common():
                for t in dstemmed.get(value):
                    f.write("{}\t{}\n".format(t, count))
