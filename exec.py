# -*- coding: utf-8 -*-

"""Feature extraction and traditional classifiers for toponym interlinking.

Usage:
  exec.py [options]
  exec.py (-h | --help)
  exec.py --version

Options:
  -h --help                     show this screen.
  --version                     show version.
  -d <dataset-name>             relative path to the directory of the script being run of the dataset to use for
                                experiments. [default: dataset-string-similarity.txt].
  --canonical                   perform canonical decomposition (NFKD). Default is False.
  --sort                        sort alphanumerically.
  -e <encoding_type>            Check for similarities only for the specified encoding type. [default: latin].

Arguments:
  encoding_type             'global'
                            'latin'
"""

import os, sys
from docopt import docopt
from kitchen.text.converters import getwriter

import methods
from helpers import getRelativePathtoWorking, StaticValues
from sim_measures import LSimilarityVars


def main(args):
    UTF8Writer = getwriter('utf8')
    sys.stdout = UTF8Writer(sys.stdout)

    LSimilarityVars.per_metric_optimal_values = StaticValues.MetricOptimalValues[args["-e"].lower()]
    dataset_path = [x for x in args['-d'].split(',')]

    evaluator = methods.Evaluator(args['--sort'], args['--canonical'], args['-e'])

    fpath_ds = getRelativePathtoWorking(dataset_path[0])
    if os.path.isfile(fpath_ds):
        evaluator.hyperparamTuning(fpath_ds)
    else: print("No file {0} exists!!!\n".format(fpath_ds))


if __name__ == "__main__":
    arguments = docopt(__doc__, version='LGM-Interlinking 0.1.0')
    main(arguments)
