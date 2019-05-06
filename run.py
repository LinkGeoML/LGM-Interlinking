# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

"""Feature extraction and traditional classifiers for toponym interlinking.

Command line::

    Usage:
      run.py [options]
      run.py (-h | --help)
      run.py --version

    Options:
        -h --help                   show this screen.
        --version                   show version.
        -d <dataset-name>           relative path to the directory of the script being run of the dataset to use for
                                    experiments. [default: dataset-string-similarity.txt].
        -e <encoding_type>          specify the encoding type followed by toponyms in datasets. [default: latin].

    Arguments:
        encoding_type               'global'
                                    'latin'
"""

import os, sys
from docopt import docopt
from kitchen.text.converters import getwriter

import src.methods
from src.helpers import getRelativePathtoWorking, StaticValues
from src.sim_measures import LSimilarityVars


def main(args):
    UTF8Writer = getwriter('utf8')
    sys.stdout = UTF8Writer(sys.stdout)

    LSimilarityVars.per_metric_optimal_values = StaticValues.MetricOptimalValues[args["-e"].lower()]

    fpath_ds = getRelativePathtoWorking(args['-d'])
    if os.path.isfile(fpath_ds):
        evaluator = src.methods.Evaluator(args['-e'])
        evaluator.hyperparamTuning(fpath_ds)
    else: print("No file {0} exists!!!\n".format(fpath_ds))


if __name__ == "__main__":
    arguments = docopt(__doc__, version='LGM-Interlinking 0.1.0')
    main(arguments)
