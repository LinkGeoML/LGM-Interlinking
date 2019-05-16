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
        -e <encoding_type>          specify the encoding of toponyms in datasets. [default: global].

    Arguments:
        encoding_type               global
                                    latin
"""

import os, sys
from docopt import docopt
from kitchen.text.converters import getwriter

from src.strategy_eval import Evaluator
from src.helpers import getRelativePathtoWorking, StaticValues
from src.sim_measures import LSimilarityVars
import src.config as config


def main(args):
    UTF8Writer = getwriter('utf8')
    sys.stdout = UTF8Writer(sys.stdout)

    LSimilarityVars.per_metric_optimal_values = StaticValues.opt_values[args["-e"].lower()]

    fpath_ds = getRelativePathtoWorking(config.ML.test_dataset)
    if os.path.isfile(fpath_ds) and os.path.isfile(getRelativePathtoWorking(config.ML.train_dataset)):
        seval = Evaluator(args['-e'])
        seval.hyperparamTuning(fpath_ds)
    else:
        print("File {0} and/or {1} is not found!!!\n".format(
            fpath_ds, getRelativePathtoWorking(config.ML.train_dataset)))


if __name__ == "__main__":
    arguments = docopt(__doc__, version='LGM-Interlinking 0.1.0')
    main(arguments)
