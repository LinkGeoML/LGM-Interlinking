# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

"""Feature extraction and state-of-the-art classifiers for toponym interlinking.

Command line::

    Usage:
      run.py [options]
      run.py (-h | --help)
      run.py --version

    Options:
        -h --help                   show this screen.
        --version                   show version.
        --dtrain <fpath>            relative path to the train dataset. If this is None, the train_dataset parameter in
                                    config.py is used instead.
        --dtest <fpath>             relative path to the test dataset. If this is None, the train_dataset parameter in
                                    config.py is used instead.
        -e <encoding_type>          specify the encoding of toponyms in datasets. [default: global].

    Arguments:
        encoding_type               global
                                    latin
"""

import os, sys
from docopt import docopt
from kitchen.text.converters import getwriter

from src.core import StrategyEvaluator
from src.helpers import getRelativePathtoWorking, StaticValues
from src.sim_measures import LSimilarityVars
import src.config as config


def main(args):
    UTF8Writer = getwriter('utf8')
    sys.stdout = UTF8Writer(sys.stdout)

    LSimilarityVars.per_metric_optValues = StaticValues.opt_values[args["-e"].lower()]

    d_test = getRelativePathtoWorking(config.test_dataset) if args['--dtest'] is None \
        else getRelativePathtoWorking(args['--dtest'])
    d_train = getRelativePathtoWorking(config.train_dataset) if args['--dtrain'] is None \
        else getRelativePathtoWorking(args['--dtrain'])
    if os.path.isfile(d_test) and os.path.isfile(d_train):
        seval = StrategyEvaluator(args['-e'])
        seval.hyperparamTuning(d_train, d_test)
    else:
        print("File {0} and/or {1} is not found!!!\n".format(d_test, d_train))


if __name__ == "__main__":
    arguments = docopt(__doc__, version='LGM-Interlinking 0.1.0')
    main(arguments)
