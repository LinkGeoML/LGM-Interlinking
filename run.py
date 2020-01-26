# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

"""A complete pipeline of processes for features extraction and training/evaluating state-of-the-art classifiers for
toponym interlinking.

Command line::

    Usage:
      run.py [options]
      run.py (-h | --help)
      run.py --version

    Options:
        -h --help                   show this screen.
        --version                   show version.
        --dtrain <fpath>            relative path to the train dataset. If this is null, the assigned
                                    value to `train_dataset` parameter in config.py is used instead.
        --dtest <fpath>             relative path to the test dataset. If this is null, the assigned
                                    value to `test_dataset` parameter in config.py is used instead.
        -e <encoding_type>          specify the encoding of toponyms in datasets. [default: global].
        --customparams              run classifiers with custom hyper-parameters defined in config.py file [default: False].
        --build                     build the dataset for evaluation.

    Arguments:
        encoding_type               global
                                    latin

"""

import os, sys
# import codecs
from docopt import docopt

from src.core import StrategyEvaluator
from src.helpers import getRelativePathtoWorking, StaticValues
from src.sim_measures import LGMSimVars, build_dataset
import src.config as config


def main(args):
    # UTF8Writer = codecs.getwriter('utf8')
    # sys.stdout = UTF8Writer(sys.stdout)

    d_test = getRelativePathtoWorking(config.test_dataset) if args['--dtest'] is None \
        else getRelativePathtoWorking(args['--dtest'])
    d_train = getRelativePathtoWorking(config.train_dataset) if args['--dtrain'] is None \
        else getRelativePathtoWorking(args['--dtrain'])
    if os.path.isfile(d_test) and os.path.isfile(d_train):
        if args['--build']:
            build_dataset(d_train, args['-e'])
            sys.exit(0)

        LGMSimVars.per_metric_optValues = StaticValues.opt_values[args["-e"].lower()]
        seval = StrategyEvaluator(args['-e'])

        if args['--customparams']: seval.exec_classifiers(d_train, d_test)
        else: seval.hyperparamTuning(d_train, d_test)
    else:
        print("File {0} and/or {1} is not found!!!\n".format(d_test, d_train))


if __name__ == "__main__":
    arguments = docopt(__doc__, version='LGM-Interlinking 0.2.1')
    main(arguments)
