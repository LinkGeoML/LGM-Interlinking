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
        --n_alternates <no>         min number of alternative names in order to process the record [default: -1]
        --num_instances <no>        total number of toponym pairs to create per status (true/false) [default: 500000]

    Arguments:
        encoding_type               global
                                    latin

"""

import os, sys
# import codecs

from interlinking.sim_measures import build_dataset
import interlinking.config as config


def main(args):
    # UTF8Writer = codecs.getwriter('utf8')
    # sys.stdout = UTF8Writer(sys.stdout)
    if args['--build']:
        build_dataset(config.raw_dataset, int(args['--n_alternates']), int(args['--num_instances']), args['-e'])
        sys.exit(0)


# if __name__ == "__main__":
#     arguments = docopt(__doc__, version='LGM-Interlinking 0.2.1')
#     main(arguments)
