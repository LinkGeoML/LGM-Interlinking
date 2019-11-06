# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

from __future__ import print_function
import time

from src import param_tuning
from src import config
from src.features import Features


class StrategyEvaluator:
    """
    This class implements the pipeline for various strategies.
    """
    def __init__(self, encoding='latin'):
        self.encoding = encoding

    def hyperparamTuning(self, train_data='data/dataset-string-similarity_global_1k.csv', test_data='data/dataset-string-similarity.txt'):
        """A complete process of distinct steps in figuring out the best ML algorithm with best hyperparameters to
        toponym interlinking problem.

        :param train_data: Relative path to the train dataset.
        :type train_data: str
        :param test_data: Relative path to the test dataset.
        :type test_data: str
        """
        pt = param_tuning.ParamTuning()
        f = Features()

        tot_time = time.time(); start_time = time.time()
        f.load_data(train_data, self.encoding)
        fX, y = f.build()
        print("Loaded train dataset and build features for {} setup; {} sec.".format(
            config.MLConf.classification_method, time.time() - start_time))

        start_time = time.time()
        # 1st phase: find out best classifier from a list of candidate ones
        best_clf = pt.getBestClassifier(fX, y)
        print("Best classifier is {} with score {}; {} sec.".format(
            best_clf['classifier'], best_clf['accuracy'], time.time() - start_time))

        start_time = time.time()
        #  2nd phase: fine tune the best classifier in previous step
        estimator, params, score = pt.fineTuneClassifier(fX, y, best_clf)
        print("Best hyperparams, {}, with score {}; {} sec.".format(params, score, time.time() - start_time))

        start_time = time.time()
        # 3nd phase: train the fine tuned best classifier on the whole train dataset (no folds)
        estimator = pt.trainClassifier(fX, y, estimator)
        print("Finished training model on the dataset; {} sec.".format(time.time() - start_time))

        start_time = time.time()
        f.load_data(test_data, self.encoding)
        fX, y = f.build()
        print("Loaded test dataset and build features; {} sec".format(time.time() - start_time))

        start_time = time.time()
        # 4th phase: test the fine tuned best classifier on the test dataset
        acc, pre, rec, f1 = pt.testClassifier(fX, y, estimator)
        print("| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (sec)")
        print("||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(
            best_clf['classifier'], acc, pre, rec, f1, time.time() - start_time))

        print("The whole process took {} sec.".format(time.time() - tot_time))
