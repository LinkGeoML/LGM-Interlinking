# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

from __future__ import print_function
import time

from helpers import getRelativePathtoWorking
import param_tuning
import config
from featuresConstruction import Features


class Evaluator:
    """
    Main class Evaluator...
    """
    def __init__(self, encoding='latin'):
        self.encoding = encoding

    def hyperparamTuning(self, dataset='dataset-string-similarity.txt'):
        """
        The main pipeline to figure out the best ML algorithm with best hyperparameters for our problem.

        :param dataset: relative path to the test dataset
        :type dataset: str
        """
        pt = param_tuning.ParamTuning()
        f = Features()

        tot_time = time.time(); start_time = time.time()
        f.load_data(getRelativePathtoWorking(config.initialConfig.train_dataset), self.encoding)
        fX, y = f.build_features()
        print("Loaded train dataset and build features for {} setup; {} sec.".format(
            config.initialConfig.classification_method, time.time() - start_time))

        start_time = time.time()
        # 1st phase: find out best classifier from a list of candidate ones
        best_clf = pt.getBestClassifier(fX, y)
        print("Best classifier is {} with score {}; {} sec.".format(
            best_clf['classifier'], best_clf['accuracy'], time.time() - start_time))

        start_time = time.time()
        #  2nd phase: fine tune the best classifier in previous step
        estimator, params, score = pt.fineTuningBestClassifier(fX, y, best_clf)
        print("Best hyperparams, {}, with score {}; {} sec.".format(params, score, time.time() - start_time))

        start_time = time.time()
        # 3nd phase: train the fine tuned best classifier on the whole train dataset (no folds)
        estimator = pt.trainBestClassifier(fX, y, estimator)
        print("Finished training model on the dataset; {} sec.".format(time.time() - start_time))

        start_time = time.time()
        f.load_data(dataset, self.encoding)
        fX, y = f.build_features()
        print("Loaded test dataset and build features; {} sec".format(time.time() - start_time))

        start_time = time.time()
        # 4th phase: test the fine tuned best classifier on the test dataset
        acc, pre, rec, f1 = pt.testBestClassifier(fX, y, estimator)
        print("| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (sec)")
        print("||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(
            best_clf['classifier'], acc, pre, rec, f1, time.time() - start_time))

        print("The whole process took {} sec.".format(time.time() - tot_time))
