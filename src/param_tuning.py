# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import config
import numpy as np

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


seed_no = 13
np.random.seed(seed_no)


class ParamTuning:
    """
    This class provides all main methods for selecting, fine tuning hyperparameters, training and testing the best
    classifier for toponym matching. The following classifiers are examined:

    * Support Vector Machine (SVM)
    * Decision Trees
    * Multi-Layer Perceptron (MLP)
    * Random Forest
    * Extra-Trees
    * eXtreme Gradient Boosting (XGBoost)
    """
    clf_names = {
        'SVM': [SVC, config.MLConf.SVM_hyperparameters, config.MLConf.SVM_hyperparameters_dist],
        'Decision Tree': [DecisionTreeClassifier, config.MLConf.DecisionTree_hyperparameters,
                          config.MLConf.DecisionTree_hyperparameters_dist],
        'MLP': [MLPClassifier, config.MLConf.MLP_hyperparameters, config.MLConf.MLP_hyperparameters_dist],
        'Random Forest': [RandomForestClassifier, config.MLConf.RandomForest_hyperparameters,
                          config.MLConf.RandomForest_hyperparameters_dist],
        'Extra-Trees': [ExtraTreesClassifier, config.MLConf.RandomForest_hyperparameters,
                        config.MLConf.RandomForest_hyperparameters_dist],
        'XGBoost': [XGBClassifier, config.MLConf.XGBoost_hyperparameters,
                    config.MLConf.XGBoost_hyperparameters_dist]
    }

    scores = ['accuracy']

    def __init__(self):
        # To be used within GridSearch
        self.inner_cv = StratifiedKFold(n_splits=config.MLConf.kfold_inner_parameter, shuffle=False,
                                        random_state=seed_no)

        # To be used in outer CV
        self.outer_cv = StratifiedKFold(n_splits=config.MLConf.kfold_parameter, shuffle=False,
                                        random_state=seed_no)

        self.kfold = config.MLConf.kfold_parameter
        self.n_jobs = config.MLConf.n_jobs

        self.search_method = config.MLConf.hyperparams_search_method
        self.n_iter = config.MLConf.max_iter

    def getBestClassifier(self, X, y):
        """Search over specified parameter values for various estimators/classifiers and choose the best one.

        This method searches over specified values and selects the classifier that
        achieves the best avg accuracy score for all evaluations. The supported search methods are:

        * *GridSearchCV*: Exhaustive search over specified parameter values for supported estimators.
          The following variables are defined in :func:`~src.config.MLConf` :

         * :attr:`~src.config.MLConf.MLP_hyperparameters`
         * :attr:`~src.config.MLConf.RandomForests_hyperparameters`
         * :attr:`~src.config.MLConf.XGBoost_hyperparameters`
         * :attr:`~src.config.MLConf.SVM_hyperparameters`
         * :attr:`~src.config.MLConf.DecisionTree_hyperparameters`

        * *RandomizedSearchCV*: Randomized search over continuous distribution space. :attr:`~src.config.MLConf.max_iter`
          defines the number of parameter settings that are sampled. :py:attr:`~src.config.MLConf.max_iter` trades off
          runtime vs quality of the solution. The following variables are defined in :func:`~src.config.MLConf` :

         * :attr:`~src.config.MLConf.MLP_hyperparameters_dist`
         * :attr:`~src.config.MLConf.RandomForests_hyperparameters_dist`
         * :attr:`~src.config.MLConf.XGBoost_hyperparameters_dist`
         * :attr:`~src.config.MLConf.SVM_hyperparameters_dist`
         * :attr:`~src.config.MLConf.DecisionTree_hyperparameters_dist`

        Parameters
        ----------
        X: array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.
        y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values, i.e. class labels.

        Returns
        -------
        out: :obj:`dict` of {:obj:`str`: :obj:`int`, :obj:`str`: :obj:`str`}
            It returns a dictionary with keys *accuracy*: score and *classifier*: the name in reference.

        """
        hyperparams_data = {
            'KFold': {},
            'Avg': []
        }

        fold = 1

        for train_idx, test_idx in self.outer_cv.split(X, y):
            X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]

            for clf_key, clf_val in self.clf_names.iteritems():
                clf = None
                for score in self.scores:
                    if self.search_method.lower() == 'grid':
                        clf = GridSearchCV(
                            clf_val[0](probability=True) if clf_key == 'SVM' else clf_val[0](), clf_val[1],
                            cv=self.inner_cv, scoring=score, verbose=1, n_jobs=self.n_jobs)
                    # elif self.search_method.lower() == 'hyperband' and clf_key in ['XGBoost', 'Extra-Trees', 'Random Forest']:
                    #     HyperbandSearchCV(
                    #         clf_val[0](probability=True) if clf_key == 'SVM' else clf_val[0](), clf_val[2].copy().pop('n_estimators'),
                    #         resource_param='n_estimators',
                    #         min_iter=500 if clf_key == 'XGBoost' else 200,
                    #         max_iter=3000 if clf_key == 'XGBoost' else 1000,
                    #         cv=self.inner_cv, random_state=seed_no, scoring=score
                    #     )
                    else:  # randomized is used as default
                        clf = RandomizedSearchCV(
                            clf_val[0](probability=True) if clf_key == 'SVM' else clf_val[0](), clf_val[2],
                            cv=self.inner_cv, scoring=score, verbose=1, n_jobs=self.n_jobs, n_iter=self.n_iter)
                    clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)

                hyperparams_found = dict()
                hyperparams_found['accuracy'] = accuracy_score(y_test, y_pred)
                hyperparams_found['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
                hyperparams_found['f1_macro'] = f1_score(y_test, y_pred, average='macro')
                hyperparams_found['fold'] = fold
                hyperparams_found['Best Hyperparameters'] = clf.best_params_

                if clf_key in hyperparams_data['KFold']:
                    hyperparams_data['KFold'][clf_key].append(hyperparams_found)
                else:
                    hyperparams_data['KFold'][clf_key] = [hyperparams_found]

            fold += 1

        for clf_key in self.clf_names.keys():
            clf_metrics = dict()
            clf_metrics['accuracy'] = sum(float(x['accuracy']) for x in hyperparams_data['KFold'][clf_key]) / self.kfold
            clf_metrics['f1_weighted'] = sum(
                float(x['f1_weighted']) for x in hyperparams_data['KFold'][clf_key]) / self.kfold
            clf_metrics['f1_macro'] = sum(float(x['f1_macro']) for x in hyperparams_data['KFold'][clf_key]) / self.kfold
            clf_metrics['classifier'] = clf_key

            hyperparams_data['Avg'].append(clf_metrics)

        _, best_clf = max(enumerate(hyperparams_data['Avg']), key=(lambda x: x[1]['accuracy']))

        return best_clf

    def fineTuneClassifier(self, X, y, best_clf):
        """Search over specified parameter values for an estimator/classifier.

        This method searches over specified values to fine tune hyperparameters for
        best accuracy score. The supported search methods are GridSearchCV and RandomizedSearchCV
        as presented in :class:`getBestClassifier`.

        Parameters
        ----------
        X: array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.
        y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values, i.e. class labels.
        best_clf: :obj:`dict` of {:obj:`str`: :obj:`int`, :obj:`str`: :obj:`str`}
            A dictionary with values for keys *accuracy*: score and *classifier*: its name.

        Returns
        -------
        tuple of (str, dict, float)
            It returns the estimator, the parameter setting that gave the best results on the X dataset and the mean
            cross-validated score of the estimator.
        """
        clf = None
        for score in self.scores:
            if self.search_method.lower() == 'grid':
                clf = GridSearchCV(
                    self.clf_names[best_clf['classifier']][0](probability=True) if best_clf['classifier'] == 'SVM'
                    else self.clf_names[best_clf['classifier']][0](),
                    self.clf_names[best_clf['classifier']][1],
                    cv=self.outer_cv, scoring=score, verbose=0, n_jobs=self.n_jobs)
            # elif self.search_method.lower() == 'hyperband' and  best_clf['classifier'] in ['XGBoost', 'Extra-Trees', 'Random Forest']:
            #     HyperbandSearchCV(
            #         self.clf_names[best_clf['classifier']][0](probability=True) if best_clf['classifier'] == 'SVM'
            #         else self.clf_names[best_clf['classifier']][0](),
            #         self.clf_names[best_clf['classifier']][2].copy().pop('n_estimators'),
            #         resource_param='n_estimators',
            #         min_iter=500 if best_clf['classifier'] == 'XGBoost' else 200,
            #         max_iter=3000 if best_clf['classifier'] == 'XGBoost' else 1000,
            #         cv=self.inner_cv, random_state=seed_no, scoring=score
            #     )
            else:  # randomized is used as default
                clf = RandomizedSearchCV(
                    self.clf_names[best_clf['classifier']][0](probability=True) if best_clf['classifier'] == 'SVM'
                    else self.clf_names[best_clf['classifier']][0](),
                    self.clf_names[best_clf['classifier']][2],
                    cv=self.outer_cv, scoring=score, verbose=0, n_jobs=self.n_jobs, n_iter=self.n_iter)
            clf.fit(X, y)

        return clf.best_estimator_, clf.best_params_, clf.best_score_

    def trainClassifier(self, X_train, y_train, model):
        """Build a classifier from the training set (X_train, y_train).

        Parameters
        ----------
        X_train: array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.
        y_train: array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values, i.e. class labels.
        model: object
            An instance of a classifier.

        Returns
        -------
        object
            It returns a trained classifier.
        """
        model.fit(X_train, y_train)
        return model

    def testClassifier(self, X_test, y_test, model):
        """Evaluate a classifier on a testing set (X_test, y_test).

        Parameters
        ----------
        X_test: array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.
        y_test: array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values, i.e. class labels.
        model: object
            A trained classifier.

        Returns
        -------
        tuple of (float, float, float, float)
            It returns the following achieved scores on test dataset: accuracy, precision, recall and f1.
        """
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return acc, pre, rec, f1
