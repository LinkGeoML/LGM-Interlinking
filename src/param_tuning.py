# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

from src import config
import numpy as np

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


np.random.seed(config.seed_no)


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
        'DecisionTree': [DecisionTreeClassifier, config.MLConf.DecisionTree_hyperparameters,
                          config.MLConf.DecisionTree_hyperparameters_dist],
        'MLP': [MLPClassifier, config.MLConf.MLP_hyperparameters, config.MLConf.MLP_hyperparameters_dist],
        'RandomForest': [RandomForestClassifier, config.MLConf.RandomForest_hyperparameters,
                          config.MLConf.RandomForest_hyperparameters_dist],
        'ExtraTrees': [ExtraTreesClassifier, config.MLConf.RandomForest_hyperparameters,
                        config.MLConf.RandomForest_hyperparameters_dist],
        'XGBoost': [XGBClassifier, config.MLConf.XGBoost_hyperparameters, config.MLConf.XGBoost_hyperparameters_dist]
    }

    def __init__(self):
        # To be used in outer CV
        self.outer_cv = StratifiedKFold(n_splits=config.MLConf.kfold_parameter, shuffle=False,
                                        random_state=config.seed_no)

        self.kfold = config.MLConf.kfold_parameter
        self.n_jobs = config.MLConf.n_jobs

        self.search_method = config.MLConf.hyperparams_search_method
        self.n_iter = config.MLConf.max_iter

    def fineTuneClassifiers(self, X, y):
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
            It returns a dictionary with keys *accuracy*, i.e., the used similarity score, and *classifier*, i.e.,
            the name of the model in reference.
        """
        hyperparams_data = list()

        for clf_key in config.MLConf.classifiers:
            try:
                clf = None
                if self.search_method.lower() == 'grid':
                    clf = GridSearchCV(
                        self.clf_names[clf_key][0](), self.clf_names[clf_key][1],
                        cv=self.outer_cv, scoring=config.MLConf.score, verbose=1, n_jobs=self.n_jobs
                    )
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
                        self.clf_names[clf_key][0](), self.clf_names[clf_key][2],
                        cv=self.outer_cv, scoring=config.MLConf.score, verbose=1, n_jobs=self.n_jobs, n_iter=self.n_iter
                    )
                clf.fit(X, y)

                hyperparams_found = dict()
                hyperparams_found['score'] = clf.best_score_
                hyperparams_found['results'] = clf.cv_results_
                hyperparams_found['hyperparams'] = clf.best_params_
                hyperparams_found['estimator'] = clf.best_estimator_
                hyperparams_found['classifier'] = clf_key
                hyperparams_found['scorers'] = clf.scorer_

                hyperparams_data.append(hyperparams_found)
            except KeyError as e:
                print("type error: {} for key: {}".format(str(e), clf_key))

        _, best_clf = max(enumerate(hyperparams_data), key=(lambda x: x[1]['score']))

        return best_clf

    def trainClassifier(self, X_train, y_train, model):
        """Build a classifier from the training set (X_train, y_train).

        Parameters
        ----------
        X_train: array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.
        y_train: array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values, i.e. class labels.
        model: classifier object
            An instance of a classifier.

        Returns
        -------
        classifier object
            It returns a trained classifier.
        """
        if hasattr(model, "n_jobs"): model.set_params(n_jobs=config.MLConf.n_jobs)

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
        model: classifier object
            A trained classifier.

        Returns
        -------
        tuple of (float, float, float, float)
            Returns the computed metrics, i.e., *accuracy*, *precision*, *recall* and *f1*, for the specified model on the test
            dataset.
        """
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return acc, pre, rec, f1
