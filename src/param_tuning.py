# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

from src import config
import numpy as np
import src.feature_selection as fs

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import chi2, SelectKBest, VarianceThreshold
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)

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
        'SVM': [LinearSVC, config.MLConf.SVM_hyperparameters, config.MLConf.SVM_hyperparameters_dist],
        'DecisionTree': [DecisionTreeClassifier, config.MLConf.DecisionTree_hyperparameters,
                          config.MLConf.DecisionTree_hyperparameters_dist],
        'MLP': [MLPClassifier, config.MLConf.MLP_hyperparameters, config.MLConf.MLP_hyperparameters_dist],
        'RandomForest': [RandomForestClassifier, config.MLConf.RandomForest_hyperparameters,
                          config.MLConf.RandomForest_hyperparameters_dist],
        'ExtraTrees': [ExtraTreesClassifier, config.MLConf.RandomForest_hyperparameters,
                        config.MLConf.RandomForest_hyperparameters_dist],
        'XGBoost': [XGBClassifier, config.MLConf.XGBoost_hyperparameters, config.MLConf.XGBoost_hyperparameters_dist]
    }

    clf_callable_map = {
        # 'Naive Bayes': GaussianNB(),
        # 'Gaussian Process': GaussianProcessClassifier(),
        # 'AdaBoost': AdaBoostClassifier(),
        # 'Nearest Neighbors': KNeighborsClassifier(),
        # 'Logistic Regression': LogisticRegression(solver='liblinear', multi_class='auto'),
        'SVM': LinearSVC(),
        'MLP': MLPClassifier(),
        # 'Decision Tree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(random_state=1),
        'Extra Trees': ExtraTreesClassifier(),
        'XGBoost': XGBClassifier()
    }

    """
      Feature selection mapping
      """

    fs_callable_map = {
        'SelectKBest': [SelectKBest(chi2), config.MLConf.SelectKbest_hyperparameters],
        'VarianceThreshold': [VarianceThreshold(), config.MLConf.VT_hyperparameters],
        'RFE': ['RFE', {}],
        'SelectFromModel': ['SelectFromModel', config.MLConf.SelectFromModel_hyperparameters],
        'PCA': ['PCA', config.MLConf.PCA_hyperparameters]
    }

    feature_selection_getter_map = {
        'SelectKBest': 'get_stats_features',
        'VarianceThreshold': 'get_stats_features',
        'RFE': 'get_RFE_features',
        'SelectFromModel': 'get_SFM_features',
        'PCA': 'get_PCA_features'
    }

    feature_selection_getter_args_map = {
        'SelectKBest': ('fs_name', 'fsm', 'clf', 'params', 'X_train', 'y_train', 'X_test'),
        'VarianceThreshold': ('fs_name', 'fsm', 'clf', 'params', 'X_train', 'y_train', 'X_test'),
        'RFE': ('clf', 'clf_name', 'X_train', 'y_train', 'X_test'),
        'SelectFromModel': ('clf', 'clf_name', 'params', 'X_train', 'y_train', 'X_test'),
        'PCA': ('clf', 'params', 'X_train', 'y_train', 'X_test'),
    }

    def __init__(self):
        # To be used in outer CV
        self.outer_cv = StratifiedKFold(n_splits=config.MLConf.kfold_parameter, shuffle=False,
                                        random_state=config.seed_no)

        self.kfold = config.MLConf.kfold_parameter
        self.n_jobs = config.MLConf.n_jobs

        self.search_method = config.MLConf.hyperparams_search_method
        self.n_iter = config.MLConf.max_iter

        self.feature_selection = config.MLConf.feature_selection
        self.feature_selection_method = config.MLConf.feature_selection_method

    def fineTuneClassifiers(self, X_train_all, y, X_test_all):
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

                if self.feature_selection:
                    X_train, y_train, X_test = self.ft_selection(clf_key, self.feature_selection_method, X_train_all,
                                                                 y,
                                                                 X_test_all)
                    print(f'Number of features {X_train.shape[1]}')
                else:
                    X_train, X_test = X_train_all, X_test_all

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
                clf.fit(X_train, y)

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

        return best_clf, X_train, X_test

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

    def ft_selection(self, clf_name, fs_method, X_train, y_train, X_test):

        clf = self.clf_names[clf_name][0]()
        fsm = self.fs_callable_map[fs_method][0]
        params = self.fs_callable_map[fs_method][1]

        args = self.create_ft_selection_args_dict(fs_method, fsm, clf, clf_name, params, X_train, y_train, X_test)
        X_train, y_train, X_test = getattr(fs, self.feature_selection_getter_map[fs_method])(
            *[args[arg] for arg in self.feature_selection_getter_args_map[fs_method]]
        )

        return X_train, y_train, X_test

    def create_ft_selection_args_dict(self, fs_name, fsm, clf, clf_name, params, X_train, y_train, X_test):
        return locals()
