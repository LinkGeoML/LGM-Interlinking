# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import numpy as np
from scipy.stats import randint as sp_randint, expon, truncnorm, uniform


#: str: The folder name, relative to root path, that contains all required input files, e.g., train/test dataset,
#: frequent terms etc.
default_data_path = 'data'
#: int: The proportion of frequent terms to utilize.
freq_term_size = 400
#: list of str: A list of names assigned to each column in train/test dataset. If a header exists, it should be set to
#: None.
fieldnames = ["s1", "s2", "status", "c1", "c2", "a1", "a2", "cc1", "cc2"]

use_cols = dict(s1='s1', s2='s2', label='status')
"""A dictionary of useful column names.

    :var s1: Denote the column name assigned to the first name of each of the toponym pair.
    :vartype s1: :obj:`str`
    :var s2: Denote the column name assigned to the second name of each of the toponym pair.
    :vartype s2: :obj:`str`
    :var label: The true labels for the toponym pairs.
    :vartype label: :obj:`str`
"""
#: char: The delimiter used to separate each column in CSV input files.
delimiter = '\t'

# #: Relative path to the train dataset. This value is used only when the *dtrain* cmd argument is None.
# train_dataset = 'data/dataset-string-similarity_global_1k.csv'
# # train_dataset = 'data/dataset-string-similarity_latin_EU_NA_1k.txt'
# # train_dataset = 'data/dataset-string-similarity-100.csv'
#
# #: Relative path to the test dataset. This value is used only when the *dtest* cmd argument is None.
# test_dataset = 'data/dataset-string-similarity.txt'

#: float: Similarity threshold on whether sorting on toponym tokens is applied or not. It is triggered on a score
#: below the assigned threshold.
sort_thres = 0.55

#: int: Seed used by each of the random number generators.
seed_no = 13


class MLConf:
    """
    This class initializes parameters that correspond to the machine learning part of the framework.

    :cvar opt_values: A list of learned parameters for LGM-Sim meta-similarity function. :math:`θ_{split}, w_b, w_m, w_f`.
    :vartype opt_values: :obj:`dict` of dicts
    :cvar clf_custom_params: A list of custom hyper-parameters to utilize for specified classifiers. These parameters
        are used when `evaluate` command is executed in cli.
    :vartype clf_custom_params: :obj:`dict` of dicts

    These variables define the parameter grid for GridSearchCV:

    :cvar SVM_hyperparameters: Defines the search space for SVM.
    :vartype SVM_hyperparameters: :obj:`list`
    :cvar MLP_hyperparameters: Defines the search space for MLP.
    :vartype MLP_hyperparameters: :obj:`dict`
    :cvar DecisionTree_hyperparameters: Defines the search space for Decision Trees.
    :vartype DecisionTree_hyperparameters: :obj:`dict`
    :cvar RandomForest_hyperparameters: Defines the search space for Random Forests and Extra-Trees.
    :vartype RandomForest_hyperparameters: :obj:`dict`
    :cvar XGBoost_hyperparameters: Defines the search space for XGBoost.
    :vartype XGBoost_hyperparameters: :obj:`dict`

    These variables define the parameter grid for RandomizedSearchCV where continuous distributions are used for
    continuous parameters (whenever this is feasible):

    :cvar SVM_hyperparameters_dist: Defines the search space for SVM.
    :vartype SVM_hyperparameters_dist: :obj:`dict`
    :cvar MLP_hyperparameters_dist: Defines the search space for MLP.
    :vartype MLP_hyperparameters_dist: :obj:`dict`
    :cvar DecisionTree_hyperparameters_dist: Defines the search space for Decision Trees.
    :vartype DecisionTree_hyperparameters_dist: :obj:`dict`
    :cvar RandomForest_hyperparameters_dist: Defines the search space for Random Forests and Extra-Trees.
    :vartype RandomForest_hyperparameters_dist: :obj:`dict`
    :cvar XGBoost_hyperparameters_dist: Defines the search space for XGBoost.
    :vartype XGBoost_hyperparameters_dist: :obj:`dict`
    """

    kfold_no = 5  #: int: The number of outer folds that splits the dataset for the k-fold cross-validation.

    n_jobs = 4  #: int: Number of parallel jobs to be initiated. -1 means to utilize all available processors.

    classification_method = 'lgm'
    """str: The classification group of features to use. (*basic* | *basic_sorted* | *lgm*).

    See Also
    --------
    :class:`~interlinking.features.Features`: Details on available inputs.    
    """

    # accepted values: randomized, grid, hyperband - not yet implemented!!!
    hyperparams_search_method = 'randomized'
    """str: Search Method to use for finding best hyperparameters. (*randomized* | *grid*).
    
    See Also
    --------
    :func:`~interlinking.hyperparam_tuning.ParamTuning.fineTuneClassifiers`: Details on available inputs.       
    """
    #: int: Number of iterations that RandomizedSearchCV should execute. It applies only when
    # :class:`hyperparams_search_method` equals to 'randomized'.
    max_iter = 250

    classifiers = [
        # 'SVM',
        # 'DecisionTree',
        'RandomForest',
        # 'ExtraTrees',
        # 'XGBoost',
        # 'MLP'
    ]
    """list of str: Define the classifiers to apply on code execution. Accepted values are: 

    - SVM 
    - DecisionTree
    - RandomForest
    - ExtraTrees
    - XGBoost
    - MLP.
    """

    score = 'accuracy'
    """str: The metric to optimize on hyper-parameter tuning. Possible valid values presented on `Scikit predefined values`_. 

    .. _Scikit predefined values:
        https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
    """

    clf_custom_params = {
        'SVM': {
            # basic
            'C': 1.0, 'max_iter': 3000,
            # 'gamma': 0.2456161956918959, 'max_iter': 3000, 'C': 199.0212721894755, 'kernel': 'rbf',
            # 'class_weight': None, 'tol': 0.0001, 'degree': 2,
            'random_state': seed_no
        },
        'DecisionTree': {
            # basic
            'max_depth': 100, 'max_features': 'auto',
            'random_state': seed_no,
        },
        'RandomForest': {
            # basic
            'n_estimators': 300, 'max_depth': 100, 'oob_score': True, 'bootstrap': True,
            'random_state': seed_no, 'n_jobs': n_jobs,  # 'oob_score': True,
        },
        'ExtraTrees': {
            # basic
            'n_estimators': 300, 'max_depth': 100,
            'random_state': seed_no, 'n_jobs': n_jobs
        },
        'XGBoost': {
            # basic
            'n_estimators': 3000,
            'seed': seed_no, 'nthread': n_jobs
        },
        'MLP': {
            'tol': 0.0001, 'learning_rate_init': 0.06794912926673598, 'max_iter': 1000, 'activation': 'logistic',
            'solver': 'lbfgs',
            'random_state': seed_no,
        },
    }

    opt_values = {
        'latin': {
            # Only latin dataset 100k lines
            'damerau_levenshtein': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.5, 0.2, 0.3]]},
            'jaro': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.9, [0.7, 0.1, 0.2]]},
            'jaro_winkler': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.9, [0.6, 0.1, 0.3]]},
            'jaro_winkler_r': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.7, 0.1, 0.2]]},
            # 'permuted_winkler': [],
            # 'sorted_winkler': [],
            'cosine': {'simple': [0.6, [0.6, 0.2, 0.2]], 'avg': [0.9, [0.4, 0.2, 0.4]]},
            'jaccard': {'simple': [0.6, [0.6, 0.1, 0.3]], 'avg': [0.9, [0.3, 0.3, 0.4]]},
            'strike_a_match': {'simple': [0.6, [0.6, 0.1, 0.3]], 'avg': [0.9, [0.5, 0.1, 0.4]]},
            'skipgram': {'simple': [0.6, [0.6, 0.2, 0.2]], 'avg': [0.9, [0.3, 0.3, 0.4]]},
            'monge_elkan': {'simple': [0.6, [0.7, 0.2, 0.1]], 'avg': [0.9, [0.6, 0.1, 0.3]]},
            'soft_jaccard': {'simple': [0.8, [0.6, 0.1, 0.3]], 'avg': [0.9, [0.5, 0.1, 0.4]]},
            'davies': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.9, [0.6, 0.1, 0.3]]},
            'tuned_jaro_winkler': {'simple': [0.8, [0.7, 0.1, 0.2]], 'avg': [0.9, [0.6, 0.1, 0.3]]},
            'tuned_jaro_winkler_r': {'simple': [0.6, [0.7, 0.1, 0.2]], 'avg': [0.8, [0.7, 0.1, 0.2]]},
        },
        'global': {
            'damerau_levenshtein': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
            'jaro': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
            'jaro_winkler': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'jaro_winkler_r': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
            # 'permuted_winkler': [],
            # 'sorted_winkler': [],
            'cosine': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'jaccard': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'strike_a_match': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.65, [0.4, 0.5, 0.1]]},
            'skipgram': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'monge_elkan': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'soft_jaccard': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.7, [0.4, 0.5, 0.1]]},
            'davies': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.7, [0.4, 0.5, 0.1]]},
            'tuned_jaro_winkler': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.6, [0.4, 0.5, 0.1]]},
            'tuned_jaro_winkler_r': {'simple': [0.6, [0.4, 0.5, 0.1]], 'avg': [0.8, [0.4, 0.5, 0.1]]},
        }
    }

    # These parameters constitute the search space for GridSearchCV in our experiments.
    SVM_hyperparameters = [
        {
            'kernel': ['rbf', 'sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 1000], 'max_iter': [3000]
        },
        {
            'kernel': ['poly'], 'degree': [1, 2, 3], 'gamma': ['scale', 'auto'], 'C': [0.01, 0.1, 1, 10, 25, 50, 100],
            'max_iter': [3000]
        },
    ]
    DecisionTree_hyperparameters = {
        'max_depth': np.arange(1, 33),
        'min_samples_split': [2, 5, 10, 20, 50, 100, 200],
        'min_samples_leaf': np.arange(1, 11, 3),
        'max_features': list(np.arange(2, 11, 2)) + ["sqrt", "log2", None]
    }
    RandomForest_hyperparameters = {
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 100],
        "n_estimators": [250, 500, 1000],
        'criterion': ['gini', 'entropy'],
        'max_features': ['log2', 'sqrt'],  # auto is equal to sqrt
        # 'min_samples_leaf': [1, 2, 4, 10],
        'min_samples_split': [2, 5, 10, 50],
    }
    XGBoost_hyperparameters = {
        "n_estimators": [500, 1000, 3000],
        'max_depth': [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        # hyperparameters to avoid overfitting
        # 'eta': list(np.linspace(0.01, 0.2, 10)),  # 'learning_rate'
        # 'gamma': [0, 1, 5],
        'subsample': [0.8, 0.9, 1],
        # # Values from 0.3 to 0.8 if you have many columns (especially if you did one-hot encoding),
        # # or 0.8 to 1 if you only have a few columns
        'colsample_bytree': list(np.linspace(0.8, 1, 3)),
        'min_child_weight': [1, 5, 10],
    }
    MLP_hyperparameters = {
        'learning_rate_init': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
        'max_iter': [300, 500, 1000],
        'solver': ['sgd', 'adam']
    }

    # These parameters constitute the search space for RandomizedSearchCV in our experiments.
    SVM_hyperparameters_dist = {
        'C': expon(scale=100), 'gamma': expon(scale=.1), 'kernel': ['rbf'], 'class_weight': ['balanced', None],
        'max_iter': [10000]
    }
    DecisionTree_hyperparameters_dist = {
        'max_depth': sp_randint(10, 100),
        'min_samples_split': sp_randint(2, 200),
        'min_samples_leaf': sp_randint(1, 10),
        'max_features': sp_randint(1, 11),
    }
    RandomForest_hyperparameters_dist = {
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 100, None],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],  # sp_randint(1, 11)
        'min_samples_leaf': sp_randint(1, 10),
        'min_samples_split': sp_randint(2, 100),
        "n_estimators": sp_randint(250, 1000),
    }
    XGBoost_hyperparameters_dist = {
        "n_estimators": sp_randint(500, 4000),
        'max_depth': sp_randint(3, 100),
        # 'eta': expon(loc=0.01, scale=0.1),  # 'learning_rate'
        # hyperparameters to avoid overfitting
        'gamma': sp_randint(0, 5),
        'subsample': truncnorm(0.7, 1),
        'colsample_bytree': truncnorm(0.8, 1),
        'min_child_weight': sp_randint(1, 10),
    }
    MLP_hyperparameters_dist = {
        'learning_rate_init': expon(loc=0.0001, scale=0.1),
        'max_iter': sp_randint(300, 1000),
        'solver': ['sgd', 'adam']
    }
