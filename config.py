import numpy as np
from scipy.stats import randint as sp_randint, expon, truncnorm


class initialConfig:
    ## The following parameters correspond to the machine learning
    ## part of the framework.

    # This parameter refers to the number of outer folds that
    # are being used in order for the k-fold cross-validation
    # to take place.
    kfold_parameter = 5
    kfold_inner_parameter = 4

    # Number of parallel jobs to be initiated:
    # -1: use all processors
    # int: no of processors to use
    n_jobs = -1

    test_dataset = './datasets/dataset-string-similarity_original_1k.csv'
    # test_dataset = './datasets/dataset-string-similarity_latin_EU_NA_1k.txt'
    # test_dataset = './datasets/dataset-string-similarity-100.csv'

    # the classification method used: basic, basic_sorted, lgm
    classification_method = 'lgm'

    # This parameter contains a list of the various classifiers
    # the results of which will be compared in the experiments.
    # classifiers = ['SVM', 'Decision Tree', 'Random Forest', 'AdaBoost',
    #                'Naive Bayes', 'MLP', 'Gaussian Process', 'Extra Trees']

    # Search Method to use for best hyperparameters: randomized, grid, hyperband - not yet implemented!!!
    hyperparams_search_method = 'randomized'

    # These are the parameters that constitute the search space for GridSearchCV
    # in our experiments.
    SVM_hyperparameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [300]},
        {'kernel': ['poly'], 'degree': [1, 2, 3, 4], 'gamma': ['scale'],
         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [300]},
        {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': ['scale'], 'max_iter': [300]}
    ]
    DecisionTree_hyperparameters = {
        'max_depth': [i for i in range(1, 33)],
        'min_samples_split': list(np.linspace(0.1, 1, 10)),
        'min_samples_leaf': list(np.linspace(0.1, 0.5, 5)),
        'max_features': [i for i in range(1, 10)]
    }
    RandomForest_hyperparameters = {
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 100, None],
        'criterion': ['gini', 'entropy'],
        'max_features': ['log2', 'sqrt'],  # auto is equal to sqrt
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        "n_estimators": [250, 500, 1000]
    }
    XGBoost_hyperparameters = {
        "n_estimators": [500, 1000, 3000],
        # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        # hyperparameters to avoid overfitting
        'eta': list(np.linspace(0.01, 0.2, 10)),  # 'learning_rate'
        'gamma': [0, 1, 5],
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': list(np.linspace(0.3, 1, 8)),
        'min_child_weight': [1, 5, 10],
    }
    MLP_hyperparameters = {
        'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        'max_iter': [300, 500, 1000],
        'solver': ['sgd', 'adam']
    }

    # These are the parameters that constitute the search space for RandomizedSearchCV
    # in our experiments.
    SVM_hyperparameters_dist = {
        'C': expon(scale=100), 'gamma': expon(scale=.1), 'kernel': ['rbf'], 'class_weight': ['balanced', None]
    }
    DecisionTree_hyperparameters_dist = {
        'max_depth': sp_randint(10, 100),
        'min_samples_split': list(np.linspace(0.1, 1, 50)),
        'min_samples_leaf': list(np.linspace(0.1, 0.5, 25)),
        'max_features': sp_randint(1, 11),
    }
    RandomForest_hyperparameters_dist = {
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 100, None],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],  # sp_randint(1, 11)
        'min_samples_leaf': sp_randint(1, 5),
        'min_samples_split': sp_randint(2, 11),
        "n_estimators": sp_randint(250, 1000),
    }
    XGBoost_hyperparameters_dist = {
        "n_estimators": sp_randint(500, 4000),
        # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        # hyperparameters to avoid overfitting
        'eta': expon(loc=0.01, scale=0.1),  # 'learning_rate'
        'gamma': [0, 1, 5],
        'subsample': truncnorm(0.7, 1),
        'colsample_bytree': truncnorm(0, 1),
        'min_child_weight': [1, 5, 10],
    }
    MLP_hyperparameters_dist = {
        'learning_rate_init': expon(loc=0.0001, scale=0.1),
        'max_iter': [300, 500, 1000],
        'solver': ['sgd', 'adam']
    }

    max_iter = 250
