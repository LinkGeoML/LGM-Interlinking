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
    clf_names = {
        'SVM': [SVC, config.initialConfig.SVM_hyperparameters, config.initialConfig.SVM_hyperparameters_dist],
        'Decision Tree': [DecisionTreeClassifier, config.initialConfig.DecisionTree_hyperparameters,
                          config.initialConfig.DecisionTree_hyperparameters_dist],
        # 'MLP': [MLPClassifier, config.initialConfig.MLP_hyperparameters, config.initialConfig.MLP_hyperparameters_dist],
        # 'Random Forest': [RandomForestClassifier, config.initialConfig.RandomForest_hyperparameters,
        #                   config.initialConfig.RandomForest_hyperparameters_dist],
        # 'Extra-Trees': [ExtraTreesClassifier, config.initialConfig.RandomForest_hyperparameters,
        #                 config.initialConfig.RandomForest_hyperparameters_dist],
        # 'XGBoost': [XGBClassifier, config.initialConfig.XGBoost_hyperparameters,
        #             config.initialConfig.XGBoost_hyperparameters_dist]
    }

    scores = ['accuracy']  # , 'f1_macro', 'f1_micro']

    def __init__(self):
        # To be used within GridSearch
        self.inner_cv = StratifiedKFold(n_splits=config.initialConfig.kfold_inner_parameter, shuffle=False,
                                        random_state=seed_no)

        # To be used in outer CV
        self.outer_cv = StratifiedKFold(n_splits=config.initialConfig.kfold_parameter, shuffle=False,
                                        random_state=seed_no)

        self.kfold = config.initialConfig.kfold_parameter
        self.n_jobs = config.initialConfig.n_jobs

        self.search_method = config.initialConfig.hyperparams_search_method
        self.n_iter = config.initialConfig.max_iter

    def getBestClassifier(self, X, y):
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

    def fineTuningBestClassifier(self, X, y, best_clf):
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

    def trainBestClassifier(self, X_train, y_train, model):
        model.fit(X_train, y_train)
        return model

    def testBestClassifier(self, X_test, y_test, model):
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return acc, pre, rec, f1
