# # classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier

# # Feature selection
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

evaluated_classifiers = {
    'LogisticRegression':
        {
            'name': 'LogisticRegression',
            'estimator': LogisticRegression(),
            'selector': SelectKBest(),
            'decomposition': PCA(),
            'params':
                {
                    "classifier__penalty": ['none'],
                    "classifier__tol": [0.00005],
                    "classifier__C": [0.00085],
                    "classifier__class_weight": [None],
                    "classifier__solver": ['saga'],
                    "classifier__max_iter": [630],
                    'selector__k': [920],
                    'decomposition__n_components': [86],
                }},

    # 'SVC':
    #     {
    #         'name': 'SVC',
    #         'estimator': SVC(),
    #         'selector': SelectKBest(),
    #         'decomposition': PCA(),  # None because PCA() lowers the scores
    #         'params':
    #             {
    #                 "classifier__C": [2],  # [0.01, 0.1, 1, 2, 4]  best: 2
    #                 "classifier__kernel": ['rbf'],  # ['rbf','poly','linear']  best: 'rbf'
    #                 "classifier__degree": [1],  # [1,2,3,5]  best: 1
    #                 "classifier__max_iter": [-1],  # [-1, 1, 10, 100, 1000]  best: -1
    #                 "classifier__gamma": ['auto'],  # []  best: 'auto'
    #                 "classifier__tol": [0.001],  # [0.001, 0.01,0.1 ,1]  best: 0.1
    #                 "classifier__probability": [True],  # [True, False]  best: True
    #                 'selector__k': [30],  # [30, 50, 90, 100, 150, 200]  best: 90
    #             }},

    # 'RandomForestClassifier':
    #     {
    #         'name': 'RandomForestClassifier',
    #         'estimator': RandomForestClassifier(),
    #         'selector': SelectKBest(),
    #         'decomposition': PCA(),
    #         'params':
    #             {
    #                 'classifier__n_estimators': [300],  # [list(range(100, 900, 200))]  best: 300
    #                 'classifier__criterion': ['gini'],  # ['gini']  best:'gini'
    #                 'classifier__max_features': [0.5],  # [0.3, 0.5, 0.7]  best: 0.5
    #                 'classifier__max_depth': [100],  # [1,10,100]  best: 100
    #                 'classifier__max_leaf_nodes': [250],  # [50, 150, 250]  best: 250
    #                 'classifier__min_samples_split': [0.1],  # [0.1, 0.5, 0.09, 1, 5, 10]  best: 5
    #                 'classifier__bootstrap': [True],  # [True, False]  best: True
    #                 'classifier__max_samples': [250],  # [50, 150, 250]  best: 250
    #                 'selector__k': [100],  # [50,100,120,130,150,200, 300, 400, 600]  best: 100
    #             }},

    # 'XGBClassifier':
    #     {
    #         'name': 'XGBClassifier',
    #         'estimator': XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor'),
    #         'selector': SelectKBest(),
    #         'decomposition': PCA(),  # None because PCA() lowers the scores
    #         'params':
    #             {
    #                 "classifier__n_estimators": [600],  # [200, 500, 600, 800, 1000]  best: 600
    #                 "classifier__max_depth": [4],  # [2,4,8]  best: 4
    #                 "classifier__learning_rate": [0.01],  # [0.01, 0.1, 0.5]  best: 0.01
    #                 "classifier__subsample": [1],  # []  best: 1
    #                 "classifier__colsample_bytree": [1],  # [0.01, 0.1, 1]  best: 1
    #                 'selector__k': [30],  # [10, 20, 25, 30, 35, 40, 50, 90]  best: 30
    #             }},

    # 'CatBoostClassifier':
    #     {
    #         'name': 'CatBoostClassifier',
    #         'estimator': CatBoostClassifier(task_type="GPU",
    #                                         devices='0:2'),
    #         'selector': SelectKBest(),
    #         'decomposition': PCA(),  # None because PCA() lowers the scores
    #         'params':
    #             {
    #                 "classifier__n_estimators": [600],  # [200, 500, 600, 800, 1000]  best: 600
    #                 "classifier__max_depth": [4],  # [2,4,8]  best: 4
    #                 "classifier__learning_rate": [0.01],  # [0.01, 0.1, 0.5]  best: 0.01
    #                 'selector__k': [30],  # [10, 20, 25, 30, 35, 40, 50, 90]  best: 30
    #             }},

    # TODO add parameters belonging to the classifiers listed below

    # 'ExtraTreesClassifier':
    #     {
    #         'name': 'ExtraTreesClassifier',
    #         'estimator': ExtraTreesClassifier(task_type="GPU",
    #                                         devices='0:2'),
    #         'selector': SelectKBest(),
    #         'decomposition': PCA(),  # None because PCA() lowers the scores
    #         'params':
    #             {
    #                 "classifier__n_estimators": [600],  # [200, 500, 600, 800, 1000]  best: 600
    #                 "classifier__max_depth": [4],  # [2,4,8]  best: 4
    #                 "classifier__learning_rate": [0.01],  # [0.01, 0.1, 0.5]  best: 0.01
    #                 'selector__k': [30],  # [10, 20, 25, 30, 35, 40, 50, 90]  best: 30
    #             }},

    # 'AdaBoostClassifier':
    #     {
    #         'name': 'AdaBoostClassifier',
    #         'estimator': AdaBoostClassifier(task_type="GPU",
    #                                         devices='0:2'),
    #         'selector': SelectKBest(),
    #         'decomposition': PCA(),  # None because PCA() lowers the scores
    #         'params':
    #             {
    #                 "classifier__n_estimators": [600],  # [200, 500, 600, 800, 1000]  best: 600
    #                 "classifier__max_depth": [4],  # [2,4,8]  best: 4
    #                 "classifier__learning_rate": [0.01],  # [0.01, 0.1, 0.5]  best: 0.01
    #                 'selector__k': [30],  # [10, 20, 25, 30, 35, 40, 50, 90]  best: 30
    #             }},

    # 'DecisionTreeClassifier':
    #     {
    #         'name': 'DecisionTreeClassifier',
    #         'estimator': DecisionTreeClassifier(task_type="GPU",
    #                                         devices='0:2'),
    #         'selector': SelectKBest(),
    #         'decomposition': PCA(),  # None because PCA() lowers the scores
    #         'params':
    #             {
    #                 "classifier__n_estimators": [600],  # [200, 500, 600, 800, 1000]  best: 600
    #                 "classifier__max_depth": [4],  # [2,4,8]  best: 4
    #                 "classifier__learning_rate": [0.01],  # [0.01, 0.1, 0.5]  best: 0.01
    #                 'selector__k': [30],  # [10, 20, 25, 30, 35, 40, 50, 90]  best: 30
    #             }},
}

not_evaluated_classifiers = {
    # 'SVC':
    #     {
    #         'name': 'SVC',
    #         'estimator': SVC(),
    #         'selector': SelectKBest(),
    #         'decomposition': PCA(),  # None because PCA() lowers the scores
    #         'params':
    #             {
    #                 "classifier__C": [[0.01, 0.1, 1]],  # [0.01, 0.1, 1, 2, 4]  best: 2
    #                 "classifier__kernel": ['rbf', 'poly', 'linear'],  # ['rbf','poly','linear']  best: 'rbf'
    #                 "classifier__degree": [1, 3, 5],  # [1,2,3,5]  best: 1
    #                 "classifier__max_iter": [-1, 1, 10, 100],  # [-1, 1, 10, 100, 1000]  best: -1
    #                 "classifier__gamma": ['auto'],
    #                 "classifier__tol": [0.001, 0.1, 1],  # [0.001, 0.01,0.1 ,1]  best: 0.1
    #                 "classifier__probability": [True, False],  # [True, False]  best: True
    #                 'selector__k': [30, 200],  # [30, 50, 90, 100, 150, 200]  best: 90
    #             }},

    # 'RandomForestClassifier':
    #     {
    #         'name': 'RandomForestClassifier',
    #         'estimator': RandomForestClassifier(),
    #         'selector': SelectKBest(),
    #         'decomposition': PCA(),
    #         'params':
    #             {
    #                 'classifier__n_estimators': [300],  # [list(range(100, 900, 200))]  best: 300
    #                 'classifier__criterion': ['gini'],  # ['gini']  best:'gini'
    #                 'classifier__max_features': [0.5],  # [0.3, 0.5, 0.7]  best: 0.5
    #                 'classifier__max_depth': [100],  # [1,10,100]  best: 100
    #                 'classifier__max_leaf_nodes': [250],  # [50, 150, 250]  best: 250
    #                 'classifier__min_samples_split': [0.1],  # [0.1, 0.5, 0.09, 1, 5, 10]  best: 5
    #                 'classifier__bootstrap': [True],  # [True, False]  best: True
    #                 'classifier__max_samples': [250],  # [50, 150, 250]  best: 250
    #                 'selector__k': [100],  # [50,100,120,130,150,200, 300, 400, 600]  best: 100
    #             }},

    'XGBClassifier':
        {
            'name': 'XGBClassifier',
            'estimator': XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor'),
            'selector': SelectKBest(),
            'decomposition': PCA(),  # None because PCA() lowers the scores
            'params':
                {
                    "classifier__n_estimators": [200, 600, 1000],  # [200, 500, 600, 800, 1000]  best: 600
                    "classifier__max_depth": [2,4,8],  # [2,4,8]  best: 4
                    "classifier__learning_rate": [0.01, 0.1, 0.5],  # [0.01, 0.1, 0.5]  best: 0.01
                    "classifier__subsample": [1],
                    "classifier__colsample_bytree": [0.01, 0.1, 1],  # [0.01, 0.1, 1]  best: 1
                    'selector__k': [10, 50, 90],  # [10, 20, 25, 30, 35, 40, 50, 90]  best: 30
                }},

    # 'CatBoostClassifier':
    #     {
    #         'name': 'CatBoostClassifier',
    #         'estimator': CatBoostClassifier(task_type="GPU",
    #                                         devices='0:2'),
    #         'selector': SelectKBest(),
    #         'decomposition': PCA(),  # None because PCA() lowers the scores
    #         'params':
    #             {
    #                 "classifier__n_estimators": [600],  # [200, 500, 600, 800, 1000]  best: 600
    #                 "classifier__max_depth": [4],  # [2,4,8]  best: 4
    #                 "classifier__learning_rate": [0.01],  # [0.01, 0.1, 0.5]  best: 0.01
    #                 'selector__k': [30],  # [10, 20, 25, 30, 35, 40, 50, 90]  best: 30
    #             }},

    # TODO add parameters belonging to the classifiers listed below

    # 'ExtraTreesClassifier':
    #     {
    #         'name': 'ExtraTreesClassifier',
    #         'estimator': ExtraTreesClassifier(task_type="GPU",
    #                                         devices='0:2'),
    #         'selector': SelectKBest(),
    #         'decomposition': PCA(),  # None because PCA() lowers the scores
    #         'params':
    #             {
    #                 "classifier__n_estimators": [600],  # [200, 500, 600, 800, 1000]  best: 600
    #                 "classifier__max_depth": [4],  # [2,4,8]  best: 4
    #                 "classifier__learning_rate": [0.01],  # [0.01, 0.1, 0.5]  best: 0.01
    #                 'selector__k': [30],  # [10, 20, 25, 30, 35, 40, 50, 90]  best: 30
    #             }},

    # 'AdaBoostClassifier':
    #     {
    #         'name': 'AdaBoostClassifier',
    #         'estimator': AdaBoostClassifier(task_type="GPU",
    #                                         devices='0:2'),
    #         'selector': SelectKBest(),
    #         'decomposition': PCA(),  # None because PCA() lowers the scores
    #         'params':
    #             {
    #                 "classifier__n_estimators": [600],  # [200, 500, 600, 800, 1000]  best: 600
    #                 "classifier__max_depth": [4],  # [2,4,8]  best: 4
    #                 "classifier__learning_rate": [0.01],  # [0.01, 0.1, 0.5]  best: 0.01
    #                 'selector__k': [30],  # [10, 20, 25, 30, 35, 40, 50, 90]  best: 30
    #             }},

    # 'DecisionTreeClassifier':
    #     {
    #         'name': 'DecisionTreeClassifier',
    #         'estimator': DecisionTreeClassifier(task_type="GPU",
    #                                         devices='0:2'),
    #         'selector': SelectKBest(),
    #         'decomposition': PCA(),  # None because PCA() lowers the scores
    #         'params':
    #             {
    #                 "classifier__n_estimators": [600],  # [200, 500, 600, 800, 1000]  best: 600
    #                 "classifier__max_depth": [4],  # [2,4,8]  best: 4
    #                 "classifier__learning_rate": [0.01],  # [0.01, 0.1, 0.5]  best: 0.01
    #                 'selector__k': [30],  # [10, 20, 25, 30, 35, 40, 50, 90]  best: 30
    #             }},
}
