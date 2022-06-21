# # #
# # # # Machine Learning Part
# # #
import os
import pandas as pd
from imblearn.pipeline import Pipeline

# # Preprocessing
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler

# # Feature selection
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# # Metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# # classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier

import utils

# Output paths
MODEL_PATH = 'data_output/step_5_ml/models'
SCORE_PATH = 'data_output/step_5_ml/scores'

SEED = 123

# The params set below are the params for which I have  received the best ROC-AUC score
classifiers = {
    'LogisticRegression':
        {
            'name': 'LogisticRegression',
            'estimator': LogisticRegression(),
            'selector': SelectKBest(),
            'decomposition': PCA(),
            'params':
                {
                    "classifier__penalty": ['l1'],  # ['l1','l2', 'elasticnet', 'none'] best: l1
                    "classifier__tol": [0.01],  # [0.00001,0.0001,0.001,0.01, 0.1]  best: 0.01
                    "classifier__C": [9],  # [0.01,0.01,1,10]  best:
                    "classifier__class_weight": ['balanced'],  # ['balanced', None]  best: 'balanced'
                    "classifier__solver": ['saga'],  # []  best: 'saga'
                    "classifier__max_iter": [500],  # [60, 90, 100, 150] best:
                    'selector__k': [600],  # [200, 400, 600]  best:
                    'decomposition__n_components': [100],  # [5,10,50,100,200]  best: 100
                }},

    'SVC':
        {
            'name': 'SVC',
            'estimator': SVC(),
            'selector': SelectKBest(),
            'decomposition': PCA(),  # None because PCA() lowers the scores
            'params':
                {
                    "classifier__C": [2],  # [0.01, 0.1, 1, 2, 4]  best: 2
                    "classifier__kernel": ['rbf'],  # ['rbf','poly','linear']  best: 'rbf'
                    "classifier__degree": [1],  # [1,2,3,5]  best: 1
                    "classifier__max_iter": [-1],  # [-1, 1, 10, 100, 1000]  best: -1
                    "classifier__gamma": ['auto'],  # []  best: 'auto'
                    "classifier__tol": [0.001],  # [0.001, 0.01,0.1 ,1]  best: 0.1
                    "classifier__probability": [True],  # [True, False]  best: True
                    'selector__k': [30],  # [30, 50, 90, 100, 150, 200]  best: 90
                }},

    'RandomForestClassifier':
        {
            'name': 'RandomForestClassifier',
            'estimator': RandomForestClassifier(),
            'selector': SelectKBest(),
            'decomposition': PCA(),
            'params':
                {
                    'classifier__n_estimators': [300],  # [list(range(100, 900, 200))]  best: 300
                    'classifier__criterion': ['gini'],  # ['gini']  best:'gini'
                    'classifier__max_features': [0.5],  # [0.3, 0.5, 0.7]  best: 0.5
                    'classifier__max_depth': [100],  # [1,10,100]  best: 100
                    'classifier__max_leaf_nodes': [250],  # [50, 150, 250]  best: 250
                    'classifier__min_samples_split': [0.1],  # [0.1, 0.5, 0.09, 1, 5, 10]  best: 5
                    'classifier__bootstrap': [True],  # [True, False]  best: True
                    'classifier__max_samples': [250],  # [50, 150, 250]  best: 250
                    'selector__k': [100],  # [50,100,120,130,150,200, 300, 400, 600]  best: 100
                }},

    'XGBClassifier':
        {
            'name': 'XGBClassifier',
            'estimator': XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor'),
            'selector': SelectKBest(),
            'decomposition': PCA(),  # None because PCA() lowers the scores
            'params':
                {
                    "classifier__n_estimators": [600],  # [200, 500, 600, 800, 1000]  best: 600
                    "classifier__max_depth": [4],  # [2,4,8]  best: 4
                    "classifier__learning_rate": [0.01],  # [0.01, 0.1, 0.5]  best: 0.01
                    "classifier__subsample": [1],  # []  best: 1
                    "classifier__colsample_bytree": [1],  # [0.01, 0.1, 1]  best: 1
                    'selector__k': [30],  # [10, 20, 25, 30, 35, 40, 50, 90]  best: 30
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

kfold = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)


def get_best_classsifier(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Getting the best models for all passed estimators and saves it to joblib file.
    Using GridSearchCV for the getting the best params. Saving scores to csv file
    :param X_train: DataFrame of data for training
    :param X_val:  DataFrame of data for validation
    :param X_test: DataFrame of data for testing
    :param y_train: DataFrame of labels for training
    :param y_val: DataFrame of labels for validation
    :param y_test: DataFrame of labels for testing
    :return: DataFrame of obtained scores for each estimator, Dict of trained models
    """
    scores_index = ['best_params', 'best_score', 'roc_auc_score_train', 'roc_auc_score_val', 'roc_auc_score_test']
    scores = pd.DataFrame(index=scores_index)
    models = {}

    for key, value in classifiers.items():
        tmp_pipe = Pipeline([

            ('scaler', MinMaxScaler()),  #
            # ('sampling', RandomUnderSampler(random_state=40)),
            ('selector', value['selector']),  # Selecting k best features
            ('decomposition', value['decomposition']),  # using PCA for features decomposition
            ('classifier', value['estimator']),  # building estimation model
        ])
    
        # grid = GridSearchCV(tmp_pipe, value['params'], cv=kfold, n_jobs=-1) #fix if possible
        grid = GridSearchCV(tmp_pipe, value['params'], cv=kfold)
        grid.fit(X_train, y_train)
    
        roc_auc_score_train = roc_auc_score(y_train, grid.predict_proba(X_train)[:, 1])
        roc_auc_score_val = roc_auc_score(y_val, grid.predict_proba(X_val)[:, 1])
        roc_auc_score_test = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])
    
        # # Adding params and scores of a model to DataFrame for storage
        scores[key] = (grid.best_params_, grid.best_score_, roc_auc_score_train, roc_auc_score_val, roc_auc_score_test)
    
        # # Storing best estimator
        models[key] = grid.best_estimator_
    
        # # Saving model to file
        if key == 'LogisticRegression':
            estimator_name = 'LR'
        elif key == 'RandomForestClassifier':
            estimator_name = 'RndForest'
        elif key == 'XGBClassifier':
            estimator_name = 'XGB'
        else:
            estimator_name = key

        utils.save_as_joblib(grid.best_estimator_, estimator_name, MODEL_PATH)

        print(f'{key} has been processed')
        
    #
    # # Saving scores to file
    #
    if not os.path.isdir(SCORE_PATH):
        os.makedirs(f'{SCORE_PATH}')

    try:  # trying to open file if it exist and add scores to that file
        saved_scores = pd.read_csv(f'{SCORE_PATH}/scores.csv')
        saved_scores.index = scores_index
        scores = pd.concat([saved_scores, scores], axis=1)
        scores.index = scores_index
        scores.to_csv(f'{SCORE_PATH}/scores.csv', index=False)
    except FileNotFoundError:  # if there is no file with scores it is saving new one
        scores.to_csv(f'{SCORE_PATH}/scores.csv', index=True)
        
    return scores, models

    

