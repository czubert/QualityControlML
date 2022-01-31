# # #
# # # # Machine Learning Part
# # #
import os
import pandas as pd
from imblearn.pipeline import Pipeline

# # Saving models
from joblib import dump
# # Preprocessing
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
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
# from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier

# Output paths
# dir_path = 'data_output/step_5_ml'
# file_name = 'scores.csv'
MODEL_PATH = 'data_output/step_5_ml'
SCORE_PATH = 'data_output/step_5_ml/scores.csv'

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
                    "classifier__penalty": ['l1'],  # ['l1','l2', 'elasticnet', 'none']
                    "classifier__tol": [0.01],  # [0.00001,0.0001,0.001,0.01, 0.1]  best: 0.01
                    "classifier__C": [9],  # [0.01,0.01,1,10]  best:
                    "classifier__class_weight": ['balanced'],  # ['balanced', None]  best: 'balanced'
                    "classifier__solver": ['saga'],  # []  best: 'saga'
                    "classifier__max_iter": [500],  # [60, 90, 100, 150] best:
                    'selector__k': [600],  # [200, 400, 600]  best:
                    'decomposition__n_components': [100],  # [5,10,50,100,200]  best: 100
                }},
    
    # 'SVC':
    #     {
    #         'name': 'SVC',
    #         'estimator': SVC(),
    #         'selector': SelectKBest(),
    #         'decomposition': None,  # None because PCA() lowers the scores
    #         'params':
    #             {
    #                 "classifier__C": [0.01, 0.1, 1, 2],  # [0.01, 0.1, 1, 2]  best: 2
    #                 "classifier__kernel": ['rbf','poly','linear'],  # ['rbf','poly','linear']  best: 'rbf'
    #                 "classifier__degree": [1,3,5],  # [1,3,5]  best: 1
    #                 "classifier__max_iter": [-1, 300],  # [-1, 1, 10, 100, 1000]  best: -1
    #                 "classifier__gamma": ['auto'],  # []  best: 'auto'
    #                 "classifier__tol": [0.001, 0.01 ,1],  # [0.001, 0.01,0.1 ,1]  best: 0.1
    #                 "classifier__probability": [True],  # [True, False]  best: True
    #                 'selector__k': [90],  # [30, 50, 90, 100, 150, 200]  best: 90
    #             }},
    #
    # 'RandomForestClassifier':
    #     {
    #         'name': 'RandomForestClassifier',
    #         'estimator': RandomForestClassifier(),
    #         'selector': SelectKBest(),
    #         'decomposition': None,
    #         'params':
    #             {
    #                 'classifier__n_estimators': [100, 200],  # [list(range(100, 900, 200))]  best: 300
    #                 'classifier__criterion': ['gini'],  # ['gini']  best:'gini'
    #                 'classifier__max_features': [0.3, 0.5, 0.7],  # [0.3, 0.5, 0.7]  best: 0.5
    #                 'classifier__max_depth': [1,10,100],  # [1,10,100]  best: 100
    #                 'classifier__max_leaf_nodes': [50, 150, 250],  # [50, 150, 250]  best: 250
    #                 'classifier__min_samples_split': [0.1, 0.5, 0.09],  # [1, 5, 10]  best: 5
    #                 'classifier__bootstrap': [True],  # [True, False]  best: True
    #                 'classifier__max_samples': [50, 150, 250],  # [50, 150, 250]  best: 250
    #                 'selector__k': [50,400,600],  # [50,100,120,130,150,200]  best: 100
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
            # ('scaler', StandardScaler(with_mean=False)),
            # ('sampling', RandomUnderSampler(random_state=40)),
            ('selector', value['selector']),
            ('decomposition', value['decomposition']),
            ('classifier', value['estimator']),
        ])
    
        # WHAT: da się do grida wrzucić jakoś różne parametry dla metod z tmp_pipe?
        grid = GridSearchCV(tmp_pipe, value['params'], cv=kfold)
        grid.fit(X_train, y_train)
    
        roc_auc_score_train = roc_auc_score(y_train, grid.predict_proba(X_train)[:, 1])
        roc_auc_score_val = roc_auc_score(y_val, grid.predict_proba(X_val)[:, 1])
        roc_auc_score_test = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])
    
        # Adding params and scores of a model to DataFrame
        scores[key] = (grid.best_params_, grid.best_score_, roc_auc_score_train, roc_auc_score_val, roc_auc_score_test)
    
        # storing best model in the DataFrame
        models[key] = grid.best_estimator_
    
        #
        # # Saving models to files
        #
        if not os.path.isdir(f'{MODEL_PATH}'):
            os.makedirs(f'{MODEL_PATH}')
        dump(grid.best_estimator_, f'{MODEL_PATH}/{key}_model.joblib')
    
        print(f'{key} has been processed')
    #
    # # Saving scores to file
    #
    try:
        saved_scores = pd.read_csv(SCORE_PATH)
        saved_scores.index = scores_index
        scores = pd.concat([saved_scores, scores], axis=1)
        scores.index = scores_index
        scores.to_csv(SCORE_PATH, index=False)
    except FileNotFoundError:
        scores.index = scores_index
        scores.to_csv(SCORE_PATH, index=True)

    return scores, models

    

