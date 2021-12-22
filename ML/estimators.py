# # #
# # # # Machine Learning Part
# # #

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
SCORE_PATH = 'data_output/step_5_ml/scores.csv'

SEED = 123

# The params set below are the params for which I have  received the best ROC-AUC score
classifiers = {
    'LogisticRegression':
        {
            'name': 'LogisticRegression',
            'estimator': LogisticRegression(),
            # 'selector': SelectKBest(),
            # 'decomposition': PCA(),
            'params':
                {
                    "classifier__penalty": ['l1'],  # ['l1','l2', 'elasticnet', 'none]  best: 'l1'
                    "classifier__tol": [0.0001],  # [0.00001,0.0001,0.001,0.01, 0.1]  best: 0.0001
                    "classifier__C": [1],  # [0.01,0.01,1,10]  best: 1
                    "classifier__class_weight": ['balanced'],  # ['balanced', None]  best: 'balanced'
                    "classifier__solver": ['saga'],  # []  best: 'saga'
                    "classifier__max_iter": [90],  # [60, 90, 100, 150] best: 90
                }},

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
    value = {'params':
    {
        "penalty": 'l1',  # ['l1','l2', 'elasticnet', 'none]  best: 'l1'
        "tol": 0.0001,  # [0.00001,0.0001,0.001,0.01, 0.1]  best: 0.0001
        "C": 1,  # [0.01,0.01,1,10]  best: 1
        "class_weight": 'balanced',  # ['balanced', None]  best: 'balanced'
        "solver": 'saga',  # []  best: 'saga'
        "max_iter": 90
    }}
    
    # todo sprawdzić jaki scoring tu stosować
    # grid = GridSearchCV(LogisticRegression(), value['params'], cv=kfold)
    model = LogisticRegression(**value['params'])
    
    model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    score = model.score(X_test, y_test)
    
    print(score)
    #
    # roc_auc_score_train = roc_auc_score(y_train, grid.predict_proba(X_train)[:, 1])
    # roc_auc_score_val = roc_auc_score(y_val, grid.predict_proba(X_val)[:, 1])
    # roc_auc_score_test = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])
    #
    # # Adding params and scores of a model to DataFrame
    # scores['ag'] = (grid.best_params_, grid.best_score_, roc_auc_score_train, roc_auc_score_val, roc_auc_score_test)
    #
    # # storing best model in the DataFrame
    # models['agaga'] = grid.best_estimator_
    #
    # #
    # # # Saving models to files
    # #
    # dump(grid.best_estimator_, f'models/LR_model.joblib')
    #
    # #
    # # # Saving scores to file
    # #
    # try:
    #     saved_scores = pd.read_csv(SCORE_PATH)
    #     saved_scores.index = scores_index
    #     scores = pd.concat([saved_scores, scores], axis=1)
    #     scores.index = scores_index
    #     scores.to_csv(SCORE_PATH, index=False)
    # except FileNotFoundError:
    #     scores.index = scores_index
    #     scores.to_csv(SCORE_PATH, index=True)
        
        
    #
    # for key, value in classifiers.items():
    #     tmp_pipe = Pipeline([
    #         ('classifier', value['estimator']),
    #     ])
    #
    #     # todo sprawdzić jaki scoring tu stosować
    #     grid = GridSearchCV(tmp_pipe, value['params'], cv=kfold, scoring='roc_auc')
    #     grid.fit(X_train, y_train)
    #
    #     roc_auc_score_train = roc_auc_score(y_train, grid.predict_proba(X_train)[:, 1])
    #     roc_auc_score_val = roc_auc_score(y_val, grid.predict_proba(X_val)[:, 1])
    #     roc_auc_score_test = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])
    #
    #     # Adding params and scores of a model to DataFrame
    #     scores[key] = (grid.best_params_, grid.best_score_, roc_auc_score_train, roc_auc_score_val, roc_auc_score_test)
    #
    #     # storing best model in the DataFrame
    #     models[key] = grid.best_estimator_
    #
    #     #
    #     # # Saving models to files
    #     #
    #     dump(grid.best_estimator_, f'models/{key}_model.joblib')
    #
    #     print(f'{key} has been processed')
    # #
    # # # Saving scores to file
    # #
    # try:
    #     saved_scores = pd.read_csv(SCORE_PATH)
    #     saved_scores.index = scores_index
    #     scores = pd.concat([saved_scores, scores], axis=1)
    #     scores.index = scores_index
    #     scores.to_csv(SCORE_PATH, index=False)
    # except FileNotFoundError:
    #     scores.index = scores_index
    #     scores.to_csv(SCORE_PATH, index=True)
    
    return scores, models
