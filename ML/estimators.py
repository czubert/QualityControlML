# # #
# # # # Machine Learning Part
# # #
import os
import pandas as pd
import sklearn.metrics
from imblearn.pipeline import Pipeline

# # Preprocessing
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler

# # Metrics
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import utils
from . import ef_classifiers

# Output paths
MODEL_PATH = 'data_output/step_5_ml/models'
SCORE_PATH = 'data_output/step_5_ml/scores'

SEED = 123

# The params set below are the params for which I have  received the best ROC-AUC score
# classifiers = ef_classifiers.not_evaluated_classifiers
classifiers = ef_classifiers.evaluated_classifiers

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
    scores_index = ['best_params', 'best_score', 'roc_auc_score_train', 'roc_auc_score_val',
                    'roc_auc_score_test',
                    'precision_train', 'precision_test'
                    ]

    scoring_for_gs_cv = ['accuracy', 'f1', 'balanced_accuracy']
    scores = pd.DataFrame(index=scores_index)
    models = {}

    for estimator, est_details in classifiers.items():
        tmp_pipe = Pipeline([

            ('scaler', MinMaxScaler()),  #
            # ('sampling', RandomUnderSampler(random_state=40)),
            ('selector', est_details['selector']),  # Selecting k best features
            ('decomposition', est_details['decomposition']),  # using PCA for features decomposition
            ('classifier', est_details['estimator']),  # building estimation model
        ])

        # grid = GridSearchCV(tmp_pipe, est_details['params'], cv=kfold, n_jobs=-1) #fix if possible, to make it work
        grid = GridSearchCV(tmp_pipe, est_details['params'], cv=kfold, scoring=scoring_for_gs_cv[2], refit='f1')
        grid.fit(X_train, y_train)

        roc_auc_score_train = roc_auc_score(y_train, grid.predict_proba(X_train)[:, 1])
        roc_auc_score_val = roc_auc_score(y_val, grid.predict_proba(X_val)[:, 1])
        roc_auc_score_test = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])

        precision_train = precision_score(y_train, grid.predict(X_train))
        precision_test = precision_score(y_test, grid.predict(X_test))

        # # Adding params and scores of a model to DataFrame for storage
        scores[estimator] = (grid.best_params_, grid.best_score_, roc_auc_score_train, roc_auc_score_val,
                             roc_auc_score_test, precision_train, precision_test)

        # # Storing best estimator
        models[estimator] = grid.best_estimator_

        # # Saving model to file - names so it is easier to read in a program
        if estimator == 'LogisticRegression':
            estimator_name = 'LR'
        elif estimator == 'RandomForestClassifier':
            estimator_name = 'RndForest'
        elif estimator == 'XGBClassifier':
            estimator_name = 'XGB'
        else:
            estimator_name = estimator

        utils.save_as_joblib(grid.best_estimator_, estimator_name, MODEL_PATH)

        print(f'{estimator} has been processed')

    #
    # # Saving scores to file
    #
    if not os.path.isdir(SCORE_PATH):
        os.makedirs(f'{SCORE_PATH}')

    try:  # trying to open file if it exists and add scores to that file
        saved_scores = pd.read_csv(f'{SCORE_PATH}/scores.csv')
        saved_scores.index = scores_index
        scores = pd.concat([saved_scores, scores], axis=1)
        scores.index = scores_index
        scores.to_csv(f'{SCORE_PATH}/scores.csv', index=False)
    except FileNotFoundError:  # if there is no file with scores it is saving new one
        scores.to_csv(f'{SCORE_PATH}/scores.csv', index=True)

    return scores, models
