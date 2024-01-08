import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, ensemble
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# reading data

with open('../DataFrame/df_processed_data.pkl', 'rb+') as file:

    data = pickle.load(file)

# splitting data into training and testing

training_indices = data['substrate number'].drop_duplicates().sample(frac=0.8).values

training_set = data[data['substrate number'].isin(training_indices)]

test_set = data[~data['substrate number'].isin(training_indices)]

# ML script

X_training = training_set.loc[:, 'surface':'peaks number']
y_training = training_set.loc[:, ['ln(ef)']]

X_test = test_set.loc[:, 'surface':'peaks number']
y_test = test_set.loc[:, ['ln(ef)']]

#regr = linear_model.LinearRegression()

#regr = RandomForestRegressor()

#regr = svm.SVR()

#regr = ensemble.GradientBoostingRegressor()

#regr.fit(X_training, np.array(y_training).ravel())

#y_predicted = regr.predict(X_test)

param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a Random Forest Regressor
rf_model = RandomForestRegressor()

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Fit the model to the training data
grid_search.fit(X_training, np.array(y_training).ravel())


print('Score: {}'.format(grid_search.score(X_test, y_test)))

#print('Score: {}'.format(regr.score(X_test, y_test)))
# The mean squared error
#print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))

