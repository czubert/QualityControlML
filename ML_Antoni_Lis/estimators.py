import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, ensemble
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score

# reading data

with open('../DataFrame/df_processed_data.pkl', 'rb+') as file:

    data = pickle.load(file)

# splitting data into training and testing

training_indices = data['substrate number'].drop_duplicates().sample(frac=0.8).values

training_set = data[data['substrate number'].isin(training_indices)]

test_set = data[~data['substrate number'].isin(training_indices)]

# ML script

X_training = training_set.loc[:, 'surface':'w10']
y_training = training_set.loc[:, ['ln(ef)']]

X_test = test_set.loc[:, 'surface':'w10']
y_test = test_set.loc[:, ['ln(ef)']]

#regr = linear_model.LinearRegression()

#regr = RandomForestRegressor()

#regr = svm.SVR() # here kernel matters

regr = ensemble.GradientBoostingRegressor()

regr.fit(X_training, np.array(y_training).ravel())

y_predicted = regr.predict(X_test)



print('Score: {}'.format(regr.score(X_test, y_test)))



