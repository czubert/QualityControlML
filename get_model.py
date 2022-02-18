"""
Application responsible for training estimators based on data available
"""
import time
from datetime import datetime

import pandas as pd

from files_preparation import getting_names, reading_data, grouping_data, rating_spectra, rating_spectra_chasz, \
    data_analysis
from ML import train_test_split, estimators

now = datetime.now()

print(f'Starting time: {now.strftime("%Y_%m_%d %H:%M:%S")}')
print()

# Loading data, separating it by ag/au/tlo ag/tlo au and separating it to metadata and data
start_time = time.time()

"""
********************************************************************************
GETTING FILE NAMES FOR FURTHER USE
--------------------------------------------------------------------------------
********************************************************************************
"""

print('Getting filenames...')

file_names = getting_names.get_names(read_from_file=True)

print(f'Data loaded in {round(time.time() - start_time, 2)} seconds')
print()

"""
********************************************************************************
Reading files (based on files_names) into DataFrames
--------------------------------------------------------------------------------
stored as Dictionary, where
keys are ag, au, ag_bg, au_bg
values are tuples(metadata, data)
********************************************************************************
"""

start_time = time.time()
print('Reading files into DataFrames...')

# Getting files as a dict of tuples with metadata and data, keys are types of spectra (ag, au, ag_bg, au_bg
read_files = reading_data.read_data(file_names, read_from_file=True)

print(f'Data loaded in {round(time.time() - start_time, 2)} seconds')
print()

"""
********************************************************************************
Grouping Data
--------------------------------------------------------------------------------
Creating a DataFrame concatenated from all single spectra DataFrames
from each type of spectra (ag, au, ag_bg, au_bg)
One DataFrame per one Spectra Type
********************************************************************************
"""

start_time = time.time()
print('Grouping data...')
grouped_files = grouping_data.group_data(read_files, read_from_file=True)

print(f'Data loaded in {round(time.time() - start_time, 2)} seconds')
print()

"""
********************************************************************************
Rating Peaks
--------------------------------------------------------------------------------
Rating each spectra. Creating DataFrame with all relevant data for rating.
Creating a DataFrame with only 'id' and 'Quality' features
Adding 'Quality' feature to background spectra based on 'id'
********************************************************************************
"""

start_time = time.time()
print('Rating spectra...')
# rated_spectra = rating_spectra.rate_spectra(grouped_files, read_from_file=True, baseline_corr=False)
rated_spectra = rating_spectra_chasz.rate_spectra(grouped_files, read_from_file=True,
                                                  only_new_spectra=True, baseline_corr=False)
# data_analysis.run(rated_spectra)

print(f'Data loaded in {round(time.time() - start_time, 2)} seconds')
print()

"""
********************************************************************************
Train Test Split
--------------------------------------------------------------------------------
Train Test Split background data
********************************************************************************
"""
#
# start_time = time.time()
# print('Train Test Splitting the data...')
# train_test_data = train_test_split.splitting_data(rated_spectra, read_from_file=True)
#
# print(f'Data loaded in {round(time.time() - start_time, 2)} seconds')
# print()

"""
********************************************************************************
Looking for best estimator
--------------------------------------------------------------------------------
Checking many estimators, with different parameters
********************************************************************************
"""
start_time = time.time()
print('Looking for best estimator... be patient...')

# for key in train_test_data.keys():
#     print(f'Getting best model for {key}')
X = rated_spectra.iloc[:, :-5]
y = rated_spectra.loc[:, 'y']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=42,
                                                    stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.1111,
                                                  random_state=42,
                                                  stratify=y_train)

ml_variables = {
    'X_train': X_train,
    'X_val': X_val,
    'X_test': X_test,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,
}

scores, models = estimators.get_best_classsifier(**ml_variables)

print(f'Data loaded in {round(time.time() - start_time, 2)} seconds')
print()
