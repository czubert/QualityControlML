import time
from datetime import datetime

from files_preparation import getting_names, reading_data, grouping_data, rating_spectra, data_analysis
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
print(read_files)

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
# important while uploading new data remember to look at the plots to find the limit where substrates are good and bad
# TODO napisać moduł do podglądu danych, żeby wybrać granice
# TODO dodać parametry graniczne dla widm słabych i dobrych i przerwy
start_time = time.time()
print('Rating spectra...')
# rated_spectra = rating_spectra.rate_spectra(grouped_files, read_from_file=True, baseline_corr=False)
rated_spectra = rating_spectra.rate_spectra(grouped_files, read_from_file=True,
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

start_time = time.time()
print('Train Test Splitting the data...')
ml_variables = train_test_split.splitting_data(rated_spectra, read_from_file=True, seed=42)

print(f'Data loaded in {round(time.time() - start_time, 2)} seconds')
print()

"""
********************************************************************************
Looking for best estimator
--------------------------------------------------------------------------------
Checking many estimators, with different parameters
********************************************************************************
"""
start_time = time.time()
print('Looking for best estimator... be patient...')

scores, models = estimators.get_best_classsifier(**ml_variables)

print(f'Models trained in {round(time.time() - start_time, 2)} seconds')
print()
