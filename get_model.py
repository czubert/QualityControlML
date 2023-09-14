import time
from datetime import datetime

from files_preparation import getting_names, reading_data, grouping_data, data_analysis
from ML import rating_spectra #train_test_split, estimators

# read_from_files = True
read_from_files = False
now = datetime.now()

# PARAMS
peak = {'name': 'peak1', 'border_value': 55 * 1000000, 'margin_of_error': 0.10}
peak2 = {'name': 'peak2', 'border_value': 55 * 1000000, 'margin_of_error': 0.10}
peak3 = {'name': 'peak3', 'border_value': 55 * 1000000, 'margin_of_error': 0.10}
peak4 = {'name': 'peak4', 'border_value': 55 * 1000000, 'margin_of_error': 0.10}
peak5 = {'name': 'peak5', 'border_value': 55 * 1000000, 'margin_of_error': 0.10}


peaks = [peak, peak2, peak3, peak4, peak5]


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

file_names = getting_names.get_names(read_from_file=read_from_files)

#print(file_names)

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
read_files = reading_data.read_data(file_names, read_from_file=read_from_files)
raman_pmba = reading_data.read_spectrum('data_input/PMBA__powder_10%_1s.txt')

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

grouped_files = grouping_data.group_data(read_files, read_from_file=read_from_files)

print(f'Data loaded in {round(time.time() - start_time, 2)} seconds')
print()

"""
********************************************************************************
Preparing images for data analysis
--------------------------------------------------------------------------------

********************************************************************************
"""
# TODO probably this part should be removed
# start_time = time.time()
# print('Preparing images for data analysis...')
#
# # data_analysis.run(grouped_files, best_ratio=True, peak=peak['name'])
# data_analysis.run(grouped_files, best_ratio=True, peak='peak1')
#
# print(f'Spectra saved in {round(time.time() - start_time, 2)} seconds')
# print()

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

# PARAMS here you can change params which have impact on spectra rating
rated_spectra = rating_spectra.rate_spectra(
    grouped_files,
    raman_pmba,
    chosen_peak=peaks, #[0]['name'],  #which peak should be taken (peak1, peak2, peak3)
    border_value=peak['border_value'],  # value of the border between good/bad spectra
    margin_of_error=peak['margin_of_error'],  # % of spectra that should increase above border
    only_new_spectra=False)  # if you want to work on ML spectra only

print(f'Data loaded in {round(time.time() - start_time, 2)} seconds')
print()

"""
********************************************************************************
Train Test Split
--------------------------------------------------------------------------------
Train Test Split background data
********************************************************************************
"""

# start_time = time.time()
# print('Train Test Splitting the data...')
# ml_variables = train_test_split.splitting_data(rated_spectra, read_from_file=False, seed=42)
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
# start_time = time.time()
# print('Looking for best estimator... be patient...')
#
# scores, models = estimators.get_best_classsifier(**ml_variables)
#
# time_s = round(time.time() - start_time, 2)
# time_min = round(time_s / 60)
# time_h = round(time_min / 60)
# print(f'Models trained in {time_s} seconds')
# print(f'Models trained in {time_min} minutes')
# print(f'Models trained in {time_h} hours')
# print()
