"""
********************************************************************************
Where the idea came from
--------------------------------------------------------------------------------
Application responsible for training estimators based on data available
Program will be used as the Quality Control of SERSitive substrates based on the spectra.
There are two types of data you can acquire using SERS substrate:
- The background of the active surface area
- SERS "fingerprint" of the measured compound, in our case it was 4-paramercaptobenzoic acid (PMBA),
We use PMBA to find out if our substrates have all the parameters (reproducibility, homogeneity and enhancement)
at expected rate. Our Quality Control is not the the most efficient one as:
- we need to use at least one substrate from each batch (15 pcs) - cost-prohibitive
- we measure the background spectra after production process and then we immerse it in PMBA for 20h - time-consuming
- we estimate the quality of the whole batch basing on 1-2 substrates that were immersed - low accuracy
We found out that there is a dependence between the background spectra and the quality of the substrates.
Therefore, the idea of this program was to check if the dependence is true and if we can estimate the quality
of the substrate from the background spectrum.
To rate spectra we have took 3 characteristic peaks which values never exceeds the detector of our Raman spectrometer,
to avoid the 'shifts' of raman spectra, we choose the region where we expect the peaks [first value is the beginning,
and the second one is the end of the region in which we expect the peak]:
    'peak1': ['671', '761'],
    'peak2': ['801', '881'],
    'peak3': ['970', '1031'],
Then we count the ratio between the highest and the lowest values in each regions (we also tried the absolut values,
and the original values, but both didn't work as expected, as ).
We sort ratios of all spectra (PMBA) that we have collected, and we checked spectrum by spectrum where is the boundary
between "good" and "bad" spectra. Next we have excluded 10% of spectra closest to the boundary, so it is more clear for
machine learning algorithms to see the difference in spectra while learning. All spectra that were beneath boundary
got the "0" rating, and all above "1".


********************************************************************************
"""


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
#TODO odpalić to dla DataFrame, a nie slownika poniżej w kodzie)

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
