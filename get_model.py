"""
********************************************************************************
Where the idea came from
--------------------------------------------------------------------------------
Application responsible for training estimators based on data available.
Program will be used for the Quality Control of SERSitive substrates based on SERS spectra.
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

------------------------------------
Preprocessing
------------------------------------
To rate spectra we have took 3 characteristic peaks which values never exceeds the detector of our Raman spectrometer,
to avoid the 'shifts' of raman spectra, we choose the region where we expect the peaks [first value is the beginning,
and the second one is the end of the region in which we expect the peak]:
    'peak1': ['671', '761'],
    'peak2': ['801', '881'],
    'peak3': ['970', '1031'],
Then we count the ratio between the highest and the lowest values in each regions (we also tried the absolute values,
and the original values, but both didn't work as expected).
We sorted the ratios of all PMBA spectra, that we have collected, in ascending order.
Then we checked plots spectrum by spectrum where is the boundary between "good" and "bad" spectra.
Next we have excluded 10% of spectra closest to the boundary (beneath and above), so it is more clear for
machine learning algorithms to see the difference between "good" and "bad" spectra during the learning process.
All spectra that were beneath boundary got the "0", and all above "1" rating.
The last step of preprecessing the data was to assign the labels of the PMBA spectra to the background spectra
by the IDs of the substrate.

------------------------------------
Train Test Split
------------------------------------
First step was splitting data into training, validation and test sets of data.

Training data set is used by the ML algorithm to learn what kind of features (in this case background spectra)
results in which label (0 or 1). This is the place where the magic of Machine Learning appears.
Depending on the chosen estimator there are some differences in the way of achieving the best fitted function,
but the gol is the same - achieving the best score of predicting the class.

Validation data set is used to check the performance of the trained model, as you should not use the training data
for this purpose, because algorithm already "know" this features and class connected with it.
Based only on the features of validation data set (that program should don't recognize) model is trying to predict
the class of the substrate (0 or 1) and after that the predicted and true class values are compared and the
score is calculated. When we are satisfied with the score on validation dataset, we are trying to predict the class
basing on the test dataset, to be sure that the score was valid and not a lucky mistake.
Test dataset should not be "known" by the estimator

------------------------------------
Preprocessing and Machine Learning
------------------------------------
To find the best performance of the Quality Control program, we have combined GridSearchCV and Pipeline modules.
Pipeline allows you to create a queue for performing actions on data as part of preprocessing.
GridSearchCV is using cross-validation, to check the performance of all given variables for all specified parameters.
By using the ability to combine this two modules, we could find the best performing ML model of all.
Our Pipeline consists of 5 steps to find the best combination of preprocessing the features
and learning on different estimator parameters (for each step there is a control probe,
where we don't do any operation on the features and it is described as None.
Pipeline:
 - scaler (StandardScaler, MinMaxScaler, None) - scaling features to see if reducing numbers to one range
   will improve the performance of the model or not
 - sampling (RandomUnderSampler, RandomOverSampler, None) - in case of imbalanced data,
   checking if under or over sampling will have an impact on model performance
 - selector (SelectKBest, None) - selecting specified number (K) of the best features,
   to see the impact on the model performance compared to using all features (None)
 - decomposition (PCA, None) - The principal component analysis looks for relationships between
   features based on variance and groups them on this basis, making the differences more significant.
   As a result, the estimator may have a better chance of correctly predicting the results
 - classifier (LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier,
   XGBClassifier, CatBoostClassifier)


------------------------------------
Parameters of the estimators
------------------------------------
    Logistic Regression
            Name: 'LogisticRegression',
            Estimator: LogisticRegression(),
            Parameters:
                    Classifier, selector, decomposition         Tested parameters and best scores:
                    best params:
                    
                    "classifier__penalty": ['l1'],              # ['l1','l2', 'elasticnet', 'none'] best: l1
                    "classifier__tol": [0.01],                  # [0.00001,0.0001,0.001,0.01, 0.1]  best: 0.01
                    "classifier__C": [9],                       # [0.01,0.01,1,10]  best:
                    "classifier__class_weight": ['balanced'],   # ['balanced', None]  best: 'balanced'
                    "classifier__solver": ['saga'],             # []  best: 'saga'
                    "classifier__max_iter": [500],              # [60, 90, 100, 150] best:
                    'selector__k': [600],                       # [200, 400, 600]  best:
                    'decomposition__n_components': [100],       # [5,10,50,100,200]  best:

    Support Vector Classification
            Name: 'SVC',
            Estimator: SVC(),
            Parameters:
                    Classifier, selector, decomposition         Tested parameters and best scores:
                    best params:
                    
                    "classifier__C": [0.01, 0.1, 1, 2,4],       # [0.01, 0.1, 1, 2]  best: 2
                    "classifier__kernel": ['rbf'],              # ['rbf','poly','linear']  best: 'rbf'
                    "classifier__degree": [1,2],                # [1,3,5]  best: 1
                    "classifier__max_iter": [-1],               # [-1, 1, 10, 100, 1000]  best: -1
                    "classifier__gamma": ['auto'],              # []  best: 'auto'
                    "classifier__tol": [0.001],                 # [0.001, 0.01,0.1 ,1]  best: 0.1
                    "classifier__probability": [True],          # [True, False]  best: True
                    'selector__k': [30],                        # [30, 50, 90, 100, 150, 200]  best: 90

    Random Forest Classifier
            Name: 'RandomForestClassifier',
            Estimator: RandomForestClassifier(),
            Parameters:
                    Classifier, selector, decomposition         Tested parameters and best scores:
                    best params:
                    
                    'classifier__n_estimators': [300],          # [list(range(100, 900, 200))]  best: 300
                    'classifier__criterion': ['gini'],          # ['gini']  best:'gini'
                    'classifier__max_features': [0.5],          # [0.3, 0.5, 0.7]  best: 0.5
                    'classifier__max_depth': [100],             # [1,10,100]  best: 100
                    'classifier__max_leaf_nodes': [250],        # [50, 150, 250]  best: 250
                    'classifier__min_samples_split': [0.1, 0.5, 0.09],  # [1, 5, 10]  best: 5
                    'classifier__bootstrap': [True],            # [True, False]  best: True
                    'classifier__max_samples': [250],           # [50, 150, 250]  best: 250
                    'selector__k': [100],                       # [50,100,120,130,150,200, 300, 400, 600]  best: 100
    
    XGB Classifier
            Name: 'XGBClassifier',
            Estimator: XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor'),
            Parameters:
                    Classifier, selector, decomposition         Tested parameters and best scores:
                    best params:
                    
                    "classifier__n_estimators": [600],          # [200, 500, 600, 800, 1000]  best: 600
                    "classifier__max_depth": [4],               # [2,4,8]  best: 4
                    "classifier__learning_rate": [0.01],        # [0.01, 0.1, 0.5]  best: 0.01
                    "classifier__subsample": [1],               # []  best: 1
                    "classifier__colsample_bytree": [1],        # [0.01, 0.1, 1]  best: 1
                    'selector__k': [30],                        # [10, 20, 25, 30, 35, 40, 50, 90]  best: 30

    CatBoost Classifier
            Name: 'CatBoostClassifier',
            Estimator: CatBoostClassifier(task_type="GPU", devices='0:2'),
            Parameters:
                    Classifier, selector, decomposition         Tested parameters and best scores:
                    best params:
                    
                    "classifier__n_estimators": [600],          # [200, 500, 600, 800, 1000]  best: 600
                    "classifier__max_depth": [4],               # [2,4,8]  best: 4
                    "classifier__learning_rate": [0.01],        # [0.01, 0.1, 0.5]  best: 0.01
                    'selector__k': [30],                        # [10, 20, 25, 30, 35, 40, 50, 90]  best: 30



------------------------------------
Streamlit - GUI
------------------------------------

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
