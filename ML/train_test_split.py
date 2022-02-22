import os
from sklearn.model_selection import train_test_split

import utils

# Output paths
dir_path_out = 'data_output/step_4_train_test_split_data'
file_name_out = 'train_test_split_data'


def main(rated_spectra, seed):
    """
    Train test split method used on DataFrame of SERSitive silver substrate
    background features (X) + the rating of the spectrum (y)
    :param data_for_ml: DataFrame, that consists of all data and it's labels - "y"
    :param seed: int, seed number
    :return: dict, consisting of train, val, test values after train_test_split
    """
    
    X = rated_spectra.iloc[:, :-5]
    y = rated_spectra.loc[:, 'y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.1,
                                                        random_state=seed,
                                                        stratify=y)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.1111,
                                                      random_state=seed,
                                                      stratify=y_train)
    
    ml_variables = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
    }
    
    utils.save_as_joblib(ml_variables, file_name_out, dir_path_out)
    
    return ml_variables


def splitting_data(rated_spectra, read_from_file=True, seed=42):
    if read_from_file:
        if not os.path.isfile(dir_path_out + '//' + file_name_out + '.joblib'):
            return main(rated_spectra, seed)
        else:
            return utils.read_joblib(file_name_out, dir_path_out)
    
    else:
        return main(rated_spectra, seed)
