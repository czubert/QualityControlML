import os
from sklearn.model_selection import train_test_split

import utils

# Output paths
dir_path_out = 'data_output/step_4_train_test_split_data'
file_name_out = 'train_test_split_data'

# Input paths
dir_path_in = 'data_output/step_3_rate_data'
file_name_in = 'rated_data'

# Constants
SUBSTRATE_TYPES = ['ag_bg']


def main(rated_spectra, seed):
    """
    Train test split method used on features and labels
    :param data_for_ml: dict
    :return: dict
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
