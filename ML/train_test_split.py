import os
from sklearn.model_selection import train_test_split

import utils

# Output paths
dir_path_out = 'data_output/step_4_train_test_split_data'
file_name_out = 'train_test_split_data'

# Input paths
dir_path_in = 'data_output/step_3_rate_data'
file_name_in = 'rated_data'


SUBSTRATE_TYPES = ['ag_bg']
# SUBSTRATE_TYPES = ['ag_bg', 'au_bg']  # jak dodamy złoto


def main(rated_spectra):
    """
    Train test split method used on features and labels
    :param data_for_ml: dict
    :return: dict
    """
    
    split_data = {}
    for substrate_type in SUBSTRATE_TYPES:
        rated_spectra[substrate_type].round()
        X = rated_spectra.iloc[:, 1:-4]
        y = rated_spectra.loc[:, 'y']

        #
        # # Train, Test, Validation Split
        #
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.1,
                                                            random_state=42,
                                                            stratify=y)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_size=0.1111,
                                                          random_state=42,
                                                          stratify=y_train)

        split_data[substrate_type] = X_train, X_test, y_train, y_test, X_val, y_val
        
        
    utils.save_as_joblib(split_data, file_name_out, dir_path_out)
    
    return split_data





def splitting_data(rated_spectra,read_from_file=True):
    if read_from_file:
        if not os.path.isfile(dir_path_out + '//' + file_name_out + '.joblib'):
            return main(rated_spectra)
        else:
            return utils.read_joblib(file_name_in, dir_path_in)
    
    else:
        return main(rated_spectra)

    