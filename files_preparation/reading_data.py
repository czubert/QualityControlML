import os
import re

import pandas as pd

import utils

# Output paths
dir_path = 'data_output/step_1_reading_data'
file_name = 'read_files'

DARK = 'Dark Subtracted #1'


def main(file_names):
    """
    Returns|saves dict of tuples(metadata, data) consisting of DataFrames of single spectra
    segregated by types of spectra (ag, au, ag_gb, au_bg
    Additionally it gives corresponding file name as a feature name.
    :return: dict of list of tuples
    """
    files = {}
    for file_names_types in file_names:
        data = []

        for name in file_names[file_names_types]:
            # Splits path name into two elements in a tuple. One is path and the second is name of a file

            if isinstance(name, str):
                path_split = os.path.split(name)

                # TODO koniecznie poprawić tutaj ten regex, jak nie tu, to przy przypisywaniu "id"
                prefix = re.search(r'20.{8}', path_split[0]).group(0)
                main_name = re.search(r'.*', path_split[1]).group(0)[:-4]  # getting filenames and deleting '.txt'
                # Getting rid of automatically saved spectra of the background
                if 'sp_0' in main_name:
                    continue

                tmp_file_name = prefix + ' ' + main_name
            else:
                continue

            # adding name of file into metadata DataFrame
            meta_df = read_metadata(name)
            meta_df.loc['file_name'] = name

            data_df = read_spectrum(name)

            # Solution for files in which there were no values in th ecolumn "Raman Shift"
            if data_df is None:
                continue

            # Distinguish between two types of spectra names
            if len(main_name) > 10:
                id_name = main_name.split('_')
                id_name[1].lstrip('0')

                id_name = '_'.join(id_name[0:2])
                data_df.loc['id'] = id_name

                data_df.rename(columns={DARK: main_name}, inplace=True)
            else:
                data_df.loc['id'] = tmp_file_name
                data_df.rename(columns={DARK: tmp_file_name}, inplace=True)

            # data_df.rename(columns={'Dark Subtracted #1': main_name}, inplace=True)

            # creates list of tuples containing 2 elements metadata and data
            data.append((meta_df, data_df))

        files[file_names_types] = data

    utils.save_as_joblib(files, file_name, dir_path)
    return files


def read_spectrum(filepath):
    """
    Reads numeric data from file and returns DataFrame
    :param filepath: String
    :return: DataFrame
    """
    print(filepath)
    read_params = {'sep': ';', 'skiprows': lambda x: x < 79 or x > 1500, 'decimal': ',',
                   'usecols': ['Raman Shift', DARK],
                   'skipinitialspace': True, 'encoding': "utf-8", 'na_filter': True}

    data_df = pd.read_csv(filepath, **read_params)

    data_df.dropna(axis=0, how="any")

    data_df = data_df[data_df.iloc[:, 0] > 253]

    data_df = data_df.astype({'Raman Shift': 'int'})

    # Solution for files in which there were no values in the column "Raman Shift"
    if data_df.empty:
        return None

    data_df.set_index('Raman Shift', inplace=True)
    return data_df


def read_metadata(filepath):
    """
    Reads metadata from file and returns DataFrame
    :param filepath: String
    :return: data frame
    """

    read_params = {'sep': ';', 'skiprows': lambda x: x > 78, 'decimal': ',', 'index_col': 0,
                   'skipinitialspace': True, 'encoding': "utf-8", 'header': None}
    
    meta_df = pd.read_csv(filepath, **read_params)

    return meta_df


def read_data(file_names, read_from_file=False):
    if read_from_file:
        if not os.path.isfile(dir_path + '//' + file_name + '.joblib'):
            read_files = main(file_names)
        else:
            read_files = utils.read_joblib(file_name, dir_path)
        return read_files
    else:
        return main(file_names)
