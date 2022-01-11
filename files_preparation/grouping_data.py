import os
import pandas as pd
import utils

# Output paths
dir_path = 'data_output/step_2_group_data'
file_name = 'grouped_data'


def main(separated_names):
    """
    Returns dict consisting of one DataFrames per spectra type, other consists of mean values per data type.
    :param separated_names: dict
    :return: dict
    """
    # groups Dark Subtracted column from all dfs to one and overwrites data df in dictionary
    dict_of_grouped_dfs = {}
    
    for substrate_type in separated_names.keys():
        data_list = []
        for metadata_data in separated_names[substrate_type]:
            tmp_meta = metadata_data[0]
            tmp_data = metadata_data[1]
            tmp_data.loc['laser_power'] = tmp_meta.loc['laser_powerlevel'][1]
            tmp_data.loc['integration_time'] = int(tmp_meta.loc['intigration times(ms)'][1]) * \
                                               int(tmp_meta.loc['time_multiply'][1]) / 1000
            data_list.append(tmp_data)
        
        grouped_df = pd.concat(data_list, axis=1, sort=False).T
        # grouped_df.dropna(how="all",axis=1, inplace=True)
        
        #TODO sprawdzić czy to ma wpływ na szybkość działania i wyniki, przerzucić do preprocessingu potem
        #WHAT na szczescie uint16 ma zasieg do 65535,
        # czyli odrobinę powyżej max z instrumentu pomiarowego
        # grouped_df = grouped_df.astype(np.uint16)
        #meta_df.loc['file_name'] = name
        dict_of_grouped_dfs[substrate_type] = grouped_df
    
    utils.save_as_joblib(dict_of_grouped_dfs, file_name, dir_path)
    return dict_of_grouped_dfs


def group_data(read_files, read_from_file=True):
    if read_from_file:
        if not os.path.isfile(dir_path + '//' + file_name + '.joblib'):
            grouped_files = main(read_files)
        else:
            grouped_files = utils.read_joblib(file_name, dir_path)
        return grouped_files
    else:
        return main(read_files)


if __name__ == '__main__':
    dir_path = 'data_output/step_1_reading_data'
    file_name = 'read_files'
    read_files = utils.read_joblib(file_name, '../' + dir_path)
    grouped_spectra = main(read_files)
