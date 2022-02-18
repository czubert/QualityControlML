import os
import pandas as pd
import utils


# constants
peaks = {
    # beginning of the peak and end of the peak to estimate max and min values
    'peak1': ['671', '761'],
    'peak2': ['801', '881'],
    'peak3': ['970', '1031'],
}
# Input path
dir_path = 'data_output/step_2_group_data'
file_name = 'grouped_data'
# Output path
output_dir_path = 'data_output/step_3_rate_data'
output_file_name = 'rated_data'


def main(grouped_files, only_new_spectra=True):
    # Getting relevant data
    ag_df = grouped_files['ag']  # Takes only ag spectra
    
    utils.change_col_names_type_to_str(ag_df)  # changes col names type from int to str, for .loc
    
    # if only_new_spectra == True, this part takes oly new spectra with names a1, a2 etc.
    if only_new_spectra:
        mask = ag_df['id'].str.startswith('s')  # mask to get only new spectra
        ag_df = ag_df[~mask]  # Takes only new spectra out of all ag spectra
    
    ratio_df = pd.DataFrame()  # DataFrame that will consists only of max/min ratio for each peak
    ratio_df['id'] = ag_df['id'].str.replace(r'_.*', '')
    
    # Getting ratio between max and mean value for each peak
    for name, values in peaks.items():
        ratio_df.loc[:, name] = ag_df.loc[:, values[0]:values[1]].max(axis=1) \
                                / ag_df.loc[:, values[0]:values[1]].min(axis=1)
    
    # Getting best ratio for each peak for each substrate
    best = ratio_df.groupby('id').max()
    
    chujowe = set(best.reset_index().sort_values('peak1')['id'].iloc[:90])
    spoko = set(best.reset_index().sort_values('peak1')['id'].iloc[130:])
    
    wszystkie = {*spoko, *chujowe}

    df = grouped_files['ag_bg']

    mask = df['id'].str[:-2].isin(wszystkie)
    df_wybrane = df[mask]
    
    df_wybrane['y'] = df['id'].str[:-2].isin(spoko).astype(int)
    
    utils.save_as_joblib(df_wybrane, output_file_name, output_dir_path)
    
    return df_wybrane




def rate_spectra(grouped_files, read_from_file=True, only_new_spectra=True,  baseline_corr=False):
    if read_from_file:
        if not os.path.isfile(output_dir_path + '//' + output_file_name + '.joblib'):
            return main(grouped_files, only_new_spectra)
        else:
            return utils.read_joblib(output_file_name, output_dir_path)
    
    else:
        return main(grouped_files, only_new_spectra)



if __name__ == "__main__":
    grouped_files = utils.read_joblib(file_name, '../' + dir_path)
    
    df_wybrane = main(grouped_files)
    
    print('done')
