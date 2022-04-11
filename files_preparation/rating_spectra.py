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


def main(grouped_files, border_value, only_new_spectra=True):
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
    
    # TODO, czy sklejanie 2 widm na jednym podłożu ma sens? nie lepiej traktować to jako dwa różne wyniki?
    # Getting best ratio for each peak for each substrate
    best = ratio_df.groupby('id').max()
    
    """
    Selecting spectra, based on the max/min ratio, that are of high or low quality,
    also making a gap between high and low bigger, so the estimator has higher chances
    to see the differences between low and high spectra.
    """
    # Making the difference between good and low (1/0) spectra more significant
    low_value = int(border_value - (border_value * 0.2))
    high_value = int(border_value + (border_value * 0.2))
    
    low = set(best.reset_index().sort_values('peak1')['id'].iloc[:low_value])
    high = set(best.reset_index().sort_values('peak1')['id'].iloc[high_value:])
    
    # Creating a tuple of high and low spectra without the spectra that are close to the middle
    all_spectra = {*high, *low}
    
    # Getting all background spectra of silver substrates
    df = grouped_files['ag_bg']
    
    # Getting only selected (high, low) spectra out of all spectra, based on max/min ratio of the peaks
    mask = df['id'].str[:-2].isin(all_spectra)
    df_wybrane = df[mask]
    
    # Giving a  1/0 label to a spectrum, depending if a spectrum is in a list of high or low of max/min ratio
    df_wybrane['y'] = df['id'].str[:-2].isin(high).astype(int)
    
    # Saving results to a joblib file
    utils.save_as_joblib(df_wybrane, output_file_name, output_dir_path)
    
    return df_wybrane


def rate_spectra(grouped_files, border_value, read_from_file=True, only_new_spectra=True):
    if read_from_file:
        if not os.path.isfile(output_dir_path + '//' + output_file_name + '.joblib'):
            return main(grouped_files, border_value, only_new_spectra)
        else:
            return utils.read_joblib(output_file_name, output_dir_path)
    
    else:
        return main(grouped_files, border_value, only_new_spectra)


if __name__ == "__main__":
    grouped_files = utils.read_joblib(file_name, '../' + dir_path)
    
    df_wybrane = main(grouped_files, border_value=130)
    
    print('done')
