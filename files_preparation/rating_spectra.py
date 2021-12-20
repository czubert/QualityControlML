import numpy as np
import os

import utils

# Output paths
dir_path = 'data_output/step_3_rate_data'
file_name = 'rated_data'

# Constants
peaks = {
    # beginning of the peak and end of the peak to estimate max and min values
    'peak1': ['671.12', '761.54'],
    'peak2': ['801.91', '881.67'],
    'peak3': ['970.39', '1031.33'],
    'peak4': ['1530.47', '1641.35'],
}
LP = 'laser_power'
IT = 'integration_time'
FN = 'File_name'


# TODO dodać opcje z korekcją baseline, która już nie będzie uwzględniała minimum, tylko będzie brała wartość maks piku
def main(grouped_files, baseline_corr=False):
    for substr_type in ['ag', 'au']:
        tmp_data_df = change_col_names_type_to_str(grouped_files[substr_type])  # So there is a possibility to slice
        rate_df = tmp_data_df.loc[:, ['id', 'laser_power', 'integration_time']]
        rate_df = get_peak_value(tmp_data_df, rate_df)
        rate_df = rate_peaks(rate_df)
        
        grouped_files[substr_type + '_rated'] = rate_df
    
    utils.save_as_joblib(grouped_files, file_name, dir_path)
    return grouped_files


def change_col_names_type_to_str(df):
    df.copy()
    cols = [str(x) for x in df.columns]
    df.columns = cols
    return df


def get_peak_value(tmp_data_df, df):
    for name, values in peaks.items():
        df.loc[:, name] = tmp_data_df.loc[:, values[0]:values[1]].max(axis=1) \
                          - tmp_data_df.loc[:, values[0]:values[1]].min(axis=1)
    return df


def rate_peaks(df):
    it = [4, 2, 1]
    lp = [20, 15, 10, 5, 3, 2, 1]
    
    for integr_time in it:
        for power in lp:
            df.loc[((df['integration_time'] == integr_time) & (df['laser_power'] == power)), 'Quality'] = \
                rate_peak(df, power, integr_time)
    # df['Quality'] = df['Quality']
    return df


def rate_peak(df, power, integr_time):
    tmp_df = df[(df[LP] == power) & (df[IT] == integr_time)].copy()
    
    quality_peak_names = []
    
    # TODO tutaj zrobić słownik ID: Ocena
    
    for peak_name in peaks.keys():
        quantiles = tmp_df[peak_name].quantile([0.25, 0.45, 0.50, 0.75])
        
        quality_name = f'Quality-{peak_name}'
        quality_peak_names.append(quality_name)
        
        tmp_df[quality_name] = np.NaN
        
        # TODO koniecznie zrobić wstępną ocenę, że wszystko co ma parametry niskie, to ma ocene max (1% 1s itp)
        # te co maja 10% 1s trzeba dobrze ocenić, inaczej dostosować te kwantyle, bo wystarczy, że jest średni pik,
        # to już nadaje się do sprzedaży, może po prostu 0.25, 0.75 i wszystko co powyzej pierwszych .25 to juz max
        
        # For Quality == 0
        tmp_df.loc[tmp_df.loc[:, peak_name] <= quantiles[0.45], quality_name] = 0
        
        # For Quality == 1
        tmp_df.loc[tmp_df.loc[:, peak_name] > quantiles[0.45], quality_name] = 1
        
        # For Quality == 2
        tmp_df.loc[tmp_df.loc[:, peak_name] >= quantiles[0.50], quality_name] = 2
        
        # For Quality == 3
        tmp_df.loc[tmp_df.loc[:, peak_name] > quantiles[0.75], quality_name] = 3
    
    # TODO max median mean wrzucic jako parametry funkcji, zeby latwo bylo to zmienic
    # important returns statistics of peak (max, median, mean)
    return round(tmp_df.loc[:, quality_peak_names].max(axis=1))


def rate_spectra(grouped_files, read_from_file=True, baseline_corr=False):
    if read_from_file:
        if not os.path.isfile(dir_path + '//' + file_name + '.joblib'):
            return main(grouped_files, baseline_corr=False)
        else:
            return utils.read_joblib(file_name, dir_path)
    
    else:
        return main(grouped_files, baseline_corr=False)


if __name__ == '__main__':
    grouped_files = utils.read_joblib(file_name, '../' + dir_path)
    rated_spectra = main(grouped_files)
