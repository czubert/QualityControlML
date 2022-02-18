import numpy as np
import os

import pandas as pd

import utils


# Output paths
dir_path = 'data_output/step_3_rate_data'
file_name = 'rated_data'

# Constants
peaks = {
    # beginning of the peak and end of the peak to estimate max and min values
    'peak1': ['671', '761'],
    'peak2': ['801', '881'],
    'peak3': ['970', '1031'],
    'peak4': ['1530', '1641'],
}

quality_peak_names = ['Quality-peak1', 'Quality-peak2', 'Quality-peak3', 'Quality-peak4']

LP = 'laser_power'
IT = 'integration_time'
FN = 'File_name'


# TODO dodać opcje z korekcją baseline, która już nie będzie uwzględniała minimum, tylko będzie brała wartość maks piku
def main(grouped_files, baseline_corr=False):
    """
    Steps made to rate the spectra
    :param grouped_files:Each record holds a type of spectra as a key, and data in DataFrame as value
    :param baseline_corr:
    :return:
    """
    rated_grouped_files = grouped_files.copy()
    tmp_dict = {}  # for plots etc. to estimate the limit of the good / bad substrate
    
    for substr_type in ['ag', 'au']:
        # Changing col names type so there is a possibility to use pd.loc
        tmp_data_df = utils.change_col_names_type_to_str(rated_grouped_files[substr_type])
        
        # Creating a dataframe that consists of data relevant to evaluate the quality of SERS substrate
        rate_df = tmp_data_df.loc[:, ['id', 'laser_power', 'integration_time']]
        
        # take df with data, and gets peaks values which are then used to rate spectra
        rate_df = get_peak_value(tmp_data_df, rate_df)
        
        # for the purpose of checking where should be the limit set
        tmp_df = rate_df.copy()
        tmp_dict[substr_type] = tmp_df
        
        # rate spectra basing on peak values
        rate_df = rate_peaks(rate_df)
        
        rate_df['Quality'] = rate_df['Quality'].astype(int)
        
        # Adds the dataframe of relevant inforamtion about quality of SERS substrates to the dict of grouped spectra
        rated_grouped_files[substr_type + '_rated'] = rate_df
        
        # todo koniecznie przypisać oceny koncowe danego podłoża do widm TEŁ, na których jest nauka
        rated_grouped_files[substr_type + '_marks'] = pd.DataFrame(rate_df, columns=['id', 'Quality'])
        rated_grouped_files[substr_type + '_bg'] = rated_grouped_files[substr_type + '_bg'].join(
            rated_grouped_files[substr_type + '_marks'].set_index('id'), on='id')
    
    for key in rated_grouped_files.keys():
        rated_grouped_files[key].dropna(inplace=True, how='any', axis=0)
        try:
            rated_grouped_files[key]['Quality'] = rated_grouped_files[key]['Quality'].astype(np.uint8)
        except KeyError:
            continue
    
    utils.save_as_joblib(rated_grouped_files, file_name, dir_path)
    return rated_grouped_files, tmp_dict





def get_peak_value(tmp_data_df, new_df):
    for name, values in peaks.items():
        new_df.loc[:, name] = tmp_data_df.loc[:, values[0]:values[1]].max(axis=1) \
                              - tmp_data_df.loc[:, values[0]:values[1]].min(axis=1)
    return new_df


def rate_peaks(df):
    mask = df['id'].str.startswith('s')
    old_data_df = df.loc[mask]
    new_data_df = df.loc[~mask]
    
    # Rating the old data
    it = [4, 2, 1]
    lp = [20, 15, 10, 5, 3, 2, 1]
    
    # for integr_time in it:
    #     for power in lp:
    #         df.loc[((df['integration_time'] == integr_time) & (df['laser_power'] == power) & df['id'].str.startswith('s')), 'Quality'] = \
    #             rate_old_data_peaks(df.loc[mask], power, integr_time)
    # TODO zrobić coś takiego jak poniżej, nie używając do tego pośrednich DataFramów, pracować na wyjściowym df
    for integr_time in it:
        for power in lp:
            old_data_df.loc[((old_data_df['integration_time'] == integr_time) & (
                old_data_df['laser_power'] == power)), 'Quality'] = rate_old_data_peaks(old_data_df, power,
                                                                                        integr_time)
    
    # Rating the new data
    
    new_data_df['Quality'] = rate_new_data_peaks(new_data_df)
    
    # return df
    # return old_data_df
    return new_data_df


def rate_old_data_peaks(df, power, integr_time):
    # tmp_df = df[(df[LP] == power) & (df[IT] == integr_time)].copy()
    
    # TODO uwzględnić, że na niższe parametry schodziło się tylko jak widmo na wyższych wykraczało poza skalę, więc
    # można założyć, że widma przy 1% i 1/5/10 sekundach były bardzo dobre -
    # albo jeżeli dane id ma pomiar na 1s i 5% tzn, że wszystki widma o tym id powinny mieć max ocenę
    
    for peak_name in peaks.keys():
        quantiles = df[peak_name].quantile([0.25, 0.45, 0.50, 0.75])
        
        quality_name = f'Quality-{peak_name}'
        quality_peak_names.append(quality_name)
        
        df[quality_name] = np.NaN
        
        # TODO koniecznie zrobić wstępną ocenę, że wszystko co ma parametry niskie, to ma ocene max (1% 1s itp)
        # te co maja 10% 1s trzeba dobrze ocenić, inaczej dostosować te kwantyle, bo wystarczy, że jest średni pik,
        # to już nadaje się do sprzedaży, może po prostu 0.25, 0.75 i wszystko co powyzej pierwszych .25 to juz max
        
        # For Quality == 0
        df.loc[df.loc[:, peak_name] <= quantiles[0.45], quality_name] = 0
        
        # For Quality == 1
        df.loc[df.loc[:, peak_name] > quantiles[0.45], quality_name] = 1
        
        # For Quality == 2
        df.loc[df.loc[:, peak_name] >= quantiles[0.50], quality_name] = 2
        
        # For Quality == 3
        df.loc[df.loc[:, peak_name] > quantiles[0.75], quality_name] = 3
    
    # TODO max median mean wrzucic jako parametry funkcji, zeby latwo bylo to zmienic
    # important returns statistics of peak (max, median, mean)
    
    return df.loc[:, quality_peak_names].max(axis=1)


def rate_new_data_peaks(df):
    # todo przerobić ocenianie, wziąć pod uwagę, że w tym wypadku są zawsze takie same warunki,
    #  więc kwantyle mogą być spoko
    
    for peak_name in peaks.keys():
        quantiles = df[peak_name].quantile([0.25, 0.45, 0.50, 0.75])
        
        quality_name = f'Quality-{peak_name}'
        quality_peak_names.append(quality_name)
        
        df[quality_name] = np.NaN
        
        # TODO koniecznie zrobić wstępną ocenę, że wszystko co ma parametry niskie, to ma ocene max (1% 1s itp)
        # te co maja 10% 1s trzeba dobrze ocenić, inaczej dostosować te kwantyle, bo wystarczy, że jest średni pik,
        # to już nadaje się do sprzedaży, może po prostu 0.25, 0.75 i wszystko co powyzej pierwszych .25 to juz max
        
        # For Quality == 0
        df.loc[df.loc[:, peak_name] <= quantiles[0.25], quality_name] = 0
        
        # For Quality == 1
        df.loc[df.loc[:, peak_name] > quantiles[0.25], quality_name] = 1
        #
        # # For Quality == 2
        # df.loc[df.loc[:, peak_name] >= quantiles[0.50], quality_name] = 2
        #
        # # For Quality == 3
        # df.loc[df.loc[:, peak_name] > quantiles[0.75], quality_name] = 3
    
    # TODO max median mean wrzucic jako parametry funkcji, zeby latwo bylo to zmienic
    # important returns statistics of peak (max, median, mean)
    return round(df.loc[:, quality_peak_names].median(axis=1))


def rate_spectra(grouped_files, read_from_file=True, baseline_corr=False):
    if read_from_file:
        if not os.path.isfile(dir_path + '//' + file_name + '.joblib'):
            return main(grouped_files, baseline_corr=False)
        else:
            return utils.read_joblib(file_name, dir_path)
    
    else:
        return main(grouped_files, baseline_corr=False)


if __name__ == '__main__':
    dir_path = 'data_output/step_2_group_data'
    file_name = 'grouped_data'
    grouped_files = utils.read_joblib(file_name, '../' + dir_path)
    rated_spectra, tmp_dict = main(grouped_files)
