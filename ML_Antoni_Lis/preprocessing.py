import pandas as pd
from scipy import signal
import numpy as np
import pickle
import utils


# TODO
class Preprocessing:

    def __init__(self, path):
        self.df_path = path
        self.raw_data = self.get_raw_data()
        self.spectra_end = self.where_spectra_ends()
        self.processed_data = self.analyze_data()

    @staticmethod
    def move_columns(daf):
        col = list(daf.columns)

        moved_columns = ['id', 'laser_power', 'integration_time']

        col = [item for item in col if item not in moved_columns]

        col = col + moved_columns

        dat = daf[col]

        return dat

    def get_raw_data(self):
        # reading DF from file
        with open(self.df_path, 'rb') as file:
            df = pickle.load(file)

        # adding new column, which assigns index to spectra which are from same substrate (id based)

        df.sort_values(by='id', inplace=True)

        df.reset_index(inplace=True, drop=True)

        df['substrate id'] = df.index // 11
        # moving columns in the more natural order
        df = self.move_columns(df)

        #df = df.iloc[0:110, :]

        return df

    def where_spectra_ends(self):

        for ind, item in enumerate(self.raw_data.columns):

            if isinstance(item, str):
                return ind - 1

    def get_surface(self):
        surface = np.sum(self.raw_data.iloc[:, 0:self.spectra_end], axis=1)

        return surface

    def get_ef(self):
        ef = np.log(self.raw_data.loc[:, 'peak1': 'peak5'].min(axis=1))

        return ef

    @staticmethod
    def get_peaks_baseline(x, y):
        base, coeff = utils.baseline(y, deg=7)
        
        amplitude = y - base
        # finding indices of many peaks, due to low height argument
        peaks_position = signal.find_peaks(amplitude, height=10, distance=80)[0]

        peaks = x[np.array(peaks_position, dtype=np.int32)]
        width = signal.peak_widths(amplitude, peaks_position)[0]
        peaks_height = amplitude[peaks_position]

        dic = {}

        for item in range(len(peaks)):
            dic[peaks[item]] = (width[item], peaks_height[item])

        return dic

    @staticmethod
    def fill_dictioanairy(dic, list):
        for key, value in zip(dic.keys(), list):
            dic[key].append(value)

    def get_peaks_baseline_df(self):

        dic_list = []

        for i in range(self.raw_data.shape[0]):
            series = self.raw_data.iloc[i, 0:self.spectra_end].sort_index().dropna()
            intensity = series.values

            raman_shift = series.index.to_numpy(dtype=np.int64)

            peaks_dictionairy = self.get_peaks_baseline(raman_shift, intensity)

            dic_list.append(peaks_dictionairy)


        df = pd.DataFrame({'peaks dic': dic_list})

        return df

    def normalize_data(self, df, columns):

        for column in columns:
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

        return df

    def analyze_data(self):
        data = pd.DataFrame()

        data['id'], data['substrate number'] = self.raw_data['id'], self.raw_data['substrate id']

        data['ln(ef)'] = self.get_ef()

        data['surface'] = self.get_surface()

        peaks_coeff_df = self.get_peaks_baseline_df()

        data = pd.concat((data, peaks_coeff_df), axis=1)

        # columns_to_normalize = ['surface', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

        # data = self.normalize_data(data, columns_to_normalize)

        return data


if __name__ == '__main__':
    df_path = '../DataFrame/df.pkl'

    data = Preprocessing(df_path)

    with open('../DataFrame/df_peaks_dic.pkl', 'wb+') as data_frame:
        pickle.dump(data.processed_data, data_frame)
