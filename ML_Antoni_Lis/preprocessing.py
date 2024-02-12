import pandas as pd
from scipy import signal
import numpy as np
import pickle
import utils

#TODO 
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
        peaks_position = signal.find_peaks(amplitude, height=100, distance=80)[0]
        #amplitude of those peaks
        peaks_amplitude = amplitude[peaks_position]

        # choosing indices of 5 highest amplitudes
        highest_peak_indices = np.argsort(peaks_amplitude)[::-1][0:5]

        #getting indices of the chosen peaks and sorting them
        chosen_peaks_position = np.sort(peaks_position[highest_peak_indices])

        peaks_height = y[np.array(chosen_peaks_position, dtype=np.int32)]

        width = signal.peak_widths(amplitude, chosen_peaks_position)

        return coeff, x[peaks_position], peaks_height, width[0]

    @staticmethod
    def fill_dictioanairy(dic, list):
        for key, value in zip(dic.keys(), list):
            dic[key].append(value)

    def get_peaks_baseline_df(self):

        baseline_dic = {'a1': [], 'a2': [], 'a3': [], 'a4': [], 'a5': [], 'a6': [], 'a7': [], }

        pos_dic = {'p1': [], 'p2': [], 'p3': [], 'p4': [], 'p5': []}

        height_dic = {'h1': [], 'h2': [], 'h3': [], 'h4': [], 'h5': []}

        width_dic = {'w1': [], 'w2': [], 'w3': [], 'w4': [], 'w5': []}

        for i in range(self.raw_data.shape[0]):

            series = self.raw_data.iloc[i, 0:self.spectra_end].sort_index().dropna()
            intensity = series.values

            raman_shift = series.index.to_numpy(dtype=np.int64)

            coeff, peaks_position, peaks_height, width = self.get_peaks_baseline(raman_shift, intensity)

            self.fill_dictioanairy(baseline_dic, coeff)
            self.fill_dictioanairy(pos_dic, peaks_position)
            self.fill_dictioanairy(height_dic, peaks_height)
            self.fill_dictioanairy(width_dic, width)

        combined_dic = {**baseline_dic, **pos_dic, **height_dic, **width_dic}

        df = pd.DataFrame(combined_dic)

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

    with open('../DataFrame/df_processed_data.pkl', 'wb+') as data_frame:
        pickle.dump(data.processed_data, data_frame)
