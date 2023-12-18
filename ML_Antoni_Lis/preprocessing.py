import pandas as pd
import numpy as np
import pickle
import utils


# import matplotlib.pyplot as plt
# import joblib


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

    def analyze_peaks_baseline(self):
        coeff_list, base_line_list, peaks_list = [], [], []

        for i in range(self.raw_data.shape[0]):
            dat = self.raw_data.iloc[i, 0:self.spectra_end].sort_index().dropna()

            array = np.array(dat, dtype=np.float64)
            base_line, coefficients = utils.baseline(array, deg=7)
            peaks_name = utils.indexes(array - base_line, thres=0.4, min_dist=100)

            coeff_list.append(np.array(coefficients))
            base_line_list.append(np.array(base_line))
            peaks_list.append(np.array(dat.index[peaks_name]))

        coefficients = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

        coeff_arr = np.vstack(coeff_list)

        coeff_df = pd.DataFrame(coeff_arr, columns=coefficients)

        peaks_list = np.array(peaks_list)

        max_length = max(len(arr) for arr in peaks_list)

        padded_data = np.array(
            [np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan) for arr in peaks_list])

        peaks_data = np.array(padded_data)

        peaks_name = list(np.arange(peaks_data.shape[1]))

        peaks_name = ['p' + str(item) for item in peaks_name]

        peaks_df = pd.DataFrame(peaks_data, columns=peaks_name)

        baseline_df = pd.DataFrame({'base line': base_line_list})

        df = pd.concat((baseline_df, coeff_df, peaks_df), axis=1)

        return df

    def clean_data(self):
        pass

    def analyze_data(self):
        data = pd.DataFrame()

        data['id'], data['substrate number'] = self.raw_data['id'], self.raw_data['substrate id']

        data['surface'] = self.get_surface()

        data['ln(ef)'] = self.get_ef()

        spectra = self.raw_data.iloc[:, 0:self.spectra_end]

        peaks_baseline_df = self.analyze_peaks_baseline()

        data = pd.concat((spectra, data, peaks_baseline_df), axis=1)

        return data


if __name__ == '__main__':
    df_path = '../DataFrame/df.pkl'

    data = Preprocessing(df_path)

    with open('../DataFrame/df_processed_data_extendent.pkl', 'wb+') as data_frame:
        pickle.dump(data.processed_data, data_frame)
