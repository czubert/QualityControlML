import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib

import utils


class Spectra:

    def __init__(self, path, baseline_df=False):
        self.path = path
        self.bg_data = self.read_bg_spectra()
        self.pmba_data = self.load_pmba()
        self.averaged_data = self.average_data()
        self.baseline_coeff = self.get_baseline(baseline_df)
        #self.peaks = self.peak_analysis()

    @staticmethod
    def get_surface(df):
        """

        :param df:
        :return:
        """
        surface = np.sum(df.iloc[:, 0:-9], axis=1)

        return surface

    @staticmethod
    def move_columns(daf):
        col = list(daf.columns)

        moved_columns = ['id', 'laser_power', 'integration_time']

        col = [item for item in col if item not in moved_columns]

        col = col + moved_columns

        dat = daf[col]

        return dat

    def load_pmba(self):
        pmba_path = './data_output/step_2_group_data/grouped_data.joblib'

        dictionairy = joblib.load(pmba_path)

        df = dictionairy['ag']

        df = self.move_columns(df)

        df.sort_values(by='id', inplace=True)

        df.reset_index(inplace=True, drop=True)

        return df

    def read_bg_spectra(self):
        # reading DF from file
        with open(self.path, 'rb') as file:
            df = pickle.load(file)

        # adding new column, which assigns index to spectra which are from same substrate (id based)

        df.sort_values(by='id', inplace=True)

        df.reset_index(inplace=True, drop=True)

        df['substrate id'] = df.index // 11

        # moving columns in the more natural order

        df = self.move_columns(df)

        df['surface'] = self.get_surface(df)

        mask = df['surface'] > 0

        dat = df.loc[mask]

        return dat

    @staticmethod
    def get_ef(df):
        df['ef'] = df.loc[:, 'peak1': 'peak5'].min(axis=1)

        to_check = df.loc[:, 'peak1': 'peak5'].columns

        df['chosen peak'] = df[to_check].idxmin(axis=1)

        return df

    # TODO Is this the best way to average it out?
    def average_data(self):

        df = self.get_ef(self.bg_data)  # EF, ID, surface, substrate ID

        surface_series, ef_series, id_series = [], [], []

        for substrate_id in df['substrate id'].unique():
            mask = df['substrate id'] == substrate_id

            dat = df[mask]

            surface_series.append(dat.sort_values('surface').iloc[2:-2]['surface'].mean())
            ef_series.append(dat.sort_values('ef').iloc[2:-2]['ef'].mean())
            id_series.append(substrate_id)

        dat = pd.DataFrame({'surface': surface_series, 'ef': ef_series, 'substrate id': id_series})

        dat['ef'] = dat['ef'] / 10 ** 6

        return dat

    def get_baseline(self, baseline_df):

        if baseline_df:
            coeff_list = []

            for i in range(self.bg_data.shape[0]):
                array = np.array(self.bg_data.iloc[i, :-12].dropna(), dtype=np.float64)
                _, coefficients = utils.baseline(array, deg=5)
                coeff_list.append(coefficients)

            coeff_arr = np.vstack(coeff_list)
            df = pd.DataFrame(coeff_arr, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
            baseline_df = pd.concat([self.bg_data['id'], self.bg_data['ef'], df], axis=1)

        return baseline_df

    # methods usefull while working in jupyter notebook:
    def get_substrate(self, substrate_id):
        # selecting data with good ID
        mask = self.bg_data['substrate id'] == substrate_id

        df = self.bg_data[mask]

        return df
    # def peak_analysis(self):
    #
    #     df['id'], df['ef'] = self.bg_data['substrate id'], self.bg_data['ef']


    def plot_background(self, substrate_id):
        """

        Parameters
        ----------
        substrate_id

        Returns
        -------

        """
        df = self.bg_data[self.bg_data['substrate id'] == substrate_id]

        text = 'EF: ' + str(round(df['ef'].max() / self.bg_data['ef'].max(), 3))

        for i in range(df.shape[0]):
            plt.plot(df.columns[0: -12], df.iloc[i, 0: -12])

        plt.text(1500, 45000, text)

    def plot_pmba(self, substrate_id):

        correct = self.bg_data[self.bg_data['substrate id'] == substrate_id]['id']
        df = self.pmba_data[self.pmba_data['id'].isin(correct)]

        for i in range(df.shape[0]):
            plt.plot(df.columns[0: -3], df.iloc[i, 0: -3])

    def plot(self, substrate_id):
        fig = plt.figure(figsize=(10, 4))
        gs = fig.add_gridspec(1, 2)

        plt.rcParams["axes.linewidth"] = 2
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.width'] = 2
        fig.add_subplot(gs[0, 0])

        self.plot_background(substrate_id)
        plt.ylabel('intensity', fontweight='bold', fontname="Calibri", fontsize=12)
        plt.xlabel('raman shift [nm]', fontweight='bold', fontname="Calibri", fontsize=12)
        plt.xticks(fontsize=8, fontweight='bold')
        plt.yticks(fontsize=8, fontweight='bold')

        plt.title('Background Raman spectra', fontsize=13, fontweight='bold')

        fig.add_subplot(gs[0, 1])
        self.plot_pmba(substrate_id)
        plt.ylabel('intensity', fontweight='bold', fontname="Calibri", fontsize=12)
        plt.xlabel('raman shift [nm]', fontweight='bold', fontname="Calibri", fontsize=12)
        plt.xticks(fontsize=8, fontweight='bold')
        plt.yticks(fontsize=8, fontweight='bold')

        plt.title('pmba Raman spectra', fontsize=13, fontweight='bold')

        plt.tight_layout()

    @staticmethod
    def get_correlation(x, y):
        x_avg, y_avg = np.sum(x), np.sum(y)

        x_dim, y_dim = np.meshgrid(x, y, indexing='ij')

        product = (x_dim - x_avg) * (y_dim - y_avg)

        # amplitude = np.sqrt(np.sum((x_dim - x_avg)) ** 2 * np.sum((y_dim - y_avg) ** 2))

        plt.imshow(product, cmap='gray')
        plt.colorbar()

        # return np.sum(product) / amplitude


if __name__ == '__main__':
    df_path = './DataFrame/df.pkl'

    spec = Spectra(df_path, baseline_df=True)

    df = spec.baseline_coeff

    # df = spec.pmba_data.iloc[100, 0:-15].dropna()
    #
    # base, _ = utils.baseline(df, deg=5)
    #
    # df = df - base
    #
    # #plt.plot(df.index, base)
    #
    # plt.plot(df)
    #
    # #plt.plot(df-base)
    #
    # peaks = utils.indexes(np.array(df, dtype=np.float64), thres=0.5, min_dist=10)
    #
    # peaks_position = []
    #
    # for i in range(peaks.size):
    #     peaks_position.append(df.index[peaks[i]])
