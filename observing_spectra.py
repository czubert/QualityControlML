import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib


class Spectra():

    def __init__(self, path):
        self.path = path
        self.data = self.read_data()
        self.pmba_spec = self.get_pmba_spec()
        self.averaged_data = self.average_data()

    def get_surface(self, df):
        """

        :param df:
        :return:
        """
        surface = np.sum(df.iloc[:, 0:-7], axis=1)

        return surface

    def move_columns(self, df):
        col = list(df.columns)

        moved_columns = ['id', 'laser_power', 'integration_time']

        col = [item for item in col if item not in moved_columns]

        col = col + moved_columns

        df = df[col]

        return df

    def get_pmba_spec(self):
        path = './QualityControlML/data_output/step_2_group_data/grouped_data.joblib'

        dictionairy = joblib.load(path)

        df = dictionairy['ag']

        df = self.move_columns(df)

        df.sort_values(by='id', inplace=True)

        df.reset_index(inplace=True, drop=True)

        return df

    def read_data(self):
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

    def get_ef(self, df):
        pass

    # TODO Is this the best way to average it out?
    def average_data(self):
        pass
    # EF, ID, surface, substrate ID

    def get_substrate(self, id):
        # selecting data with good ID
        mask = self.data['substrate id'] == id

        df = self.data[mask]

        return df

    def get_correlation(self, x, y):
        x_avg, y_avg = np.sum(x), np.sum(y)

        x_dim, y_dim = np.meshgrid(x, y, indexing='ij')

        product = (x_dim - x_avg) * (y_dim - y_avg)

        amplitude = np.sqrt(np.sum((x_dim - x_avg)) ** 2 * np.sum((y_dim - y_avg) ** 2))

        plt.imshow(product / amplitude, cmap='gray')
        plt.colorbar()

        # return np.sum(product) / amplitude


if __name__ == '__main__':
    path = './DataFrame/df.pkl'

    spec = Spectra(path)

    # plt.plot(spec.data['surface'], spec.data['peak2'], '.')

    plt.hist(spec.data['peak1'], bins=40)

    # x = np.random.randint(10, size = 500)

    # corr = spec.get_correlation(spec.data['peak3'], spec.data['surface'])

    # spec.get_correlation(spec.data['peak3'], spec.data['surface'])
    print('finished')
