import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib


class Spectra:

    def __init__(self, path):
        self.path = path
        # TODO może read_background_spec, read_bg_spectra? bo "data" jest trochę ogólne, a to dotyczy chyba tylko teł?
        self.data = self.read_data()
        self.pmba_spec = self.get_pmba_spec()  # TODO może load_pmba_spec, albo load_pmba?
        self.averaged_data = self.average_data()

    # TODO
    #  jeżeli nie używasz obiektu w metodzie (self), to taka metoda powinna być statyczna. na poniższej metodzie pokażę:
    @staticmethod
    def get_surface(df):
        """

        :param df:
        :return:
        """
        surface = np.sum(df.iloc[:, 0:-9], axis=1)

        return surface

    # TODO static
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

    # TODO static
    def get_ef(self, df):
        df['ef'] = df.loc[:, 'peak1': 'peak5'].min(axis=1)

        to_check = df.loc[:, 'peak1': 'peak5'].columns

        df['chosen peak'] = df[to_check].idxmin(axis=1)

        return df

    # TODO Is this the best way to average it out?
    def average_data(self):
        pass  # TODO tego już nie powinno tutaj być
        # EF, ID, surface, substrate ID

        df = self.get_ef(self.data)

        surface_series, ef_series, id_series = [], [], []

        for id in df['substrate id'].unique():
            mask = df['substrate id'] == id

            dat = df[mask]

            surface_series.append(dat.sort_values('surface').iloc[2:-2]['surface'].mean())
            ef_series.append(dat.sort_values('ef').iloc[2:-2]['ef'].mean())
            id_series.append(id)

        dat = pd.DataFrame({'surface': surface_series, 'ef': ef_series, 'substrate id': id_series})

        dat['ef'] = dat['ef'] / 10 ** 6

        return dat

    # TODO
    #  prawie wszedzię masz, że parametr zakrywa taki sam spoza funkcji, niestety trzebaby inaczej nazwać,
    #  też mnie to wkurza xD
    def get_substrate(self, id):
        # selecting data with good ID
        mask = self.data['substrate id'] == id

        df = self.data[mask]

        return df

    def plot_background(self, id):
        df = self.data[self.data['substrate id'] == id]

        text = 'EF: ' + str(round(df['ef'].max() / self.data['ef'].max(), 3))

        for i in range(11):
            plt.plot(df.columns[0: -12], df.iloc[i, 0: -12])

        plt.text(1500, 45000, text)

    def plot_pmba(self, id):
        id = self.data[self.data['substrate id'] == id]['id']
        df = self.pmba_spec[self.pmba_spec['id'].isin(id)]

        for i in range(11):
            plt.plot(df.columns[0: -3], df.iloc[i, 0: -3])

    def plot(self, id):
        fig = plt.figure(figsize=(10, 4))
        gs = fig.add_gridspec(1, 2)

        # TODO
        #  poniższe zmienne raczej nie są konieczne, bo operujesz na obiekcie fig i gs, a potem wywwolujesz funkcje,
        #  ktore nic nie zwracają wiec pod te zmienne nci sie nie przypisuje

        ax1 = fig.add_subplot(gs[0, 0])
        ax1 = self.plot_background(id)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2 = self.plot_pmba(id)
        plt.tight_layout()

    # TODO static
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
