import pandas as pd
import matplotlib.pyplot as plt
import pickle

class Spectra():

    def __init__(self):

        self.data = self.read_data()

    def read_data(self):

    # reading DF from file
        with open('./DataFrame/df.pkl', 'rb') as file:

            df = pickle.load(file)

    # adding new column, which assigns index to spectra which are from same substrate (id based)

        df.sort_values(by='id', inplace = True)

        df.reset_index(inplace = True, drop = True)

        df['substrate id'] = df.index // 11

        return df

    # def plot_graph(self, id):
    #
    #     mask = self.data[self.data['substrate id'] == id]
    #
    #     plt.plot(self.data[mask, ])


spec = Spectra()
