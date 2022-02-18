import pandas as pd
import utils


peaks = {
    # beginning of the peak and end of the peak to estimate max and min values
    'peak1': ['671', '761'],
    'peak2': ['801', '881'],
    'peak3': ['970', '1031'],
}
# 'peak4': ['1530', '1641'],
"""
Part responsible for the estimation of the limit which NEW spectra is good and which not
"""
import plotly.express as px
from plotly.offline import plot


def run(rated_spectra):
    # Getting relevant data
    spectra_df = rated_spectra['ag'] # Takes only ag spectra
    mask = spectra_df['id'].str.startswith('s')  # mast to get only new spectra
    new_spectra_df = spectra_df[~mask] # Takes only new spectra out of all ag spectra

    # violin plots
    ratio_df = plot_ratio(new_spectra_df)
    abs_df = plot_absolute(new_spectra_df)
    best, worst = plot_best_worst_ratio(new_spectra_df)

    # line plots
    # best_line = px.line(best)
    # ratio_above_median = plot_ratio_above_median(ratio_df, rated_spectra)

    return ratio_df, abs_df, best, worst
    # return ratio_df, abs_df, best, worst, ratio_above_median

def plot_ratio(new_spectra_df):
    """
    Peak ratio max/min value violin plot
    """
    ratio_df = pd.DataFrame() # DataFrame that will consists only of max/min ratio for each peak
    
    # Getting ratio between max and mean value for each peak
    for name, values in peaks.items():
        ratio_df.loc[:, name] = new_spectra_df.loc[:, values[0]:values[1]].max(axis=1) \
                                / new_spectra_df.loc[:, values[0]:values[1]].min(axis=1)
    
    # fig = px.violin(ratio_df, y=peaks, title='Ratio')
    # fig.show('browser')
    
    return ratio_df

def plot_absolute(new_spectra_df):
    """
    Peak absolute value violin plot
    """

    abs_df = pd.DataFrame() # DataFrame that will consists only of max/min ratio for each peak
    
    # Violin plot on peak values, that were count as mix-min value for each peak
    for name, values in peaks.items():
        abs_df.loc[:, name] = new_spectra_df.loc[:, values[0]:values[1]].max(axis=1) \
                                - new_spectra_df.loc[:, values[0]:values[1]].min(axis=1)


    # fig = px.violin(abs_df, y=peaks, title='Absolut Values')
    # fig.show('browser')
    
    return abs_df


def plot_best_worst_ratio(new_spectra_df):
    """
    Peak ratio max/min value violin plot
    """
    ratio_df = pd.DataFrame() # DataFrame that will consists only of max/min ratio for each peak
    ratio_df['id'] = new_spectra_df['id'].str.replace(r'_.*', '')
    
    # Getting ratio between max and mean value for each peak
    for name, values in peaks.items():
        ratio_df.loc[:, name] = new_spectra_df.loc[:, values[0]:values[1]].max(axis=1) \
                                / new_spectra_df.loc[:, values[0]:values[1]].min(axis=1)
    
    # Getting best ratio for each peak for each substrate
    best = ratio_df.groupby('id').max()

    # Getting worst ratio for each peak for each substrate
    worst = ratio_df.groupby('id').min()
    
    # fig = px.violin(best, y=peaks, title="Best ratio")
    # fig.show('browser')
    #
    # fig2 = px.violin(worst, y=peaks, title="Worst ratio")
    # fig2.show('browser')
    
    return best, worst


def plot_ratio_above_median(ratio_df, rated_spectra):
    above_median  = ratio_df[ratio_df['peak1'] > ratio_df['peak1'].median()]

    above_median = above_median.iloc[0:1,:]
    
    above_median_list = above_median.index
    above_median_to_plot = rated_spectra['ag'].filter(above_median_list, axis=0).iloc[:,:-3]
    
    print(above_median_to_plot.columns)
    print()
    print()
    print(above_median_to_plot.index)
    print()
    print()
    print(above_median_to_plot)
    
    
    figu = px.line(above_median_to_plot, x=above_median_to_plot.columns, y=above_median_to_plot.values[0])
    # figu = px.line(above_median_to_plot)

    figu.show('browser')
    return above_median_to_plot

if __name__ == "__main__":
    dir_path = 'data_output/step_3_rate_data'
    file_name = 'rated_data'
    rated_spectra = utils.read_joblib(file_name, '../' + dir_path)
    ratio_df, abs_df, best, worst = run(rated_spectra)
    # ratio_df, abs_df, best, worst, ratio_above_median = run(rated_spectra)
    