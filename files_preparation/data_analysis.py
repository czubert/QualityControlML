import pandas as pd
import utils


peaks = {
    # beginning of the peak and end of the peak to estimate max and min values
    'peak1': ['671', '761'],
    'peak2': ['801', '881'],
    'peak3': ['970', '1031'],
    'peak4': ['1530', '1641'],
}

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


    """
    Peak ratio max/min value violin plot
    """
    ratio_df = pd.DataFrame() # DataFrame that will consists only of max/min ratio for each peak
    
    # Getting ratio between max and mean value for each peak
    for name, values in peaks.items():
        ratio_df.loc[:, name] = new_spectra_df.loc[:, values[0]:values[1]].max(axis=1) \
                                / new_spectra_df.loc[:, values[0]:values[1]].min(axis=1)
    
    fig = px.violin(ratio_df, y=peaks)
    plot(fig)


    """
    Peak absolute value violin plot
    """

    abs_df = pd.DataFrame() # DataFrame that will consists only of max/min ratio for each peak
    
    # Violin plot on peak values, that were count as mix-min value for each peak
    for name, values in peaks.items():
        abs_df.loc[:, name] = new_spectra_df.loc[:, values[0]:values[1]].max(axis=1) \
                                - new_spectra_df.loc[:, values[0]:values[1]].min(axis=1)


    fig2 = px.violin(abs_df, y=peaks)
    plot(fig2)

    return ratio_df, abs_df

if __name__ == "__main__":
    dir_path = 'data_output/step_3_rate_data'
    file_name = 'rated_data'
    rated_spectra = utils.read_joblib(file_name, '../' + dir_path)
    ratio_df, abs_df = run(rated_spectra)
    