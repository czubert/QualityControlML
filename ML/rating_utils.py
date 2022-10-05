import os
import pandas as pd
import numpy as np
import plotly.express as px

from ML import enhancement_factor
import utils

# constants
PEAKS = {
    # beginning of the peak and end of the peak to estimate max and min values
    'peak1': ['1031', '1129'],
    'peak2': ['1156', '1221'],
    'peak3': ['1535', '1685'],
}

DARK = 'Dark Subtracted #1'

def get_raman_intensities(raman_pmba):
    # Getting RAMAN spectra of PMBA
    raman_pmba = raman_pmba.reset_index()
    raman_pmba.rename(columns={DARK: "Raman PMBA"}, inplace=True)
    raman_pmba = raman_pmba.set_index('Raman Shift')
    raman_pmba = raman_pmba.T  # transposition of the DF so it fits the ag_df for concat
    utils.change_col_names_type_to_str(raman_pmba)  # changes col names type from int to str, for .loc

    # Getting the value of the peak (max - min values in the range) so-called baseline subtraction,
    subtracted_raman_df = pd.DataFrame()
    for name, values in PEAKS.items():
        subtracted_raman_df.loc[:, name] = raman_pmba.loc[:, values[0]:values[1]].max(axis=1) \
                                           - raman_pmba.loc[:, values[0]:values[1]].min(axis=1)

    return subtracted_raman_df


def get_sers_intensities(grouped_files, only_new_spectra):
    # Getting SERS spectra of PMBA
    ag_df = grouped_files['ag']  # Takes only ag spectra
    ag_df = utils.change_col_names_type_to_str(ag_df)  # changes col names type from int to str, for .loc

    # This part takes only new spectra with names a1, a2 etc. - spectra collected specially for ML
    if only_new_spectra:
        mask = ag_df['id'].str.startswith('s')  # mask to get only new spectra
        ag_df = ag_df[~mask]  # Takes only new spectra out of all ag spectra

    subtracted_sers_df = pd.DataFrame()  # DataFrame that will consist only of max/min ratio for each peak
    subtracted_sers_df['id'] = ag_df['id'].str.replace(r'_.*', '')

    # TODO czy robić wstępną selekcję na podstawie widma PMBA? Że jak w jakimś punkcie, w którym nie ma peaku
    #  będzie wartość przekraczająca jakiś próg, to dajemy ocene "0"? Przez to nauczymy go też,
    #  żeby nie brać pod uwagę brzydkich widm PMBA

    # We are taking the intensities of the peak (without background) for the calculations
    for name, values in PEAKS.items():
        subtracted_sers_df.loc[:, name] = ag_df.loc[:, values[0]:values[1]].max(axis=1) \
                                          - ag_df.loc[:, values[0]:values[1]].min(axis=1)

    # TODO, czy sklejanie 2 widm na jednym podłożu ma sens? nie lepiej traktować to jako dwa różne wyniki?
    # Getting the highest ambivalent intensities for each peak (for few spots on one substrate)
    best_of_sers_subtracted = subtracted_sers_df.groupby('id').max()

    return best_of_sers_subtracted


def draw_plot(best, subtracted_raman_df):
    if not os.path.exists("images"):
        os.mkdir("images")

    for peak in PEAKS.keys():
        best['ef'] = best[peak].apply(
            lambda sers_intensity: enhancement_factor.calculate_ef(sers_intensity, subtracted_raman_df[peak]))

        best['ef'].sort_values().to_csv(f'images/EFs_{peak}.csv')

        # Hist plots
        hist_plot = px.histogram(best, x='ef', nbins=200, marginal='box', title=f'Hist plot of {peak}', width=1280,
                                 height=800)
        hist_plot.write_image(f'images/hist_plot-{peak}.jpg')

        # Bar plots
        best_sorted = best.sort_values('ef')
        bar_plot = px.bar(best_sorted, y='ef', title=f'Bar plot of {peak}', width=1280, height=800)
        bar_plot.write_image(f'images/bar_plot-{peak}.jpg')

        # Violin plots
        vio_plot = px.violin(best_sorted, y='ef', title=f'Violin plot of {peak}', width=1280, height=800)
        vio_plot.write_image(f'images/vio_plot-{peak}.jpg')

        # cumulative plots
        cum_plot = ecdf_plot(best_sorted['ef'], peak)
        cum_plot.write_image(f'images/cum_plot-{peak}.jpg')

        # showing cumulative plots in a browser
        # import plotly.io as pio
        # pio.renderers.default = 'browser'
        # pio.show(cum_plot)

def ecdf_plot(ser, title):
    """
    Cumulated Distribution Plot
    :param ser: Series
    :param title: Str
    :return: Plotly.Express Figure
    """
    sq = pd.Series(1, index=ser).sort_index()
    sq = sq.cumsum() / sq.sum()

    quant = ser.quantile(np.linspace(0, 1, 11))
    n = len(ser)

    fig = px.line(x=sq.index, y=sq)
    fig.update_layout(
        title=f'Cumulated Distribution Plot, {title}',
        xaxis=dict(
            title='Values',
            tickmode='array',
            tickvals=quant,
        ),
        yaxis=dict(
            title='Cumulative count',
            tickmode='array',
            tickvals=quant.index,
            ticktext=['{count} ({perc:.0%})'.format(count=int(n * i), perc=i) for i in quant.index]
        )
    )
    return fig
