import os
import pandas as pd
import numpy as np
import plotly.express as px

from ML import enhancement_factor


def main(peaks, best, subtracted_raman_df):
    if not os.path.exists("images"):
        os.mkdir("images")

    for peak in peaks.keys():
        best['ef'] = best[peak].apply(
            lambda raman_intensity: enhancement_factor.calculate_ef(raman_intensity, subtracted_raman_df[peak]))

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


def ecdf_plot(ser, title):
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


