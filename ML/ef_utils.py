import streamlit as st
import plotly.express as px

from . import vis_utils
import processing
from visualisation.draw import fig_layout

def get_laser_intensities():
    # # #
    # # Intensities
    #
    vis_utils.print_widget_labels('Intensity', 5, 0)
    cols = st.columns(2)
    with cols[0]:
        i_raman = st.number_input('Raman Intensity', 1, 10000, 1000, 100)
    with cols[1]:
        i_sers = st.number_input('SERS Intensity', 1, 500000, 60000, 1000)
    return i_sers, i_raman


def num_of_molecules(conc, vol):
    """
    Calculating number of particles contained in the solution.
    :param conc: Float, Must be in mol/dm**3
    :param vol: Float, Must be in ml
    :return: Float, Number of Molecules
    """
    n_av = 6.023 * 10 ** 23
    v_dm = vol * 10 ** (-3)
    
    return n_av * conc * v_dm


def cal_size_of_laser_spot(wave_length_nm, lens_numeric_aperture):
    """
    Calculating size of laser spot.
    :param wave_length_nm: Int, laser wavelength given in nm
    :param lens_numeric_aperture: Float, Lens parameter - Numerical aperture
    :return: Float, Size of laser spot in meters
    """
    wave_length_m = wave_length_nm * 10 ** (-9)  # changing nm to m as it is needed for formula
    return (1.22 * wave_length_m) / lens_numeric_aperture


def cal_n_raman(v_compound, compound_density, compound_molecular_weight):
    """
    Calculating the number of molecules of the irradiated crystal
    :param v_compound: The volume of your chemical compound crystal subjected to laser illumination [m^3]
    :param compound_density:density of the chemical compound g/cm^3
    :param compound_molecular_weight: Mass of the irradiated crystal [g]
    :return: Float, N_Raman - number of molecules of the irradiated crystal
    """
    # calculating the mass of the irradiated crystal [g]
    m = v_compound * (compound_density * 10 ** 6)
    # calculating the number of moles of the irradiated crystal [mol]
    n = m / compound_molecular_weight
    
    # mol * 1/mol
    return n * 6.02e+23  # calculating the number of molecules using the Avogadro constant


def get_df(file, spectrometer):
    df = processing.save_read.files_to_df(file, spectrometer)
    df = df.interpolate().bfill().ffill()
    return df


def draw_plots_for_ef(df, plot_palette, plot_template, plot_x_min, plot_x_max, plot_y_min, plot_y_max):
    fig = px.line(df, color_discrete_sequence=plot_palette)
    fig_layout(plot_template, fig, plots_colorscale=plot_palette)
    fig.update_xaxes(range=[plot_x_min, plot_x_max])
    fig.update_yaxes(range=[plot_y_min, plot_y_max])
    
    return fig


def get_axes_values(df):
    x_min = int(df.index.min())
    x_max = int(df.index.max())
    y_min = int(df.iloc[:, :].min().min())
    y_max = int(df.iloc[:, :].max().max())
    
    return x_min, x_max, y_min, y_max
