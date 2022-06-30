import numpy as np
import plotly.express as px
import streamlit as st

import processing
from constants import LABELS
from visualisation.draw import fig_layout
import ef_utils
from . import vis_utils

SLIDERS_PARAMS_RAW = {'rel_height': dict(min_value=1, max_value=100, value=20, step=1),
                      'height': dict(min_value=1000, max_value=100000, value=10000, step=1000),
                      }
SLIDERS_PARAMS_NORMALIZED = {'rel_height': dict(min_value=0.01, max_value=1., value=0.5, step=0.01),
                             'height': dict(min_value=0.001, max_value=1., value=0.1, step=0.001),
                             }

spectra_types = ['EMPTY', 'BWTEK', 'RENI', 'WITEC', 'WASATCH', 'TELEDYNE', 'JOBIN']


def main():
    """
    Calculating EF for spectra quality labeling
    :return:  Float
    """
    
    # # # #
    # # #
    # # First step of calculations
    
    # #  Concentration of analyte in solution (in mol/dm^3)
    concentration = 1 * 10 ** (-6)
    
    # # Volume of solution (ml)
    volume = 200.0
    
    # # Calculating the number of molecules
    num_molecules = ef_utils.num_of_molecules(concentration, volume)
    
    # # # #
    # # #
    # # Second step of calculations - Calculating the laser spot (S_{Laser}S
    #
    
    # # Laser wavelength in nm
    laser_wave_length = 785
    
    # # Lens parameter - Numerical aperture
    lens_params = 0.22
    
    # # Calculating the Laser spot
    s_laser = ef_utils.cal_size_of_laser_spot(laser_wave_length, lens_params)
    
    # # # #
    # # #
    # # Third step of calculations - Calculating the surface area irradiated with the laser S0
    #
    
    s0_spot_area = np.pi * (s_laser / 2) ** 2
    
    # # # #
    # # #
    # # Fourth step of calculations - Determination of the number of molecules per laser irradiated surface Nsers
    #
    
    # # Active surface dimensions
    x_dimension = 3.5
    y_dimension = 3.5
    
    # # The area of active surface of the SERS substrate
    s_platform = ef_utils.get_active_surface_area(x_dimension, y_dimension)
    
    # # The coverage of the analyte on the surface between 10^-6 and 6*10^-6 ~=10%
    surface_coverage = 0.1
    
    # n_sers = (num_molecules * s_laser * surface_coverage) / s_platform  # formula version
    n_sers = (num_molecules * s0_spot_area * surface_coverage) / s_platform  # Szymborski use
    
    # # # #
    # # #
    # # Fifth step of calculations - Calculation of the volume from which the Raman
    # # signal for your compound in solids is recorded
    #
    
    penetration_depth = 2
    v_compound = s0_spot_area * penetration_depth
    
    # # # #
    # # #
    # # Sixth step of calculations - Determining the number of p-MBA molecules
    # # from which the Raman signal (N_Raman) comes
    #

    # # Molecular weight
    compound_density = 1.3
    compound_molecular_weight = 154.19
    
    n_raman = ef_utils.cal_n_raman(v_compound, compound_density, compound_molecular_weight)
    
    # # # #
    # # #
    # # Final, seventh, step of calculations - Calculating the Enhancement Factor
    #

    # # SERS intensity and Raman Intensity
    
    intensities_options = {'input': 'Input the intensities for Raman and SERS',
                           'from_spec': 'Get Raman and SERS intensities from their spectra'}
    
    intensities_radio = st.radio('Choose whether you want to input intensities or get the values from spectra:',
                                 ['input', 'from_spec'],
                                 format_func=intensities_options.get
                                 )
    
    spectrometer = st.sidebar.selectbox("Choose spectra type",
                                        spectra_types,
                                        format_func=LABELS.get,
                                        index=0
                                        )
    if intensities_radio == 'input':
        i_sers, i_raman = ef_utils.get_laser_intensities()
    
    elif intensities_radio == 'from_spec':
        
        main_expander = st.expander("Customize your chart")
        # Choose plot colors and templates
        with main_expander:
            plot_palette, plot_template = vis_utils.get_chart_vis_properties()
        
        # TODO need to add it one day
        # rescale = st.sidebar.checkbox("Normalize")
        # if rescale:
        #     scaler = MinMaxScaler()
        #     rescaled_data = scaler.fit_transform(sers_df)
        #     df = pd.DataFrame(rescaled_data, columns=sers_df.columns, index=sers_df.index)
        #     sliders_params = SLIDERS_PARAMS_NORMALIZED
        # else:
        #     sliders_params = SLIDERS_PARAMS_RAW
        
        cols = st.columns(2)
        
        with cols[0]:
            raman_file = st.file_uploader(label='Upload Raman spectrum',
                                          accept_multiple_files=True,
                                          type=['txt', 'csv'])
            
            if not raman_file:
                st.warning("Upload Raman spectrum")
            
            else:
                raman_df = ef_utils.get_df(raman_file, spectrometer)
                
                raman_plot_x_min, raman_plot_x_max, raman_plot_y_min, raman_plot_y_max = ef_utils.get_axes_values(
                    raman_df)
        
        with cols[1]:
            sers_file = st.file_uploader(label='Upload SERS spectrum',
                                         accept_multiple_files=True,
                                         type=['txt', 'csv'])
            if not sers_file:
                st.warning("Upload SERS spectrum")
            
            else:
                sers_df = ef_utils.get_df(sers_file, spectrometer)
                
                sers_plot_x_min, sers_plot_x_max, sers_plot_y_min, sers_plot_y_max = ef_utils.get_axes_values(
                    sers_df)
        
        if not raman_file or not sers_file:
            return
        
        # Getting min and max values for both y and x axis, to make both plots of the same size,
        # and not to loose any data
        plot_x_min = min(raman_plot_x_min, sers_plot_x_min)
        plot_x_max = max(raman_plot_x_max, sers_plot_x_max)
        plot_y_min = min(raman_plot_y_min, sers_plot_y_min)
        plot_y_max = max(raman_plot_y_max, sers_plot_y_max)
        
        axes = [plot_x_min, plot_x_max, plot_y_min, plot_y_max]  # packing axes values in the list of parameters
        
        # Plotting plot of Raman spectra
        with cols[0]:
            raman_fig = ef_utils.draw_plots_for_ef(raman_df, plot_palette, plot_template, *axes)
        
        # Plotting plot of SERS spectra
        with cols[1]:
            sers_fig = ef_utils.draw_plots_for_ef(sers_df, plot_palette, plot_template, *axes)
        
        #
        bg_color = 'yellow'
        with st.columns([1, 7, 10])[1]:
            peak_range = st.slider(f'Peak range ({bg_color})',
                                   min_value=plot_x_min,
                                   max_value=plot_x_max,
                                   value=[plot_x_min, plot_x_max])
        
        if peak_range != [plot_x_min, plot_x_max]:
            raman_fig.add_vline(x=peak_range[0], line_dash="dash")
            raman_fig.add_vline(x=peak_range[1], line_dash="dash")
            raman_fig.add_vrect(x0=peak_range[0], x1=peak_range[1], line_width=0, fillcolor=bg_color, opacity=0.15)
            
            sers_fig.add_vline(x=peak_range[0], line_dash="dash")
            sers_fig.add_vline(x=peak_range[1], line_dash="dash")
            sers_fig.add_vrect(x0=peak_range[0], x1=peak_range[1], line_width=0, fillcolor=bg_color, opacity=0.15)
        
        cols = st.columns(2)
        with cols[0]:
            st.plotly_chart(raman_fig, use_container_width=True)
        with cols[1]:
            st.plotly_chart(sers_fig, use_container_width=True)
        
        raman_mask = (peak_range[0] <= raman_df.index) & (raman_df.index <= peak_range[1])
        sers_mask = (peak_range[0] <= sers_df.index) & (sers_df.index <= peak_range[1])
        raman_peak = raman_df[raman_mask]
        sers_peak = sers_df[sers_mask]
        
        i_raman = raman_peak.max()[0]
        i_sers = sers_peak.max()[0]
    else:
        raise ValueError()  # just to satisfy pycharm linter
    
    enhancement_factor = (i_sers / n_sers) * (n_raman / i_raman)
    
    st.markdown(r'$EF =$' + f' {"{:.1e}".format(enhancement_factor)}')
    
    st.markdown('---')
    st.stop()
    return enhancement_factor
