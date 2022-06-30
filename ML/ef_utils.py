import streamlit as st
import plotly.express as px


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
