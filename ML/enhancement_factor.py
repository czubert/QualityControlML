import numpy as np
from ML import ef_utils


def calculate_ef(i_sers=100000, i_raman=1000):
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
    
    # # Surface area development coefficient
    surface_dev = 2
    
    # # The area of active surface of the SERS substrate
    s_platform = x_dimension * y_dimension * 10 ** (-6) * surface_dev
    
    # # The coverage of the analyte on the surface between 10^-6 and 6*10^-6 ~=10%
    surface_coverage = 0.1
    
    # n_sers = (num_molecules * s_laser * surface_coverage) / s_platform  # formula version (s_laser zamiast s0)
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
    # raman and sers intensities must be provided as parameters
    enhancement_factor = (i_sers / n_sers) * (n_raman / i_raman)
    
    return enhancement_factor


if __name__ == "__main__":
    ef = calculate_ef()
    power = int(np.log10(ef)//1)
    ef_shortened = ef/10**power
    print(f'{ef_shortened} x 10^{power}')
