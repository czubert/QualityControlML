import os
import pandas as pd
import pickle

import utils
from ML import enhancement_factor
from ML import rating_utils

# Input path
dir_path = 'data_output/step_2_group_data'
file_name = 'grouped_data'
# Output path
output_dir_path = 'data_output/step_3_rate_data'
output_file_name = 'rated_data'


def main(grouped_files, raman_pmba, chosen_peak, border_value, margin_of_error, only_new_spectra, save):
    """
    Module created to rate spectra based on the enhancement factor.
    First it calculates enhancement factor for each spectrum.
    Then creates plots of enhancement factor to find a place which suggests that it might be the border between "good" and "bad" spectra / substrate.
    :param grouped_files:
    :param raman_pmba:
    :param chosen_peak:
    :param border_value:
    :param margin_of_error:
    :param only_new_spectra:
    :return:
    """
    # Getting RAMAN intensities of PMBA
    subtracted_raman = rating_utils.get_raman_intensities(raman_pmba)

    # Getting SERS intensities of PMBA
    best_of_sers_subtracted = rating_utils.get_sers_intensities(grouped_files, only_new_spectra)

    # Creating plots to find the best_of_sers_subtracted parameters to distinguish good from bad spectra
    #rating_utils.draw_plot(best_of_sers_subtracted, subtracted_raman)

    # Calculating enhancement factor for each  peak

    df = grouped_files['ag_bg']

    for item in chosen_peak:

        peak = item['name']

        best_of_sers_subtracted[peak] = best_of_sers_subtracted[peak].apply(
            lambda sers_intensity: enhancement_factor.calculate_ef(sers_intensity, subtracted_raman[peak]))

        ef_for_chosen_peak = pd.DataFrame(best_of_sers_subtracted.loc[:, peak])

        df[peak] = ef_for_chosen_peak[peak]

    if save:

        with open('./DataFrame/df.pkl', 'wb') as file:

            pickle.dump(df, file)

    return df


def rate_spectra(grouped_files, raman_pmba, border_value, margin_of_error, chosen_peak, only_new_spectra,
                 read_from_file=False, save = True):
    if read_from_file:
        if not os.path.isfile(output_dir_path + '//' + output_file_name + '.joblib'):
            return main(grouped_files, raman_pmba, chosen_peak, border_value, margin_of_error, only_new_spectra)
        else:
            return utils.read_joblib(output_file_name, output_dir_path)

    else:
        return main(grouped_files, raman_pmba, chosen_peak, border_value, margin_of_error, only_new_spectra, save)


if __name__ == "__main__":
    from files_preparation import reading_data

    grouped_files = utils.read_joblib(file_name, '../' + dir_path)
    raman_pmba = reading_data.read_spectrum('../data_input/PMBA__powder_10%_1s.txt')

    rated_spectra = main(grouped_files,
                         raman_pmba,
                         chosen_peak='peak1',
                         border_value=65*1000000,
                         margin_of_error=0.10,
                         only_new_spectra=True)

    print('done')
