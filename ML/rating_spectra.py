import os
import pandas as pd

import utils
from ML import enhancement_factor
from ML import rating_utils

# Input path
dir_path = 'data_output/step_2_group_data'
file_name = 'grouped_data'
# Output path
output_dir_path = 'data_output/step_3_rate_data'
output_file_name = 'rated_data'


def main(grouped_files, raman_pmba, chosen_peak, border_value, margin_of_error, only_new_spectra):
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

    # # Creating plots to find the best_of_sers_subtracted parameters to distinguish good from bad spectra
    # rating_utils.draw_plot(best_of_sers_subtracted, subtracted_raman)

    best_of_sers_subtracted['ef'] = best_of_sers_subtracted[chosen_peak].apply(
        lambda sers_intensity: enhancement_factor.calculate_ef(sers_intensity, subtracted_raman[chosen_peak]))

    ef_for_chosen_peak = pd.DataFrame(best_of_sers_subtracted.loc[:, 'ef'])

    """
    Selecting spectra, based on Enhancement Factor, that are of high or low quality,
    also making a gap between high and low bigger, so the estimator has higher chances
    to see the differences between "good" and "bad" spectra.
    """
    # Making the difference between good and low (1/0) spectra more significant
    low_value = int(border_value - (border_value * margin_of_error))
    high_value = int(border_value + (border_value * margin_of_error))

    low = set(ef_for_chosen_peak[ef_for_chosen_peak['ef'] < low_value].reset_index()['id'])
    high = set(ef_for_chosen_peak[ef_for_chosen_peak['ef'] > high_value].reset_index()['id'])

    # low = set(ef_for_chosen_peak.reset_index().sort_values('ef')['id'].iloc[:low_value])
    # high = set(ef_for_chosen_peak.reset_index().sort_values('ef')['id'].iloc[high_value:])

    # Creating a tuple of high and low spectra without the spectra that are close to the middle
    all_spectra = {*high, *low}

    # Getting all background spectra of silver substrates
    df = grouped_files['ag_bg']

    # Getting only selected (high, low) spectra out of all spectra, based on max/min ratio of the peaks
    mask = df['id'].str[:-2].isin(all_spectra)
    df_chosen = df[mask]

    # Giving a  1/0 label to a spectrum, depending on if a spectrum is in a list of high or low of max/min ratio
    df_chosen['y'] = df['id'].str[:-2].isin(high).astype(int)

    # Saving results to a joblib file
    utils.save_as_joblib(df_chosen, output_file_name, output_dir_path)

    return df_chosen


def rate_spectra(grouped_files, raman_pmba, border_value, margin_of_error, chosen_peak, only_new_spectra,
                 read_from_file=False):
    if read_from_file:
        if not os.path.isfile(output_dir_path + '//' + output_file_name + '.joblib'):
            return main(grouped_files, raman_pmba, chosen_peak, border_value, margin_of_error, only_new_spectra)
        else:
            return utils.read_joblib(output_file_name, output_dir_path)

    else:
        return main(grouped_files, raman_pmba, chosen_peak, border_value, margin_of_error, only_new_spectra)


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
