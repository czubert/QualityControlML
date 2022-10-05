import os
import pandas as pd

import enhancement_factor
import utils
import img_for_ef_evaluation
import rating_utils

# constants
peaks = {
    # beginning of the peak and end of the peak to estimate max and min values
    'peak1': ['1031', '1119'],
    'peak2': ['1156', '1221'],
    'peak3': ['1535', '1685'],
}
# Input path
dir_path = 'data_output/step_2_group_data'
file_name = 'grouped_data'
# Output path
output_dir_path = 'data_output/step_3_rate_data'
output_file_name = 'rated_data'



def main(grouped_files, raman_pmba, chosen_peak, border_value, margin_of_error, only_new_spectra):
    # Getting RAMAN spectra of PMBA
    subtracted_raman_df = rating_utils.get_raman_intensities(raman_pmba)


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

    # We are taking the height of the peak (without background) for the calculations
    for name, values in peaks.items():
        subtracted_sers_df.loc[:, name] = ag_df.loc[:, values[0]:values[1]].max(axis=1) \
                                          - ag_df.loc[:, values[0]:values[1]].min(axis=1)

    # TODO, czy sklejanie 2 widm na jednym podłożu ma sens? nie lepiej traktować to jako dwa różne wyniki?
    # Getting the highest ambivalent intensities for each peak (for few spots on one substrate)
    best = subtracted_sers_df.groupby('id').max()

    # Creating plots to find the best parameters to distinguish good from bad spectra
    img_for_ef_evaluation.main(peaks, best, subtracted_raman_df)

    best['ef'] = best[chosen_peak].apply(
        lambda sers_intensity: enhancement_factor.calculate_ef(sers_intensity, subtracted_raman_df[chosen_peak]))

    """
    Selecting spectra, based on the max/min ratio, that are of high or low quality,
    also making a gap between high and low bigger, so the estimator has higher chances
    to see the differences between low and high spectra.
    """
    # # Making the difference between good and low (1/0) spectra more significant
    # low_value = int(border_value - (border_value * margin_of_error))
    # high_value = int(border_value + (border_value * margin_of_error))
    #
    # low = set(best.reset_index().sort_values(chosen_peak)['id'].iloc[:low_value])
    # high = set(best.reset_index().sort_values(chosen_peak)['id'].iloc[high_value:])
    #
    # # Creating a tuple of high and low spectra without the spectra that are close to the middle
    # all_spectra = {*high, *low}
    #
    # # Getting all background spectra of silver substrates
    # df = grouped_files['ag_bg']
    #
    # # Getting only selected (high, low) spectra out of all spectra, based on max/min ratio of the peaks
    # mask = df['id'].str[:-2].isin(all_spectra)
    # df_chosen = df[mask]
    #
    # # Giving a  1/0 label to a spectrum, depending on if a spectrum is in a list of high or low of max/min ratio
    # df_chosen['y'] = df['id'].str[:-2].isin(high).astype(int)
    #
    # # Saving results to a joblib file
    # utils.save_as_joblib(df_chosen, output_file_name, output_dir_path)

    return df_chosen


def rate_spectra(grouped_files, raman_pmba, border_value, margin_of_error, chosen_peak, only_new_spectra,
                 read_from_file=True):
    if read_from_file:
        if not os.path.isfile(output_dir_path + '//' + output_file_name + '.joblib'):
            return main(grouped_files, raman_pmba, border_value, margin_of_error, chosen_peak, only_new_spectra)
        else:
            return utils.read_joblib(output_file_name, output_dir_path)

    else:
        return main(grouped_files, raman_pmba, border_value, margin_of_error, chosen_peak, only_new_spectra)


if __name__ == "__main__":
    from files_preparation import reading_data

    grouped_files = utils.read_joblib(file_name, '../' + dir_path)
    raman_pmba = reading_data.read_spectrum('../data_input/PMBA__powder_10%_1s.txt')

    rated_spectra = main(grouped_files,
                         raman_pmba,
                         chosen_peak='peak2',
                         border_value=10,
                         margin_of_error=0.35,
                         only_new_spectra=True)

    print('done')
