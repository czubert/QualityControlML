import pandas as pd
import utils

DARK = 'Dark Subtracted #1'


def get_raman_intensities(raman_pmba, peaks):
    # Getting RAMAN spectra of PMBA
    raman_pmba = raman_pmba.reset_index()
    raman_pmba.rename(columns={DARK: "Raman PMBA"}, inplace=True)
    raman_pmba = raman_pmba.set_index('Raman Shift')
    raman_pmba = raman_pmba.T  # transposition of the DF so it fits the ag_df for concat
    utils.change_col_names_type_to_str(raman_pmba)  # changes col names type from int to str, for .loc

    # Getting the value of the peak (max - min values in the range) so-called baseline subtraction,
    subtracted_raman_df = pd.DataFrame()
    for name, values in peaks.items():
        subtracted_raman_df.loc[:, name] = raman_pmba.loc[:, values[0]:values[1]].max(axis=1) \
                                           - raman_pmba.loc[:, values[0]:values[1]].min(axis=1)

    return subtracted_raman_df