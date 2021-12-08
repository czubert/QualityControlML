import numpy as np
import os

import utils


# Output paths
dir_path = 'data_output/step_2_group_data'
file_name = 'grouped_data'

def main(grouped_files, baseline_corr=False):
    pmba = estimate_pmba_enhancement(pmba)  # Estimating quality based on quartiles
    bg = assign_quality_to_bg2(bg, pmba)  # Setting bg quality based on pmba quality








def rate_spectra(grouped_files, read_from_file=True, baseline_corr=False):
    if read_from_file:
        if not os.path.isfile(dir_path + '//' + file_name + '.joblib'):
            rated_spectra = main(grouped_files, baseline_corr=False)
        else:
            rated_spectra = utils.read_joblib(file_name, dir_path)
        return rated_spectra
    else:
        return main(grouped_files, baseline_corr=False)
