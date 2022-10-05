import glob
import os

import utils
from . import prep_utils

# Output paths
dir_path = 'data_output/step_0_get_file_names'
file_path = 'file_names'

# Input paths
# TODO po skonczeniu pisania kodu zmienic lokalizacje folderow do uczenia
path = 'data_input/data_example/*/*.txt'  # For data with strange long names starting with S28.234...
new_data_url = 'data_input/ML/'
new_ag_url = new_data_url + 'PMBA/*/*.txt'
new_ag_bg_url = new_data_url + 'bg/*/*.txt'


def main():
    file_names = get_names_lower(path)
    new_ag_filenames = get_names_lower(new_ag_url)
    new_ag_bg_filenames = get_names_lower(new_ag_bg_url)
    
    # Splitting filenames into groups ag/au/ag_bg/au_bg
    file_names, not_assigned_spectra_names = separate_by_type_names(file_names)
    
    new_file_names = {'ag': new_ag_filenames, 'ag_bg': new_ag_bg_filenames}
    
    # Merging both groups of names together
    
    for spectra_type in file_names.keys():
        if spectra_type in new_file_names.keys():
            file_names[spectra_type] = file_names[spectra_type] + new_file_names[spectra_type]
    
    utils.save_as_joblib(file_names, file_path, dir_path)
    
    return file_names


def get_names_lower(url):
    names = glob.glob(url)
    return lower_names(names)


def lower_names(file_names):
    """
    Takes list of strings and makes characters lower
    :param file_names: List
    :return: List
    """
    
    return [x.lower() for x in file_names]


def separate_by_type_names(files_names_lower):
    """
    Creates dictionary in which data is separated into 4 groups: ag, au, ag_bg, au_bg,
    each name of a group is a key in the dictionary. List of corresponding file names are values.
    :param files_names_lower: list
    :return: dictionary
    """
    separated_by_type_names = {'ag': [], 'au': [], 'ag_bg': [], 'au_bg': []}
    not_assigned_spectra = []
    
    bg_pattern = r"t[lł][ao]"
    pmba_pattern = r"pmba"
    au_pattern = r"[ g_]au"
    
    au_bg_pattern = r"t[lł][oa][, _](?!(.*(pod|do au))).*(ag)?_?au"
    ag_bg_pattern = r"t[lł][oa][, _](?!(.*(do|pod))).*ag"
    
    for name in files_names_lower:
        agau_in_name = prep_utils.pattern_in_name(name, au_pattern)
        bg_in_name = prep_utils.pattern_in_name(name, bg_pattern)
        pmba_in_name = prep_utils.pattern_in_name(name, pmba_pattern)
        
        # Ag substrates with PMBA analyte
        if not agau_in_name and not bg_in_name and pmba_in_name:
            separated_by_type_names['ag'].append(name)
        
        # Au substrates with PMBA analyte
        elif agau_in_name and not bg_in_name and pmba_in_name:
            separated_by_type_names['au'].append(name)
        
        # Au substrates background
        elif (prep_utils.pattern_in_name(name, au_bg_pattern)) & (not pmba_in_name):
            separated_by_type_names['au_bg'].append(name)
        
        # Ag substrates background
        elif (prep_utils.pattern_in_name(name, ag_bg_pattern)) & (not pmba_in_name):
            separated_by_type_names['ag_bg'].append(name)
        
        # Files with wrong names or different analytes
        else:
            not_assigned_spectra.append(name)
    
    return separated_by_type_names, not_assigned_spectra


def get_names(read_from_file=False):
    if read_from_file:
        if not os.path.isfile(dir_path + '//' + file_path + '.joblib'):
            file_names = main()
        else:
            file_names = utils.read_joblib(file_path, dir_path)
        return file_names
    else:
        return main()
