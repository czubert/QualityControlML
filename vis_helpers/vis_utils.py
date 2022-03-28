import pandas as pd
import streamlit as st

RS = 'Raman Shift'
DARK = 'Dark Subtracted #1'


def read_metadata_vis(uploaded_files):
    read_params = {'sep': ';', 'skiprows': lambda x: x > 78, 'decimal': ',', 'index_col': 0,
                   'skipinitialspace': True, 'encoding': "utf-8", 'header': None}
    
    # uploaded_data_read = [pd.read_csv(file, **read_params) for file in uploaded_files]
    uploaded_data_read = []
    
    for file in uploaded_files:
        uploaded_data_read.append(pd.read_csv(file, **read_params))
    
    raw_data = pd.concat(uploaded_data_read)
    return raw_data


def read_spectrum_vis(uploaded_files):
    """
    Reads numeric data from file and returns DataFrame
    :param uploaded_files: String
    :return: DataFrame
    """
    read_params = {'sep': ';', 'skiprows': lambda x: x < 79 or x > 1500, 'decimal': ',',
                   'usecols': ['Raman Shift', DARK],
                   'skipinitialspace': True, 'encoding': "utf-8", 'na_filter': True}
    
    spectra = pd.DataFrame()
    
    for file in uploaded_files:
        tmp_df = (pd.read_csv(file, **read_params))

        if tmp_df.empty:
            continue
        
        tmp_df = tmp_df.rename(columns={DARK: file.name[:-4]})
        tmp_df.dropna(axis=0, how="any")
        tmp_df = tmp_df[tmp_df.iloc[:, 0] > 253]
        tmp_df.set_index('Raman Shift', inplace=True)
    
        spectra = pd.concat([spectra, tmp_df], axis=1)

    spectra.reset_index(inplace=True)
    spectra = spectra.astype({'Raman Shift': 'int'})
    spectra.set_index('Raman Shift', inplace=True)
    spectra_transp = spectra.T

    st.write(spectra_transp)
