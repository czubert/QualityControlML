import io
import streamlit as st
import pandas as pd
from detect_delimiter import detect
from bs4 import UnicodeDammit
from constants import LABELS
from vis_helpers import vis_utils
from files_preparation import reading_data

RS = 'Raman Shift'
DARK = 'Dark Subtracted #1'

uploaded_files = st.file_uploader('Upload data for Quality Ctrl',
                                  accept_multiple_files=True,
                                  type=['txt', 'csv'])

if uploaded_files:
    for file in uploaded_files:
        file.seek(0)
    
    # a = vis_utils.read_metadata_vis(uploaded_files)
    b = vis_utils.read_spectrum_vis(uploaded_files)
    
    # st.write(a)
    st.write(b)
