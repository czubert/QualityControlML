import streamlit as st
from joblib import load
from vis_helpers import vis_utils
import pandas as pd

RS = 'Raman Shift'
DARK = 'Dark Subtracted #1'

uploaded_files = st.file_uploader('Upload data for Quality Ctrl',
                                  accept_multiple_files=True,
                                  type=['txt', 'csv'])

if uploaded_files:
    for file in uploaded_files:
        file.seek(0)
    
    spectra = vis_utils.read_spectrum_vis(uploaded_files)
    
    model = load(f'data_output/step_5_ml/{"LogisticRegression"}_model.joblib')
    predicted = model.predict(spectra)
    
    result = pd.DataFrame(index=pd.Index(spectra.index, name='my_index'))
    result['Approved'] = predicted
    result.columns.name = result.index.name
    result.index.name = None
    
    st.write(result.to_html(), unsafe_allow_html=True)
    
    st.write(result.index.name)
    st.write(result)
