import streamlit as st
from joblib import load
from vis_helpers import vis_utils
import pandas as pd
import glob

RS = 'Raman Shift'
DARK = 'Dark Subtracted #1'
MODEL_PATH = 'data_output/step_5_ml/models'

uploaded_files = st.sidebar.file_uploader('Upload data for Quality Ctrl',
                                          accept_multiple_files=True,
                                          type=['txt', 'csv'])

trained_models_names = glob.glob(f"{MODEL_PATH}/*")

st.subheader('SERSitive QualityControl')

if uploaded_files:
    for file in uploaded_files:
        file.seek(0)
    
    spectra = vis_utils.read_spectrum_vis(uploaded_files)
    
    result = pd.DataFrame(index=pd.Index(spectra.index, name='File name'))
    
    for trained_model_name in trained_models_names:
        trained_model_name = trained_model_name.split('\\')[1]
        trained_model_short_name = trained_model_name.replace('Classifier', '')
        trained_model_short_name = trained_model_short_name.replace('LogisticRegression', 'LR*')
        trained_model = load(f'data_output/step_5_ml/models/{trained_model_name}')
        predicted = trained_model.predict(spectra)
        result[trained_model_short_name.split('_')[0]] = predicted
        
    # Calculating median and mean of predictions of all estimators
    median = result.median(axis=1, skipna=True)
    mean = result.mean(axis=1, skipna=True)
    result['Median'] = median
    result['Mean'] = mean
    
    st.write('Predictions')
    st.write(result)
    st.write('*LogisticRegression')
    mean = pd.DataFrame()
    median = pd.DataFrame()
    median['Median'] = result.Median.value_counts()
    mean['Mean'] = result.Mean.value_counts()
    results = pd.concat([mean, median], axis=1)
    st.write('Summary')
    st.write(results)
