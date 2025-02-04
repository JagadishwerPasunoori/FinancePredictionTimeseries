import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load artifacts
year_scaler = joblib.load('scalers/year_scaler.pkl')
feature_scalers = joblib.load('scalers/feature_scalers.pkl')
feature_names = joblib.load('scalers/feature_names.pkl')
model = load_model('model.h5')

st.title('Financial Metrics Predictor App')

# User inputs
year = st.number_input('Year', min_value=1900, max_value=2100, value=2023)
selected_feature = st.selectbox('Select Feature to Input', feature_names)
feature_value = st.number_input(f'Enter Value for {selected_feature}')

if st.button('Predict'):
    # Prepare input vector
    scaled_year = year_scaler.transform([[year]])[0][0]
    feat_index = feature_names.index(selected_feature)
    
    # Scale feature value
    feat_scaler = feature_scalers[selected_feature]
    scaled_feat_value = feat_scaler.transform([[feature_value]])[0][0]
    
    # Create one-hot encoding
    one_hot = np.zeros(len(feature_names))
    one_hot[feat_index] = 1
    
    # Build input vector
    input_vector = np.concatenate([[scaled_year], one_hot, [scaled_feat_value]]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_vector)[0]
    
    # Reconstruct full feature vector
    full_prediction = np.zeros(len(feature_names))
    full_prediction[feat_index] = scaled_feat_value
    output_indices = [i for i in range(len(feature_names)) if i != feat_index]
    full_prediction[output_indices] = prediction
    
    # Inverse transform predictions
    results = {}
    for i, feat in enumerate(feature_names):
        scaler = feature_scalers[feat]
        results[feat] = scaler.inverse_transform([[full_prediction[i]]])[0][0]
    
    # Display results in table
    st.subheader('Predicted Values')
    
    # Create DataFrame and format
    results_df = pd.DataFrame(
        [(feat, f"${val:,.2f}B") if '($B)' in feat else (feat, f"{val:.2f}%") if '%' in feat else (feat, f"{val:.2f}") 
         for feat, val in results.items()],
        columns=['Metric', 'Predicted Value']
    )
    
    # Display styled table
    st.table(results_df.style.set_properties(**{
        'background-color': '#f0f2f6',
        'color': 'black',
        'border': '1px solid white'
    }))