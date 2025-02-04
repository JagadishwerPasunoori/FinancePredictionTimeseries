import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import os

def load_data(filepath='Data1.csv'):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Create directory for scalers if not exists
    os.makedirs('scalers', exist_ok=True)
    
    # Separate year and features
    years = df['Year'].values.reshape(-1, 1)
    features_df = df.drop('Year', axis=1)
    feature_names = features_df.columns.tolist()
    
    # Initialize scalers
    year_scaler = MinMaxScaler()
    scaled_years = year_scaler.fit_transform(years)
    
    feature_scalers = {}
    scaled_features = np.zeros_like(features_df.values)
    for i, col in enumerate(feature_names):
        scaler = MinMaxScaler()
        scaled_features[:, i] = scaler.fit_transform(features_df[[col]]).flatten()
        feature_scalers[col] = scaler
    
    # Create training samples
    X, y = [], []
    for idx in range(len(df)):
        current_year = scaled_years[idx][0]
        current_features = scaled_features[idx]
        
        for feat_idx, feat_name in enumerate(feature_names):
            # Create one-hot encoded feature
            one_hot = np.zeros(len(feature_names))
            one_hot[feat_idx] = 1
            
            # Prepare input vector
            input_vec = np.concatenate([[current_year], one_hot, [current_features[feat_idx]]])
            
            # Prepare output vector
            output_mask = np.ones(len(feature_names), dtype=bool)
            output_mask[feat_idx] = False
            output_features = current_features[output_mask]
            
            X.append(input_vec)
            y.append(output_features)
    
    # Save scalers and feature names
    joblib.dump(year_scaler, 'scalers/year_scaler.pkl')
    joblib.dump(feature_scalers, 'scalers/feature_scalers.pkl')
    joblib.dump(feature_names, 'scalers/feature_names.pkl')
    
    return np.array(X), np.array(y)

# Add this to actually execute when the script runs
if __name__ == '__main__':
    # Load data - make sure you have a 'data.csv' file in your directory
    df = load_data()
    
    # Run preprocessing
    X, y = preprocess_data(df)
    
    # Verify creation
    print(f"Created training data with shape: {X.shape}")
    print("Scalers saved in ./scalers directory:")
    print(os.listdir('scalers'))