import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from data_preprocessing import load_data, preprocess_data

def build_model(input_shape, output_units):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_units)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(X, y):
    model = build_model(X.shape[1], y.shape[1])
    early_stop = EarlyStopping(patience=20, restore_best_weights=True)
    history = model.fit(X, y, epochs=500, validation_split=0.2, callbacks=[early_stop], verbose=1)
    model.save('model.h5')
    return history

# Add this to execute when the script runs
if __name__ == '__main__':
    # Load and preprocess data
    df = load_data()
    X, y = preprocess_data(df)
    
    # Verify shapes
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    
    # Train and save model
    train_model(X, y)
    print("Model saved as model.h5")