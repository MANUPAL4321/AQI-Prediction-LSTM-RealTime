import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# --- Step 3-4 Logic (Sequence Creation) ---
def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# --- Step 5 & 6: Model Design & Training ---
def build_and_train_model(csv_path, target_col='NOx(GT)', seq_length=24):
    print(f"--- Loading Cleaned Data: {csv_path} ---")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' not found! Run 'preprocess_data.py' first.")
    
    df = pd.read_csv(csv_path, index_col=0)
    data = df[[target_col]].values 
    
    # Normalization (MinMaxScaler)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Save the scaler early
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create Sequences
    X, y = create_sequences(scaled_data, seq_length)
    
    # Step 6: 80/20 Split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Data split complete: {len(X_train)} training samples, {len(X_test)} test samples.")

    # Step 5: Build LSTM Model Architecture
    # Input Layer -> LSTM Layer -> Dropout Layer -> Dense Layer -> Output Layer
    model = Sequential([
        Input(shape=(seq_length, 1)),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("\n---  Step 5: Model Architecture ---")
    model.summary()

    # Step 6: Train the Model
    print("\n---  Step 6: Training the Model (with Early Stopping) ---")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Step 6: Evaluate Model (MAE & RMSE)
    print("\n---  Evaluation Results ---")
    predictions_scaled = model.predict(X_test)
    
    # Inverse transform to get real concentrations
    y_test_real = scaler.inverse_transform(y_test)
    predictions_real = scaler.inverse_transform(predictions_scaled)
    
    mae = mean_absolute_error(y_test_real, predictions_real)
    mse = mean_squared_error(y_test_real, predictions_real)
    rmse = np.sqrt(mse)
    
    print(f"Mean Absolute Error (MAE): {mae:.2f} µg/m³")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f} µg/m³")
    
    # Save the final model
    model.save('aqi_lstm_model.keras')
    print("\n AI Braiin saved as 'aqi_lstm_model.keras'")
    
    return model, history

if __name__ == "__main__":
    try:
        build_and_train_model('cleaned_air_quality.csv')
    except Exception as e:
        print(f" Error: {e}")
