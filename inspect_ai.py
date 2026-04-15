import pickle
from tensorflow.keras.models import load_model
import pandas as pd

def inspect_project_files():
    print("=== 🧠 INSPECTING THE AI BRAIN (aqi_lstm_model.keras) ===")
    try:
        model = load_model('aqi_lstm_model.keras')
        # This shows the architecture of the brain
        model.summary()
        print("\n✅ Verification: This file contains the LSTM layers and neural weights.")
    except Exception as e:
        print(f"❌ Could not read model file: {e}")

    print("\n\n=== 📏 INSPECTING THE TRANSLATOR (scaler.pkl) ===")
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # This shows what the translator "remembers"
        print(f"Data Minimum seen during training: {scaler.data_min_}")
        print(f"Data Maximum seen during training: {scaler.data_max_}")
        print(f"Scaling factor: {scaler.scale_}")
        print("\n✅ Verification: This file remembers the scale of your Air Quality data.")
    except Exception as e:
        print(f"❌ Could not read scaler file: {e}")

if __name__ == "__main__":
    inspect_project_files()
