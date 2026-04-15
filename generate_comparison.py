import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison():
    # --- 1. SET UP THE RAW DATA (BEFORE) ---
    raw_df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',', low_memory=False)
    raw_df = raw_df.iloc[:500, :] # Take first 500 hours for clarity
    
    # Plotting Raw NOx data (with -200 spikes)
    plt.figure(figsize=(12, 5))
    plt.plot(raw_df['NOx(GT)'].values, color='red', alpha=0.7)
    plt.title('BEFORE: Raw Sensor Data (Showing massive -200 error spikes)', fontsize=14, color='darkred')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Concentration')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('before_cleaning_spikes.png')
    print(" 'before_cleaning_spikes.png' created.")

    # --- 2. SET UP THE CLEANED DATA (AFTER) ---
    clean_df = pd.read_csv('cleaned_air_quality.csv', index_col=0)
    clean_df = clean_df.iloc[:500, :] # Take same first 500 hours
    
    plt.figure(figsize=(12, 5))
    plt.plot(clean_df['NOx(GT)'].values, color='green', alpha=0.8)
    plt.title('AFTER: Cleaned Data (Gaps filled with Smart Interpolation)', fontsize=14, color='darkgreen')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Concentration')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('after_cleaning_smooth.png')
    print(" 'after_cleaning_smooth.png' created.")

if __name__ == "__main__":
    generate_comparison()
