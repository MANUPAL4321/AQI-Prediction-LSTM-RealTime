import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocess_aqi_data(file_path):
    print(f"--- Loading Dataset: {file_path} ---")
    
    # 1. Load data
    # The UCI dataset uses ';' as a separator and ',' as a decimal point.
    df = pd.read_csv(file_path, sep=';', decimal=',', low_memory=False)
    
    # 2. Clean up empty rows/columns
    # The UCI CSV has extra empty columns and rows at the end.
    df = df.iloc[:, :-2] # Drop two empty columns at the end
    df = df.dropna(how='all') # Drop rows that are completely empty
    
    print(f"Initial shape: {df.shape}")
    
    # 3. Handle the "-200" problem (Missing Data)
    # Replace -200 with NaN (Not a Number) so pandas knows they are missing
    df.replace(-200, np.nan, inplace=True)
    
    # 4. Handle Date and Time
    # Combine Date and Time columns into one 'Datetime' column
    # Date format is DD/MM/YYYY, Time format is HH.MM.SS
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
    df.set_index('Datetime', inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # 5. Fix Missing Values using Linear Interpolation
    # This fills the -200 gaps by looking at the values before and after.
    df.interpolate(method='linear', limit_direction='forward', inplace=True)
    
    # In case there are missing values at the very start/end that couldn't be interpolated
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    print("--- Pre-processing Complete! ---")
    print(f"Final shape: {df.shape}")
    print("\nCleaned Column Samples:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    filepath = 'AirQualityUCI.csv'
    cleaned_df = preprocess_aqi_data(filepath)
    
    # Save the cleaned data for our LSTM model
    cleaned_df.to_csv('cleaned_air_quality.csv')
    print("\n Cleaned data saved to 'cleaned_air_quality.csv'")
    
    # Quickly visualize one pollutant (e.g., NOx)
    plt.figure(figsize=(12, 6))
    plt.plot(cleaned_df['NOx(GT)'].head(500), label='NOx(GT) Concentration')
    plt.title('Air Quality Trend (First 500 Hours)')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    plt.savefig('data_trend_preview.png')
    print(" Trend preview saved as 'data_trend_preview.png'")
