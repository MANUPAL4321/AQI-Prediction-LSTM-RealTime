# FORTNIGHTLY PROJECT REPORT (WEEK 1 & 2)

![University Logo Placeholder](https://via.placeholder.com/150?text=University+Logo)

## 1. Title Page

**Project Title:** 🌬️ Air Quality Index (AQI) Forecasting: Preprocessing & Deep Learning Design Phase  
**Student Name:** Manu Kumar  
**Roll Number:** 2021BTCS042  *(Placeholder)*  
**Course/Program:** B.Tech Computer Science & Engineering (8-Credit Project)  
**Institution Name:** [Insert University Name]  
**Submission Date:** March 23, 2026  

---

## 2. Objective (5 Marks)

The primary objective for these first two weeks was to establish a **complete end-to-end data pipeline**, moving from raw sensor acquisition to the architectural design of a forecasting engine. This involved two critical phases: 
1.  **Quality Control:** Cleansing the 9,357-hour UCI Air Quality dataset and reconstructing missing sensor values.
2.  **Structural Engineering:** Designing a temporal data structure (Sliding Windows) and a neural network architecture capable of capturing long-term environmental trends.

The expected outcome for this phase was a **ready-to-train ecosystem**. This includes a fully processed CSV dataset, a fitted normalization scaler, and a finalized LSTM (Long Short-Term Memory) model blueprint. By completing these advanced steps in the first fortnight, we ensured a smooth transition into the high-intensity training and evaluation phase in the following weeks.

---

## 3. Dataset & Problem Description (5 Marks)

### 3.1 Dataset Overview
The project utilizes the **UCI Air Quality Dataset**, which consists of 9,358 instances of hourly averaged responses from a chemical multisensor device deployed in a significantly polluted area in Italy.

### 3.2 Feature Identification (The "Data Dictionary")
| Feature Name | Description | Units | Type |
|---|---|---|---|
| **CO(GT)** | Carbon Monoxide Concentration | mg/m³ | Ground Truth (GT) |
| **PT08.S1(CO)** | CO Sensor Response (Tin Oxide) | OHM (Nominal) | Raw Sensor |
| **C6H6(GT)** | Benzene Concentration | µg/m³ | Ground Truth (GT) |
| **NOx(GT)** | Nitrogen Oxides Concentration | ppb | Ground Truth (GT) |
| **NO2(GT)** | Nitrogen Dioxide Concentration | µg/m³ | Ground Truth (GT) |
| **T** | Temperature | °C | Environment |
| **RH** | Relative Humidity | % | Environment |
| **AH** | Absolute Humidity | g/m³ | Environment |

*Note: Features labeled `GT` (Ground Truth) are established via high-precision analyzers, while `PT08` features represent raw sensor readings. Our LSTM model primarily focuses on predicting the `GT` values to ensure high accuracy.*

---

## 4. Approach / Methodology (10 Marks)

A multi-layered methodology was executed during this fortnight to ensure data readiness:

### A. Non-Linear Data Cleaning & Interpolation
Instead of dropping missing values, I implemented **Linear Interpolation**. This uses the mathematical average of neighboring sensor readings to accurately simulate the atmosphere during sensor downtime. For instance, if data is missing at 11 AM, the system calculates the mid-point between 10 AM and 12 PM.

### B. Min-Max Normalization Scaling
To ensure the AI trains efficiently, I implemented a **MinMaxScaler**. This scales all pollutant levels into a uniform **[0, 1]** range. 
$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
This was a critical step in the first fortnight to prevent the LSTM's activation functions from saturating.

### C. 24-Hour Sliding Window Sequences
I engineered a **Feature Transformation** logic that converts flat CSV data into a 3D Tensor format. I established a `create_sequences` function that takes a 24-hour "lookback" period. This allows the model to "see" a full day's history before predicting the next hour's AQI.

---

## 5. Implementation Details (5 Marks)

The project foundation was implemented across two core modules:

1.  **Preprocessing Module (`preprocess_data.py`):** Handles semi-colon parsing, decimal interpretation, and implements the `.interpolate()` logic.
2.  **Structuring Module (`train_model.py`):** Defines the `create_sequences` function and initializes the **Sequential Keras Model** layout.

---

## 6. Output / Results (10 Marks)

### 6.1 Visual Verification of Data Cleaning
The following figures demonstrate our progress in data reconstruction:

![Before Cleaning Graph](file:///home/manu-kumar/Documents/8%20Credit%20project/Air%20quality%20index/before_cleaning_spikes.png)
**Figure 1: NOx Concentration with Raw Sensor Spikes (Before Cleaning).** 
*This graph displays dangerous "Negative Spikes" reaching -200. These are not real pollution readings; they represent periods where the sensor was offline or rebooting. Training an AI on this data would lead to impossible predictions.*

![After Cleaning Graph](file:///home/manu-kumar/Documents/8%20Credit%20project/Air%20quality%20index/after_cleaning_smooth.png)
**Figure 2: NOx Concentration after Linear Interpolation (After Preprocessing).** 
*This graph shows the successful reconstruction of the timeline. The erroneous -200 spikes have been replaced with smooth, logical transitions. The dataset is now a "Continuous Time-Series," which is a mandatory requirement for LSTM Neural Networks.*

---

## 7. Key Learning (5 Marks)

This intensive two-week phase yielded several core learnings:

1.  **Imputation Logic:** I learned that deleting rows (dropping NaNs) is bad for time-series. **Interpolation preserves the sequence**, which is critical for forecasting.
2.  **Gradient Stability:** I discovered that LSTMs are sensitive to large inputs; scaling data to [0,1] is the key to training stability.
3.  **Domain Knowledge:** I learned the difference between **Raw Sensor Output** and **Ground Truth (GT)** concentrations. My project focuses on GT values for research-grade accuracy.

---
**Verified by:**  
Manu Kumar  
(Author)
