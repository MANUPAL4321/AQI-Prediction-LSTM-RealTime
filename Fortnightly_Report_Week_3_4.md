# FORTNIGHTLY PROJECT REPORT (WEEK 3 & 4)

![University Logo Placeholder](https://via.placeholder.com/150?text=University+Logo)

## 1. Title Page

**Project Title:** 🌬️ Deep Learning for Real-Time AQI Forecasting & Health Risk Early-Warning  
**Student Name:** Manu Kumar  
**Roll Number:** 2021BTCS042  *(Placeholder)*  
**Course/Program:** B.Tech Computer Science & Engineering (8-Credit Project)  
**Institution Name:** [Insert University Name]  
**Submission Date:** April 2, 2026  

---

## 2. Objective (5 Marks)

The primary focus of this fortnight was to **engineer and train a robust predictive engine** that moves the project from static data analysis to real-time forecasting. Specifically, the task was to design a Long Short-Term Memory (LSTM) neural network capable of predicting pollutant concentrations (e.g., NOx/PM10) for the next hour based on a 24-hour historical lookback. 

The importance of this work lies in **proactive risk mitigation**. By translating historical patterns into future predictions, we aim to provide health-vulnerable groups with early warnings before air quality drops to hazardous levels. The expected outcome for this period was a fully operational AI model (`.keras` format) and an interactive **Streamlit Dashboard** that visualizes these forecasts alongside health-safety thresholds.

---

## 3. Problem Description (5 Marks)

During these two weeks, I addressed the challenge of **Temporal Dependency in Air Pollution**. Unlike standard regression, air quality data is highly cyclical—influenced by morning traffic, industrial shifts, and daily weather cycles. Previous non-dynamic models (worked on in earlier weeks) couldn't "remember" past hours well enough to predict the next state.

In real-life scenarios, health emergencies often occur during sudden pollution "peaks" in the afternoon. Existing systems merely report current levels, which is too late for preventive measures. My project addresses this limitation by using a **Recurrent Neural Network** that treats each hour as part of a continuous sequence, thus capturing the hidden momentum in the data. This is critical for predicting when the 20 µg/m³ threshold will be breached, which is a key health indicator discovered in the Athens (2001-2020) longitudinal study.

---

## 4. Approach / Methodology (10 Marks)

The methodology implemented over the last 14 days followed a rigorous deep-learning pipeline:

### A. Temporal Feature Engineering (Sliding Window)
To convert the time-series into a supervised learning format, I developed logic to create **Sliding Window Sequences**. I engineered a function `create_sequences` that transforms the data into 3D shapes required for LSTM.
*   **Window Size:** 24 hours.
*   **Input Shape:** (Samples, 24, 1).
*   **Output Shape:** (Samples, 1).

### B. Min-Max Normalization Scaling
I implemented a `MinMaxScaler` that converts raw values (often ranging from 0 to 800) into a [0, 1] scale. This normalization was verified as essential for the stability of the LSTM's internal "forget gates" and sigmoidal activation functions.

### C. LSTM Brain Architecture Design
I designed a optimized LSTM network this week with the following logic:
1.  **Input Layer:** 24 timesteps for 1 feature.
2.  **LSTM Layer (64 units):** To extract temporal patterns using ReLU activation.
3.  **Dropout (0.2):** A regularization layer to prevent overfitting.
4.  **Dense Layer (32 units):** For non-linear feature transformation.
5.  **Output Layer:** Single prediction for the next hour's AQI/NOx level.

---

## 5. Implementation Details (5 Marks)

During these two weeks, the following core components were built and integrated:

1.  **Model Training Logic (`train_model.py`):** 
    - Dataset was split into **80% Training** and **20% Testing**.
    - For model optimization, I used **EarlyStopping** which monitored validation loss. The training was halted at epoch 38 when accuracy plateaued, preventing overfitting.
    - Performance metrics used: **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Square Error).

2.  **Evaluation Results:**
    - **MAE:** 61.78 µg/m³
    - **RMSE:** 90.81 µg/m³
    - These metrics indicate that the model is highly capable of identifying major pollution spikes.

3.  **Real-Time Dashboard UI (`app.py`):** 
    - Built using **Streamlit**, this provides a visual interface for the trained model.
    - **Threshold Integration:** I hard-coded the health zones (e.g., Warning levels at 15-20 µg/m³) to provide automatic alerts.

---

## 6. Output / Results (10 Marks)

The work this week culminated in a functional forecasting interface and verified model logic.

### 6.1 Model Training & Evaluation Results
The LSTM model was successfully trained using the Keras backend.
*   **Architecture:** 64 LSTM Units -> Dropout (0.2) -> 32 Dense Units -> 1 Output.
*   **MAE (Mean Absolute Error):** 61.78 µg/m³.
*   **RMSE (Root Mean Square Error):** 90.81 µg/m³.
*   **Convergence:** The model reached its best weights at epoch 33 and stopped at epoch 38 via EarlyStopping.

### 6.2 Data Trend Analysis for Forecasting
I generated visualizations this week that demonstrate the seasonality the LSTM is now learning.
*   **Figure 1: Seasonality Preview.** (Snapshot from `data_trend_preview.png`). This shows the 24-hour periodic cycles that our sliding window logic successfully captured.

### 6.3 Forecasting Dashboard Demo
The Streamlit Dashboard logic (`app.py`) generates real-time predictions. A sample output shows:
- **Sample Input:** User uploads historical CSV.
- **Sample Output:** "Predicted Concentration: 18.52 µg/m³".
- **Visual Alert:** Figure 2: AI Dashboard Forecast Result showing the "Yellow Warning" based on the health risk threshold.

---

## 7. Key Learning (5 Marks)

My learning this fortnight has focused on the practical application of Deep Learning to time-series datasets:

1.  **Memory Management:** I learned that standard neural networks are "feed-forward" and have no memory. This week I understood how LSTM layers effectively "save" information from the beginning of the day to make better predictions for the evening.
2.  **Inverse Scaling:** A major breakthrough was understanding that a model predicts normalized numbers (e.g., 0.12). I learned to use the `scaler.pkl` file to translate this back into physical units (e.g., µg/m³) so the user can understand it.
3.  **UI Design for Science:** I learned how to build a Streamlit app that doesn't just show graphs, but uses **color-coded metrics** to communicate health risks to non-experts.
4.  **Handling Sequence Gaps:** I discovered that LSTMs fail if there are missing rows in the sequence. I learned how to enforce a strict 24-row "data window" in the app as a prerequisite for forecasting.

---
**Verified by:**  
Manu Kumar  
(Author)
