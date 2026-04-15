# 🌬️ AQI Forecasting & Health Impact Analysis
### Deep Learning (LSTM) for PM10 Prediction & Public Health System

This project integrates Deep Learning, Real-Time Data Processing, and Generative AI with Public Health Research to predict Air Quality Index (AQI), with a primary focus on PM10 concentrations.

Inspired by a 20-year longitudinal study in the Greater Athens Area, Greece (2001–2020), this system goes beyond traditional analysis by enabling:

Time-series AQI prediction using LSTM
Real-time pollution monitoring
AI-powered health recommendations

The goal is to transform raw environmental data into actionable insights for public health awareness and decision-making.

---

## 🔬 Research Context: The Athens Study (2001-2020)
This project is built upon the findings of a comprehensive study that investigated the impact of long-term PM10 exposure on human health.

### The WHO AirQ+ Model
The study applies the **WHO AirQ+ model**, which estimates health impacts by combining:
1.  **PM10 Data:** Concentration levels across different monitoring stations.
2.  **Population Data:** Demographic information of the affected area.
3.  **Relative Risk (RR):** Scaling factors that link pollution levels to the likelihood of disease.

### Key Study Findings
*   **Strong Correlation:** There is a clear link between PM10 levels and increased hospital admissions for respiratory issues.
*   **Vulnerable Groups:** **Children** are significantly more affected than adults, showing higher prevalence of bronchitis even at lower exposure levels.
*   **Critical Thresholds:** Notable health risks occur at relatively low concentration levels (**around 15–20 µg/m³**).
*   **Long-term Trends:** Pollution levels and health impacts showed a marked decrease after 2010.
*   **Spatial Variation:** Central Athens continues to show higher pollution and disease rates than peripheral areas.

---

## 🚀 Application Features
*   **🔹 Data Preprocessing & Cleaning**
        Handles missing values (-200 → NaN)
        Interpolation for continuous time-series data
        European to standard format conversion
        Datetime merging and ordering
    
*   **🔹 Temporal Feature Engineering**
        Sliding window approach (24-hour lookback)
        Converts raw data into LSTM-compatible sequences
        Captures temporal trends and daily pollution patterns
    
*   **🔹 LSTM-Based AQI Forecasting**
        Built using TensorFlow/Keras
        Learns time-dependent pollution behavior
        Predicts future AQI values (up to 24 hours ahead)
    
*   **🔹 Real-Time AQI Integration**
        Fetches live data using AQICN API
        Supports pollutants:
        PM2.5, PM10
        NO₂, CO
        Temperature & Humidity

*   **🔹 Interactive Visualization Dashboard**
        Built using Streamlit
        **Features:**
            CSV upload for prediction
            Automatic preprocessing
            Real-time charts and forecasts

*   **🔹 Health Risk Indicators**
        Based on standard AQI thresholds (US EPA)
        Color-coded categories:
            🟢 Good
            🟡 Moderate
            🔴 Unhealthy
            🟣 Hazardous
        Special alerts for:
            Children
            Elderly
            Sensitive individuals

*   **🔹 AI-Powered Health Advisory System**
        Integrated with Google Gemini (LLM)
        Generates:
        Health insights
        Risk explanations
        Personalized recommendations
---

*   **🧠 System Workflow**
      * Raw Dataset Input
      * Data Cleaning & Preprocessing
      * Feature Engineering (Sliding Window)
      * LSTM Model Training
      * AQI Prediction
      * Real-Time Data Integration
      * Health Risk Classification
      * AI-Based Health Advisory


## 🛠️ Technology Stack
*   **Deep Learning:** TensorFLow / Keras (LSTM)
*   **Web Framework:** Streamlit
*   **Data Science:** Pandas, NumPy, Scikit-learn
*   **Visuals:** Plotly / Matplotlib
*   **API Integration:** AQICN API
*   **Generative AI:** Google Gemini API

---

**📊 Model Details**
Model Type: LSTM (Recurrent Neural Network)
Input: 24-hour sequence data
Output: AQI / PM10 prediction
Scaling: MinMaxScaler
Regularization: Dropout (20%)
Evaluation Metrics: MAE, RMSE


## 📦 Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```
---

**📌 Use Case**

This system can be used for:

Air quality monitoring systems
Public health awareness platforms
Smart city applications
Environmental research

---

