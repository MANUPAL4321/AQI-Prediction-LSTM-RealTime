# 🌬️ AQI Forecasting & Health Impact Analysis
### Deep Learning (LSTM) for PM10 Prediction & Public Health

This project integrates **Deep Learning** with **Public Health Research** to predict Air Quality Index (AQI) values, specifically focusing on PM10 concentrations. Inspired by a 20-year longitudinal study in the Greater Athens Area, Greece (2001–2020), this tool allows users to upload time-series data and receive accurate forecasts to help mitigate respiratory health risks.

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
*   **User Data Upload:** Drag-and-drop CSV interface for time-series AQI/PM10 data.
*   **LSTM Forecasting:** High-accuracy predictions using Long Short-Term Memory neural networks.
*   **Interactive Visualization:** Real-time charting of historical trends and health-risk forecasts.
*   **Health Risk Indicators:** Automatic detection of threshold breaches (e.g., predicted zones above 20 µg/m³).

---

## 🛠️ Technology Stack
*   **Deep Learning:** TensorFLow / Keras (LSTM)
*   **Web Framework:** Streamlit
*   **Data Science:** Pandas, NumPy, Scikit-learn
*   **Visuals:** Plotly / Matplotlib

---

## 📦 Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```
