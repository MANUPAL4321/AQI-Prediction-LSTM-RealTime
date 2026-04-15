import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import requests
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model

# ══════════════════════════════════════════════
# CONSTANTS & CONFIG
# ══════════════════════════════════════════════
HISTORY_FILE = 'prediction_history.csv'
AQICN_API_KEY = 'ad951f808244f6a47f5205e1e2f5972a22f8647e'  # Dedicated key provided by user

# Indian state → major city mapping (AQICN needs city names, not states)
INDIA_STATE_MAP = {
    'bihar': 'patna', 'gujarat': 'ahmedabad', 'gujrat': 'ahmedabad', 'rajasthan': 'jaipur',
    'maharashtra': 'mumbai', 'karnataka': 'bengaluru', 'tamil nadu': 'chennai',
    'telangana': 'hyderabad', 'west bengal': 'kolkata', 'uttar pradesh': 'lucknow',
    'madhya pradesh': 'bhopal', 'punjab': 'ludhiana', 'haryana': 'gurugram',
    'odisha': 'bhubaneswar', 'kerala': 'kochi', 'assam': 'guwahati',
    'jharkhand': 'ranchi', 'chhattisgarh': 'raipur', 'uttarakhand': 'dehradun',
    'goa': 'goa', 'andhra pradesh': 'visakhapatnam', 'up': 'lucknow',
    'mp': 'bhopal', 'wb': 'kolkata', 'hp': 'shimla', 'j&k': 'srinagar',
    'bengaluru': 'bengaluru', 'bangalore': 'bengaluru',
    'bombay': 'mumbai', 'calcutta': 'kolkata', 'madras': 'chennai',
}

# Try importing Gemini for LLM integration
GEMINI_AVAILABLE = False
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    pass

st.set_page_config(
    page_title="AirAI · Intelligent AQI System",
    layout="wide",
    page_icon="🌬️",
    initial_sidebar_state="expanded"
)


# ══════════════════════════════════════════════
# PREMIUM CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*,*::before,*::after{font-family:'Inter',sans-serif!important}

/* Background */
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#0a0e1a!important;color:#e2e8f0!important}
[data-testid="stHeader"]{background:transparent!important}

/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f1629,#131b33)!important;border-right:1px solid rgba(99,102,241,.15)!important}
[data-testid="stSidebar"] *,[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] div{color:#cbd5e1!important}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1{background:linear-gradient(135deg,#818cf8,#6366f1,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:900!important}

/* Metrics */
[data-testid="stMetric"]{background:rgba(15,23,42,.7)!important;backdrop-filter:blur(16px)!important;border:1px solid rgba(99,102,241,.15)!important;border-radius:16px!important;padding:20px 24px!important;transition:all .3s ease!important}
[data-testid="stMetric"]:hover{border-color:rgba(99,102,241,.4)!important;transform:translateY(-2px);box-shadow:0 8px 24px rgba(99,102,241,.1)!important}
[data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:.8rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.8px!important}
[data-testid="stMetricValue"]{color:#e0e7ff!important;font-weight:800!important;font-size:1.5rem!important}

/* Buttons */
.stButton>button{background:linear-gradient(135deg,#6366f1,#818cf8)!important;color:#fff!important;border:none!important;border-radius:14px!important;padding:12px 24px!important;font-weight:700!important;transition:all .3s ease!important;box-shadow:0 4px 16px rgba(99,102,241,.25)!important}
.stButton>button:hover{transform:translateY(-2px)scale(1.02)!important;box-shadow:0 8px 32px rgba(99,102,241,.4)!important}

/* File uploader */
[data-testid="stFileUploader"]{background:rgba(15,23,42,.5)!important;border:2px dashed rgba(99,102,241,.25)!important;border-radius:16px!important;padding:16px!important}

/* Dividers */
hr{border-color:rgba(99,102,241,.1)!important;margin:24px 0!important}

/* Alerts */
[data-testid="stAlert"]{border-radius:14px!important}

/* Dataframe */
[data-testid="stDataFrame"]{border-radius:14px!important;overflow:hidden!important}

/* Text input */
.stTextInput>div>div>input{background:rgba(15,23,42,.6)!important;border:1px solid rgba(99,102,241,.2)!important;border-radius:12px!important;color:#e0e7ff!important}

/* Custom classes */
.hero{margin-bottom:28px}
.hero-title{font-size:2.4rem;font-weight:900;letter-spacing:-1.5px;line-height:1.1;background:linear-gradient(135deg,#e0e7ff,#a5b4fc 30%,#818cf8 60%,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px}
.hero-sub{font-size:1rem;color:#64748b;letter-spacing:.3px}
.sec-h{font-size:1.2rem;font-weight:700;color:#c7d2fe;margin-bottom:14px;display:flex;align-items:center;gap:10px}
.sec-h::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(99,102,241,.3),transparent)}
.gcard{background:rgba(15,23,42,.6);backdrop-filter:blur(20px);border:1px solid rgba(99,102,241,.12);border-radius:18px;padding:24px;margin-bottom:16px;transition:all .3s ease}
.gcard:hover{border-color:rgba(99,102,241,.3);box-shadow:0 8px 32px rgba(99,102,241,.08)}
.pill{display:inline-flex;align-items:center;gap:6px;background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.12);border-radius:30px;padding:6px 16px;margin:3px;font-size:.82rem;color:#a5b4fc;font-weight:500}
.rcard{background:rgba(15,23,42,.65);backdrop-filter:blur(16px);border-radius:18px;padding:24px;margin-top:12px;position:relative;overflow:hidden}
.rcard::before{content:'';position:absolute;top:0;left:0;width:5px;height:100%}
.rcard.good{border:1px solid rgba(52,211,153,.25)}.rcard.good::before{background:linear-gradient(180deg,#34d399,#10b981)}
.rcard.mod{border:1px solid rgba(251,191,36,.25)}.rcard.mod::before{background:linear-gradient(180deg,#fbbf24,#f59e0b)}
.rcard.usg{border:1px solid rgba(251,146,60,.25)}.rcard.usg::before{background:linear-gradient(180deg,#fb923c,#f97316)}
.rcard.unh{border:1px solid rgba(239,68,68,.25)}.rcard.unh::before{background:linear-gradient(180deg,#ef4444,#dc2626)}
.rcard.haz{border:1px solid rgba(168,85,247,.25)}.rcard.haz::before{background:linear-gradient(180deg,#a855f7,#9333ea)}
.ibox{background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.15);border-radius:14px;padding:18px 22px;margin-top:12px;color:#c7d2fe;font-size:.9rem;line-height:1.7}
.empty{text-align:center;padding:60px 40px;border:2px dashed rgba(99,102,241,.2);border-radius:24px;background:rgba(15,23,42,.4);margin:16px 0}
.empty h2{color:#818cf8;font-weight:700;font-size:1.4rem;margin-bottom:6px}
.empty p{color:#64748b;font-size:.95rem}
.rstat{background:rgba(15,23,42,.6);border:1px solid rgba(99,102,241,.12);border-radius:16px;padding:20px;text-align:center;transition:all .3s ease}
.rstat:hover{border-color:rgba(99,102,241,.35)}
.rstat .bn{font-size:2rem;font-weight:900;background:linear-gradient(135deg,#818cf8,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.rstat .sl{font-size:.78rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-top:4px}
.llm-box{background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(168,85,247,.06));border:1px solid rgba(99,102,241,.2);border-radius:18px;padding:24px;margin-top:12px;color:#c7d2fe;line-height:1.8;font-size:.92rem}
.llm-box .llm-header{display:flex;align-items:center;gap:10px;margin-bottom:14px;font-weight:700;font-size:1.05rem;color:#a5b4fc}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.gcard,[data-testid="stMetric"],.rcard{animation:fadeUp .5s ease-out forwards}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0a0e1a}::-webkit-scrollbar-thumb{background:#1e293b;border-radius:6px}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════
@st.cache_resource
def load_forecast_engine():
    try:
        model = load_model('aqi_lstm_model.keras')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        return None, None

def calculate_multi_forecast(model, scaler, sequence, steps=24):
    """Recursive LSTM forecasting for t+1 to t+steps."""
    current_seq = sequence.copy()
    predictions = []
    for _ in range(steps):
        input_seq = current_seq.reshape(1, 24, 1)
        pred_scaled = model.predict(input_seq, verbose=0)
        current_seq = np.append(current_seq[1:], pred_scaled)
        predictions.append(pred_scaled[0][0])
    all_preds = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return all_preds[0], all_preds[-1], all_preds

def auto_clean_dataset(df):
    """Auto-detect format, clean, and prepare any uploaded CSV."""
    log = []
    # Drop empties
    empty_r = df.isna().all(axis=1).sum()
    ec = df.shape[1]
    df = df.dropna(how='all').dropna(axis=1, how='all')
    ecd = ec - df.shape[1]
    if empty_r > 0: log.append(f"Removed {int(empty_r)} empty rows")
    if ecd > 0: log.append(f"Removed {ecd} empty columns")
    # Parse date/time
    if 'Date' in df.columns and 'Time' in df.columns:
        try:
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
            df.set_index('Datetime', inplace=True)
            df.drop(['Date', 'Time'], axis=1, inplace=True)
            log.append("Parsed Date & Time into Datetime index")
        except Exception:
            try:
                df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
                df.set_index('Datetime', inplace=True)
                df.drop(['Date', 'Time'], axis=1, inplace=True)
                log.append("Parsed Date & Time (auto-format)")
            except Exception:
                log.append("⚠️ Could not parse Date/Time")
    # Find target column
    target = None
    for pc in ['NOx(GT)', 'PT08.S3(NOx)', 'PT08.S1(CO)', 'C6H6(GT)', 'CO(GT)']:
        if pc in df.columns:
            target = pc
            break
    if target is None:
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        target = nc[0] if nc else None
    if target is None:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            if df[col].notna().sum() > 0:
                target = col
                break
    if target is None:
        return df, None, log
    # Ensure numeric
    if df[target].dtype == 'object':
        df[target] = pd.to_numeric(df[target].astype(str).str.replace(',', '.'), errors='coerce')
        log.append(f"Converted '{target}' to numeric")
    # Replace -200 sentinels
    s = (df[target] == -200).sum()
    if s > 0:
        df[target] = df[target].replace(-200, np.nan)
        log.append(f"Replaced {int(s)} sentinel values (-200)")
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].replace(-200, np.nan)
    # Negative values
    neg = (df[target] < 0).sum()
    if neg > 0:
        df.loc[df[target] < 0, target] = np.nan
        log.append(f"Removed {int(neg)} negative values")
    # Duplicates
    dup = df.duplicated().sum()
    if dup > 0:
        df = df.drop_duplicates()
        log.append(f"Removed {int(dup)} duplicates")
    # Interpolate
    nan_c = df[target].isna().sum()
    if nan_c > 0:
        df[target] = df[target].interpolate(method='linear', limit_direction='both').ffill().bfill()
        log.append(f"Interpolated {int(nan_c)} missing values")
    df = df.dropna(subset=[target])
    return df, target, log


# ── AQI Classification (Standard US EPA breakpoints) ──
def classify_aqi(value):
    """Map AQI/concentration to health category with full metadata."""
    if value <= 50:
        return {"label": "Good", "emoji": "🟢", "css": "good", "color": "#34d399", "level": 1,
                "child": "Air is safe. Normal outdoor play is encouraged for all children.",
                "adult": "No health risk. All outdoor activities are safe.",
                "advice": "Enjoy outdoor activities freely.",
                "health_msg": "Air quality is considered satisfactory, and air pollution poses little or no risk."}
    elif value <= 100:
        return {"label": "Moderate", "emoji": "🟡", "css": "mod", "color": "#fbbf24", "level": 2,
                "child": "Generally safe. Children with asthma should carry inhalers as a precaution.",
                "adult": "Acceptable quality. Unusually sensitive individuals should consider reducing prolonged outdoor exertion.",
                "advice": "Sensitive groups should limit extended outdoor exposure.",
                "health_msg": "Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people."}
    elif value <= 150:
        return {"label": "Unhealthy (Sensitive)", "emoji": "🟠", "css": "usg", "color": "#fb923c", "level": 3,
                "child": "⚠️ Children should reduce outdoor play. Higher risk of respiratory symptoms.",
                "adult": "Sensitive individuals (elderly, respiratory/heart conditions) should limit outdoor exertion.",
                "advice": "Reduce prolonged outdoor exertion. Close windows if possible.",
                "health_msg": "Members of sensitive groups may experience health effects. The general public is not likely to be affected."}
    elif value <= 200:
        return {"label": "Unhealthy", "emoji": "🔴", "css": "unh", "color": "#ef4444", "level": 4,
                "child": "🚨 Keep children indoors whenever possible. Risk of bronchitis and respiratory distress.",
                "adult": "Everyone should reduce outdoor activities. People with respiratory conditions must stay indoors.",
                "advice": "Avoid outdoor activities. Use air purifiers indoors.",
                "health_msg": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects."}
    else:
        return {"label": "Hazardous", "emoji": "🟣", "css": "haz", "color": "#a855f7", "level": 5,
                "child": "🚨 CRITICAL – All children must remain indoors. Very high risk of acute respiratory distress.",
                "adult": "🚨 EMERGENCY – Avoid ALL outdoor activity. Everyone is at risk of serious health effects.",
                "advice": "Stay indoors. Seal windows. Use air purifiers. Seek medical help if symptoms appear.",
                "health_msg": "Health alert: everyone may experience more serious health effects. Emergency conditions."}


# ── AQICN API ──
def resolve_city_name(city_input):
    """Map state names to city names and clean input."""
    cleaned = city_input.strip().lower()
    # Check if it's a state name
    if cleaned in INDIA_STATE_MAP:
        return INDIA_STATE_MAP[cleaned], f"'{city_input}' is a state. Showing data for its capital: {INDIA_STATE_MAP[cleaned].title()}"
    return cleaned, None

def fetch_city_aqi(city_input='delhi'):
    """Fetch real-time AQI with pollutant breakdown from AQICN."""
    city, state_note = resolve_city_name(city_input)
    try:
        # Try geo search first for better accuracy
        url = f'https://api.waqi.info/feed/{city}/?token={AQICN_API_KEY}'
        r = requests.get(url, timeout=8)
        d = r.json()
        if d.get('status') == 'ok':
            data = d['data']
            returned_city = data['city']['name'].lower()
            
            # Validate: check if demo key returned a wrong city (e.g., Shanghai)
            if AQICN_API_KEY == 'demo':
                # Demo key often returns Shanghai for all queries
                if 'shanghai' in returned_city and 'shanghai' not in city.lower():
                    return {
                        'status': 'error',
                        'message': f"The demo API key cannot fetch data for '{city_input}'. It only supports limited queries. "
                                   f"Get a FREE API key at: https://aqicn.org/data-platform/token/ "
                                   f"Then update AQICN_API_KEY in app.py"
                    }
            
            iaqi = data.get('iaqi', {})
            result = {
                'status': 'ok',
                'aqi': int(data['aqi']),
                'city': data['city']['name'],
                'searched': city_input,
                'time': data['time']['s'],
                'pm25': iaqi.get('pm25', {}).get('v', None),
                'pm10': iaqi.get('pm10', {}).get('v', None),
                'no2': iaqi.get('no2', {}).get('v', None),
                'co': iaqi.get('co', {}).get('v', None),
                'o3': iaqi.get('o3', {}).get('v', None),
                'so2': iaqi.get('so2', {}).get('v', None),
                'humidity': iaqi.get('h', {}).get('v', None),
                'temp': iaqi.get('t', {}).get('v', None),
                'wind': iaqi.get('w', {}).get('v', None),
                'state_note': state_note,
            }
            return result
        return {'status': 'error', 'message': f"City '{city_input}' not found. Try specific city names like 'delhi', 'mumbai', 'patna' instead of state names."}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# ── Prediction History ──
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=['Date','Predicted_Hour','Predicted_Day','Actual_AQI','Target_Column','Error'])

def save_prediction(pred_h, pred_d, target):
    row = {
        'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'Predicted_Hour': round(pred_h, 2), 'Predicted_Day': round(pred_d, 2),
        'Actual_AQI': '', 'Target_Column': target, 'Error': ''
    }
    write_header = not os.path.exists(HISTORY_FILE)
    pd.DataFrame([row]).to_csv(HISTORY_FILE, mode='a', header=write_header, index=False)

def update_actual(date_str, val):
    h = load_history()
    mask = h['Date'].dt.strftime('%Y-%m-%d %H:%M') == date_str
    if mask.any():
        h.loc[mask, 'Actual_AQI'] = val
        h.loc[mask, 'Error'] = abs(h.loc[mask, 'Predicted_Day'] - val)
    h.to_csv(HISTORY_FILE, index=False)
    return h


# ── LLM / AI Advisor ──
def generate_ai_explanation(aqi_val, pollutants=None, city=None, pred_hour=None, pred_day=None):
    """Generate health explanation using Gemini API or smart templates."""

    # Build context
    status = classify_aqi(aqi_val)
    ctx_parts = [f"Current AQI is {aqi_val} ({status['label']})."]
    if city:
        ctx_parts.append(f"City: {city}.")
    if pollutants:
        for k, v in pollutants.items():
            if v is not None:
                ctx_parts.append(f"{k.upper()}: {v}")
    if pred_hour is not None:
        ctx_parts.append(f"Predicted next hour: {pred_hour:.1f}")
    if pred_day is not None:
        ctx_parts.append(f"Predicted next day: {pred_day:.1f}")

    # Try Gemini first
    gemini_key = os.environ.get('GEMINI_API_KEY', 'AIzaSyDkqSha1rOxwdrZiQGwBaYRVJo5z2udoM4')
    if GEMINI_AVAILABLE and gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            prompt = f"""You are an air quality health advisor. Based on the following data, provide:
1. A clear 2-3 sentence summary of current air quality
2. Health impact for children and elderly
3. Specific precautions and recommendations
4. Trend analysis if prediction data is available

Data: {' '.join(ctx_parts)}

Keep the response concise, professional, and actionable. Use simple language."""
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            return response.text
        except Exception:
            pass

    # Smart template fallback
    trend = ""
    if pred_day is not None:
        if pred_day > aqi_val * 1.1:
            trend = "📈 **Worsening Trend:** AQI is predicted to increase over the next 24 hours. Take preventive measures now."
        elif pred_day < aqi_val * 0.9:
            trend = "📉 **Improving Trend:** AQI is expected to decrease. Conditions should improve gradually."
        else:
            trend = "➡️ **Stable Trend:** AQI is expected to remain at similar levels for the next 24 hours."

    pm_warning = ""
    if pollutants:
        pm25 = pollutants.get('pm25')
        if pm25 and pm25 > 35:
            pm_warning = f"\n\n⚠️ **PM2.5 Alert:** At {pm25} µg/m³, fine particulate matter exceeds WHO guidelines (15 µg/m³). PM2.5 penetrates deep into lungs and can enter the bloodstream, increasing cardiovascular and respiratory risks. Children and elderly are most vulnerable."
        no2 = pollutants.get('no2')
        if no2 and no2 > 40:
            pm_warning += f"\n\n⚠️ **NO₂ Alert:** Nitrogen dioxide at {no2} µg/m³ exceeds safe limits. Prolonged exposure can aggravate asthma and reduce lung function, particularly in children."
        co = pollutants.get('co')
        if co and co > 10:
            pm_warning += f"\n\n⚠️ **CO Alert:** Carbon monoxide elevated at {co} mg/m³. Can cause headaches, dizziness, and reduced oxygen delivery to organs."

    city_str = f" in **{city}**" if city else ""
    pred_str = ""
    if pred_hour is not None and pred_day is not None:
        pred_str = f"\n\n**Forecast:** Next hour AQI is predicted at **{pred_hour:.1f}**, and 24-hour forecast shows **{pred_day:.1f}**."

    return f"""**🩺 AI Health Assessment{city_str}**

The current Air Quality Index is **{aqi_val}**, classified as **{status['emoji']} {status['label']}**. {status['health_msg']}

**👶 Children & Elderly:** {status['child']}
**👨 General Population:** {status['adult']}

**💊 Recommended Action:** {status['advice']}{pm_warning}{pred_str}

{trend}

*Based on WHO Air Quality Guidelines & Athens Longitudinal Study (2001-2020). Model performance is continuously monitored using prediction error analysis.*"""


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.title("🌬️ AirAI")
    st.caption("Intelligent AQI Prediction & Health Analysis")
    st.markdown("---")

    menu = st.radio("NAV", ["🏠 Dashboard", "📈 History & Performance", "🔬 Research", "⚙️ About"],
                    index=0, label_visibility="collapsed")
    st.markdown("---")

    # City AQI Search in sidebar
    st.markdown("##### 🌍 Live AQI Search")
    city_name = st.text_input("City", value="", placeholder="e.g. delhi, mumbai, patna",
        help="Enter a city name (not state). Examples: delhi, mumbai, bengaluru, patna, jaipur, london, beijing")
    fetch_city = st.button("🔍 Fetch AQI", width="stretch")
    st.caption("🏙️ delhi · mumbai · kolkata · bengaluru · chennai · hyderabad · patna · jaipur · lucknow · ahmedabad")

    st.markdown("---")
    st.markdown("""
    <div style="padding:14px;background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.12);border-radius:14px">
        <p style="font-size:.78rem;color:#64748b;margin:0">
            💡 <b style="color:#a5b4fc">Tip</b><br>
            Search any city for real-time AQI, or upload CSV data for LSTM-based forecasting.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<p style="font-size:.7rem;color:#475569;text-align:center">Powered by LSTM · TensorFlow · AQICN API<br>B.Tech 8-Credit Project — Manu Kumar</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 1: MAIN DASHBOARD
# ══════════════════════════════════════════════
if menu == "🏠 Dashboard":

    st.markdown('<div class="hero"><h1 class="hero-title">Air Quality Intelligence</h1><p class="hero-sub">Real-time monitoring, LSTM predictions, health risk analysis & AI-powered insights</p></div>', unsafe_allow_html=True)

    # ──────────────────────────────────────
    # SECTION 1: REAL-TIME CITY AQI
    # ──────────────────────────────────────
    city_data = None
    if fetch_city and city_name:
        with st.spinner(f"Fetching AQI for {city_name}..."):
            city_data = fetch_city_aqi(city_name)
        if city_data and city_data.get('status') == 'ok':
            st.session_state['city_data'] = city_data
        else:
            st.error(city_data.get('message', f"❌ No data found for '{city_name}'"))
            city_data = None
    elif 'city_data' in st.session_state:
        city_data = st.session_state['city_data']

    if city_data and city_data['status'] == 'ok':
        st.markdown('<div class="sec-h">🌍 Real-Time City Air Quality</div>', unsafe_allow_html=True)

        aqi_status = classify_aqi(city_data['aqi'])

        # Top summary cards
        s1, s2, s3, s4 = st.columns(4)
        
        display_name = city_data.get('searched', 'Unknown').title()
        station_name = city_data['city']
        if len(station_name) > 28: station_name = station_name[:25] + "..."
        
        s1.metric(f"Live AQI ({display_name})", f"{city_data['aqi']}", f"📍 {station_name}")
        s2.metric("Category", aqi_status['label'])
        s3.metric("Health Risk", "Level " + str(aqi_status['level']) + "/5")
        s4.metric("Updated", city_data['time'][:16] if city_data['time'] else "—")

        # Pollutant breakdown
        st.markdown('<div class="sec-h">🧪 Pollutant Breakdown</div>', unsafe_allow_html=True)
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("PM2.5", f"{city_data['pm25'] or '—'}", "µg/m³")
        p2.metric("PM10", f"{city_data['pm10'] or '—'}", "µg/m³")
        p3.metric("NO₂", f"{city_data['no2'] or '—'}", "µg/m³")
        p4.metric("CO", f"{city_data['co'] or '—'}", "mg/m³")
        p5.metric("O₃", f"{city_data['o3'] or '—'}", "µg/m³")
        p6.metric("SO₂", f"{city_data['so2'] or '—'}", "µg/m³")

        # Environment
        e1, e2, e3 = st.columns(3)
        e1.metric("🌡️ Temperature", f"{city_data['temp']}°C" if city_data['temp'] else "—")
        e2.metric("💧 Humidity", f"{city_data['humidity']}%" if city_data['humidity'] else "—")
        e3.metric("💨 Wind", f"{city_data['wind']} m/s" if city_data['wind'] else "—")

        # Health Risk Card
        st.markdown(f"""
        <div class="rcard {aqi_status['css']}">
            <div style="font-size:1.1rem;font-weight:700;color:{aqi_status['color']};margin-bottom:14px">
                {aqi_status['emoji']} Health Risk Assessment — {aqi_status['label']}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
                <div>
                    <div style="font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:{aqi_status['color']};margin-bottom:6px">👶 Children (High Sensitivity)</div>
                    <div style="font-size:.88rem;color:#94a3b8;line-height:1.5">{aqi_status['child']}</div>
                </div>
                <div>
                    <div style="font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:{aqi_status['color']};margin-bottom:6px">🚶 Adults (General Population)</div>
                    <div style="font-size:.88rem;color:#94a3b8;line-height:1.5">{aqi_status['adult']}</div>
                </div>
            </div>
            <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,.06)">
                <div style="font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:{aqi_status['color']};margin-bottom:6px">💊 Recommended Action</div>
                <div style="font-size:.88rem;color:#94a3b8">{aqi_status['advice']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # LLM Explanation
        st.markdown("---")
        st.markdown('<div class="sec-h">🧠 AI Health Advisor</div>', unsafe_allow_html=True)
        pollutant_dict = {k: city_data.get(k) for k in ['pm25','pm10','no2','co','o3','so2']}
        explanation = generate_ai_explanation(city_data['aqi'], pollutants=pollutant_dict, city=city_data['city'])
        st.markdown(f'<div class="llm-box"><div class="llm-header">🤖 AI-Powered Health Analysis</div>{explanation}</div>', unsafe_allow_html=True)

        st.markdown("---")

    # ──────────────────────────────────────
    # SECTION 2: DATA UPLOAD & ANALYSIS
    # ──────────────────────────────────────
    st.markdown('<div class="sec-h">📂 Data Upload & LSTM Analysis</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload hourly air quality CSV", type=["csv"],
        help="Auto-detects separators, cleans missing values, and prepares data for LSTM forecasting.")

    model, scaler = load_forecast_engine()

    if uploaded_file and model and scaler:
        # Load CSV with format detection
        try:
            raw = pd.read_csv(uploaded_file, sep=';', decimal=',', low_memory=False)
            if len(raw.columns) <= 2:
                uploaded_file.seek(0)
                raw = pd.read_csv(uploaded_file, low_memory=False)
        except Exception:
            uploaded_file.seek(0)
            raw = pd.read_csv(uploaded_file, low_memory=False)

        orig_rows = len(raw)
        with st.spinner("🧹 Auto-cleaning dataset..."):
            df, target_col, clean_log = auto_clean_dataset(raw)

        if target_col is None:
            st.error("❌ No numeric column found. Check your CSV format.")
            st.stop()

        st.toast(f"✅ Cleaned {len(df)} records → {target_col}", icon='🚀')

        # Cleaning summary
        with st.expander("🧹 Auto-Cleaning Report", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Original Rows", f"{orig_rows:,}")
            c2.metric("Clean Rows", f"{len(df):,}")
            c3.metric("Actions", f"{len(clean_log)}")
            for action in clean_log:
                st.markdown(f"• {action}")

        # Data pills
        st.markdown(f"""
        <div style="margin:12px 0 20px">
            <span class="pill">📄 {len(df):,} Records</span>
            <span class="pill">🎯 Target: {target_col}</span>
            <span class="pill">📈 {df[target_col].min():.1f} – {df[target_col].max():.1f} µg/m³</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Trend Chart ──
        st.markdown('<div class="sec-h">📈 Pollutant Concentration Trend</div>', unsafe_allow_html=True)
        chart_col, stat_col = st.columns([5, 2])

        with chart_col:
            recent = df.tail(200)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(recent))), y=recent[target_col].values,
                mode='lines', fill='tozeroy', line=dict(color='#818cf8', width=2.5, shape='spline'),
                fillcolor='rgba(129,140,248,.08)', name=target_col,
                hovertemplate='<b>Hour %{x}</b><br>%{y:.2f} µg/m³<extra></extra>'))
            # Moving average
            if len(recent) >= 24:
                ma = recent[target_col].rolling(24).mean()
                fig.add_trace(go.Scatter(x=list(range(len(recent))), y=ma.values,
                    mode='lines', line=dict(color='#f59e0b', width=2, dash='dash'),
                    name='24h Moving Avg',
                    hovertemplate='<b>Hour %{x}</b><br>MA: %{y:.2f}<extra></extra>'))
            # Thresholds
            fig.add_hline(y=50, line_dash="dot", line_color="rgba(52,211,153,.4)", annotation_text="Good (50)")
            fig.add_hline(y=100, line_dash="dot", line_color="rgba(251,191,36,.4)", annotation_text="Moderate (100)")
            fig.add_hline(y=150, line_dash="dot", line_color="rgba(248,113,113,.4)", annotation_text="Unhealthy (150)")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=10,b=0), height=380,
                xaxis=dict(title="Hours", gridcolor="rgba(99,102,241,.06)", title_font_color="#64748b", tickfont_color="#475569"),
                yaxis=dict(title="Concentration (µg/m³)", gridcolor="rgba(99,102,241,.06)", title_font_color="#64748b", tickfont_color="#475569"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#94a3b8")),
                hoverlabel=dict(bgcolor="#1e293b", font_color="#e0e7ff", bordercolor="#6366f1"))
            st.plotly_chart(fig, width="stretch")

        with stat_col:
            current_val = df[target_col].iloc[-1]
            avg_24 = df[target_col].tail(24).mean()
            peak_24 = df[target_col].tail(24).max()

            st.metric("Latest", f"{current_val:.2f}", "µg/m³")
            st.metric("24h Average", f"{avg_24:.2f}", f"{current_val - avg_24:+.1f} vs avg")
            st.metric("24h Peak", f"{peak_24:.2f}")

            cs = classify_aqi(current_val)
            st.markdown(f"""
            <div style="background:rgba(15,23,42,.6);border:1px solid {cs['color']}33;border-radius:14px;padding:14px;margin-top:8px;text-align:center">
                <div style="font-size:1.6rem">{cs['emoji']}</div>
                <div style="color:{cs['color']};font-weight:700;font-size:.95rem;margin-top:4px">{cs['label']}</div>
                <div style="font-size:.72rem;color:#64748b;margin-top:4px">Current Status</div>
            </div>
            """, unsafe_allow_html=True)

        # ── LSTM Prediction ──
        st.markdown("---")
        st.markdown('<div class="sec-h">🤖 LSTM Predictive Forecasting</div>', unsafe_allow_html=True)

        pc1, pc2 = st.columns([1, 2])
        with pc1:
            st.markdown("""
            <div class="gcard">
                <h4 style="color:#c7d2fe;margin-top:0">LSTM Neural Engine</h4>
                <p style="color:#64748b;font-size:.85rem;line-height:1.6">
                    Uses the last <b style="color:#a5b4fc">24 hours</b> to recursively forecast t+1 to t+24 
                    using our trained 64-unit LSTM with dropout regularization.
                </p>
            </div>
            """, unsafe_allow_html=True)
            generate = st.button("🚀 Run AI Inference", width="stretch")

        if generate:
            if len(df) < 24:
                st.error("❌ Need at least 24 rows for LSTM inference.")
            else:
                with st.spinner("Running recursive inference (t+1 → t+24)..."):
                    last_24 = df[target_col].tail(24).values.reshape(-1, 1)
                    last_24_s = scaler.transform(last_24).flatten()
                    pred_h, pred_d, all_preds = calculate_multi_forecast(model, scaler, last_24_s)

                # Save prediction
                save_prediction(pred_h, pred_d, target_col)
                st.toast("💾 Prediction saved to history", icon='📊')

                with pc2:
                    hd = pred_h - current_val
                    dd = pred_d - current_val
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current", f"{current_val:.2f}", "µg/m³")
                    m2.metric("Next Hour", f"{pred_h:.2f}", f"{hd:+.2f}", delta_color="inverse")
                    m3.metric("Next Day", f"{pred_d:.2f}", f"{dd:+.2f}", delta_color="inverse")

                # 24h Forecast chart
                st.markdown('<div class="sec-h">📉 24-Hour Forecast Trajectory</div>', unsafe_allow_html=True)
                hist_vals = df[target_col].tail(24).values
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=list(range(-24, 0)), y=hist_vals, mode='lines',
                    line=dict(color='#64748b', width=2, dash='dot'), name='Historical',
                    hovertemplate='<b>%{x}h ago</b><br>%{y:.2f}<extra></extra>'))
                fig2.add_trace(go.Scatter(x=list(range(0, 24)), y=all_preds, mode='lines+markers',
                    line=dict(color='#818cf8', width=3, shape='spline'),
                    marker=dict(size=5, color='#a5b4fc'), fill='tozeroy',
                    fillcolor='rgba(129,140,248,.06)', name='Forecast',
                    hovertemplate='<b>+%{x}h</b><br>%{y:.2f}<extra></extra>'))
                fig2.add_vline(x=0, line_dash="solid", line_color="rgba(99,102,241,.5)", line_width=2)
                fig2.add_annotation(x=0, y=max(max(hist_vals), max(all_preds)), text="NOW",
                    showarrow=False, font=dict(color="#818cf8", size=12), yshift=15)
                # Threshold zones
                fig2.add_hrect(y0=0, y1=50, fillcolor="rgba(52,211,153,.03)", line_width=0)
                fig2.add_hrect(y0=50, y1=100, fillcolor="rgba(251,191,36,.03)", line_width=0)
                fig2.add_hrect(y0=100, y1=200, fillcolor="rgba(248,113,113,.03)", line_width=0)
                fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0,r=0,t=10,b=0), height=340,
                    xaxis=dict(title="Hours from Now", gridcolor="rgba(99,102,241,.06)", zeroline=False, title_font_color="#64748b", tickfont_color="#475569"),
                    yaxis=dict(title="Concentration (µg/m³)", gridcolor="rgba(99,102,241,.06)", title_font_color="#64748b", tickfont_color="#475569"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#94a3b8")),
                    hoverlabel=dict(bgcolor="#1e293b", font_color="#e0e7ff", bordercolor="#6366f1"))
                st.plotly_chart(fig2, width="stretch")

                # Health Risk for prediction
                st.markdown('<div class="sec-h">🚨 Predicted Health Risk</div>', unsafe_allow_html=True)
                risk = classify_aqi(pred_d)
                st.markdown(f"""
                <div class="rcard {risk['css']}">
                    <div style="font-size:1.1rem;font-weight:700;color:{risk['color']};margin-bottom:14px">
                        {risk['emoji']} 24-Hour Outlook: {risk['label']}
                        <span style="font-size:.78rem;color:#64748b;font-weight:400;margin-left:8px">Predicted: {pred_d:.2f} µg/m³</span>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
                        <div>
                            <div style="font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:{risk['color']};margin-bottom:6px">👶 Children</div>
                            <div style="font-size:.88rem;color:#94a3b8;line-height:1.5">{risk['child']}</div>
                        </div>
                        <div>
                            <div style="font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:{risk['color']};margin-bottom:6px">🚶 Adults</div>
                            <div style="font-size:.88rem;color:#94a3b8;line-height:1.5">{risk['adult']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Alert system
                if pred_d > 150 and current_val <= 150:
                    st.warning("🚨 **ALERT:** AQI is predicted to cross the Unhealthy threshold (150) within 24 hours. Take preventive measures now!")
                elif pred_d > 100 and current_val <= 100:
                    st.warning("⚠️ **WARNING:** AQI is predicted to enter Moderate/Unhealthy zone within 24 hours.")

                # LLM Explanation
                st.markdown("---")
                st.markdown('<div class="sec-h">🧠 AI Health Advisor</div>', unsafe_allow_html=True)
                explanation = generate_ai_explanation(current_val, pred_hour=pred_h, pred_day=pred_d)
                st.markdown(f'<div class="llm-box"><div class="llm-header">🤖 AI-Powered Health Analysis</div>\n\n{explanation}</div>', unsafe_allow_html=True)

                # Insight box
                trend = "📈 Increasing" if dd > 0 else "📉 Decreasing"
                risk_esc = "⚠️ Risk escalation detected" if dd > current_val * 0.2 else "✅ No immediate escalation"
                st.markdown(f"""
                <div class="ibox">
                    <b style="color:#a5b4fc">🧠 Forecast Intelligence</b><br><br>
                    • <b>Trend:</b> {trend} — change of <b>{dd:+.2f} µg/m³</b><br>
                    • <b>Risk:</b> {risk_esc}<br>
                    • <b>Architecture:</b> 24-step recursive LSTM (64 units, dropout 0.2)
                </div>
                """, unsafe_allow_html=True)

    elif uploaded_file and (not model or not scaler):
        st.error("❌ Model/Scaler not found. Run `python train_model.py` first.")
    elif not uploaded_file and not (city_data and city_data.get('status') == 'ok'):
        st.markdown("""
        <div class="empty">
            <div style="font-size:3.5rem;margin-bottom:14px">🌬️</div>
            <h2>Welcome to AirAI</h2>
            <p>Search a city in the sidebar for real-time AQI, or upload a CSV for LSTM predictions.</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 2: HISTORY & PERFORMANCE
# ══════════════════════════════════════════════
elif menu == "📈 History & Performance":
    st.markdown('<div class="hero"><h1 class="hero-title">Prediction Intelligence Hub</h1><p class="hero-sub">Track predictions, compare with reality, monitor model performance</p></div>', unsafe_allow_html=True)

    history = load_history()

    if len(history) == 0:
        st.markdown('<div class="empty"><div style="font-size:3.5rem;margin-bottom:14px">📊</div><h2>No Predictions Yet</h2><p>Run AI Inference on the Dashboard to start building prediction history.</p></div>', unsafe_allow_html=True)
    else:
        total = len(history)
        has_actual = history['Actual_AQI'].notna()
        verified = int(has_actual.sum())

        if verified > 0:
            v_df = history[has_actual].copy()
            v_df['Error'] = abs(v_df['Predicted_Day'] - v_df['Actual_AQI'])
            mae = v_df['Error'].mean()
            mape = (v_df['Error'] / v_df['Actual_AQI'].replace(0, np.nan)).mean() * 100
            acc = max(0, 100 - mape)
        else:
            mae = acc = None

        # Overview
        st.markdown('<div class="sec-h">📊 Overview</div>', unsafe_allow_html=True)
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Total Predictions", total)
        o2.metric("Verified", verified)
        o3.metric("MAE", f"{mae:.2f} µg/m³" if mae else "—")
        o4.metric("Accuracy", f"{acc:.1f}%" if acc else "—")

        # Table
        st.markdown("---")
        st.markdown('<div class="sec-h">🗂 Prediction History</div>', unsafe_allow_html=True)
        disp = history.copy()
        disp['Date'] = disp['Date'].dt.strftime('%d %b %Y, %H:%M')
        disp.columns = ['Date', 'Pred (Hour)', 'Pred (Day)', 'Actual', 'Target', 'Error']
        st.dataframe(disp.iloc[::-1].reset_index(drop=True), width="stretch", hide_index=True)

        # Predicted vs Actual graph
        if verified >= 2:
            st.markdown("---")
            st.markdown('<div class="sec-h">📉 Predicted vs Actual</div>', unsafe_allow_html=True)
            vs = v_df.sort_values('Date')
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(x=vs['Date'], y=vs['Predicted_Day'], mode='lines+markers',
                name='Predicted', line=dict(color='#818cf8', width=3, shape='spline'),
                marker=dict(size=8, color='#a5b4fc', line=dict(color='#6366f1', width=2))))
            fig_c.add_trace(go.Scatter(x=vs['Date'], y=vs['Actual_AQI'], mode='lines+markers',
                name='Actual', line=dict(color='#34d399', width=3, shape='spline'),
                marker=dict(size=8, color='#6ee7b7', line=dict(color='#10b981', width=2))))
            fig_c.add_trace(go.Scatter(x=pd.concat([vs['Date'], vs['Date'].iloc[::-1]]),
                y=pd.concat([vs['Predicted_Day'], vs['Actual_AQI'].iloc[::-1]]),
                fill='toself', fillcolor='rgba(248,113,113,.08)', line=dict(color='rgba(0,0,0,0)'),
                name='Error Zone', hoverinfo='skip'))
            fig_c.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=10,b=0), height=380,
                xaxis=dict(gridcolor="rgba(99,102,241,.06)", tickfont_color="#475569"),
                yaxis=dict(title="µg/m³", gridcolor="rgba(99,102,241,.06)", title_font_color="#64748b", tickfont_color="#475569"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#94a3b8")),
                hoverlabel=dict(bgcolor="#1e293b", font_color="#e0e7ff", bordercolor="#6366f1"))
            st.plotly_chart(fig_c, width="stretch")

            # Error bar + accuracy trend
            st.markdown('<div class="sec-h">📈 Model Performance Trend</div>', unsafe_allow_html=True)
            pf1, pf2 = st.columns(2)
            with pf1:
                fig_e = go.Figure()
                fig_e.add_trace(go.Bar(x=vs['Date'].dt.strftime('%d %b'), y=vs['Error'],
                    marker_color=['#34d399' if e<20 else '#fbbf24' if e<50 else '#f87171' for e in vs['Error']]))
                fig_e.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0,r=0,t=30,b=0), height=260,
                    title=dict(text="Error per Prediction", font=dict(color="#94a3b8", size=13)),
                    yaxis=dict(title="Error (µg/m³)", gridcolor="rgba(99,102,241,.06)", title_font_color="#64748b", tickfont_color="#475569"),
                    xaxis=dict(tickfont_color="#475569"))
                st.plotly_chart(fig_e, width="stretch")
            with pf2:
                ra = 100 - (vs['Error'] / vs['Actual_AQI'].replace(0, np.nan) * 100).expanding().mean()
                ra = ra.clip(lower=0)
                fig_a = go.Figure()
                fig_a.add_trace(go.Scatter(x=vs['Date'].dt.strftime('%d %b'), y=ra, mode='lines+markers',
                    line=dict(color='#818cf8', width=3, shape='spline'), marker=dict(size=7, color='#a5b4fc'),
                    fill='tozeroy', fillcolor='rgba(129,140,248,.08)'))
                fig_a.add_hline(y=80, line_dash="dot", line_color="rgba(52,211,153,.4)", annotation_text="80% Target", annotation_font_color="#34d399")
                fig_a.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0,r=0,t=30,b=0), height=260,
                    title=dict(text="Accuracy Trend", font=dict(color="#94a3b8", size=13)),
                    yaxis=dict(title="Accuracy %", gridcolor="rgba(99,102,241,.06)", title_font_color="#64748b", tickfont_color="#475569", range=[0,105]),
                    xaxis=dict(tickfont_color="#475569"))
                st.plotly_chart(fig_a, width="stretch")

        # Enter actual AQI
        st.markdown("---")
        st.markdown('<div class="sec-h">✏️ Enter Actual AQI Values</div>', unsafe_allow_html=True)
        pending = history[history['Actual_AQI'].isna()].copy()
        if len(pending) > 0:
            uc1, uc2, uc3 = st.columns([2, 1, 1])
            with uc1:
                opts = pending['Date'].dt.strftime('%Y-%m-%d %H:%M').tolist()
                sel = st.selectbox("Select prediction", opts,
                    format_func=lambda x: f"{pd.to_datetime(x).strftime('%d %b %Y %H:%M')} — Pred: {pending[pending['Date'].dt.strftime('%Y-%m-%d %H:%M')==x]['Predicted_Day'].values[0]:.2f}")
            with uc2:
                av = st.number_input("Actual AQI", min_value=0.0, max_value=999.0, step=0.5)
            with uc3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✅ Save", width="stretch"):
                    if av > 0:
                        update_actual(sel, av)
                        st.success(f"✅ Saved actual AQI ({av:.1f})")
                        st.rerun()
        else:
            st.success("✅ All predictions verified!")

        # Fetch live AQI
        st.markdown("---")
        st.markdown('<div class="sec-h">🌐 Fetch Live AQI for Comparison</div>', unsafe_allow_html=True)
        fc1, fc2 = st.columns([2, 1])
        with fc1:
            fc_city = st.text_input("City name", value="delhi", key="hist_city")
        with fc2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🌐 Fetch", width="stretch"):
                r = fetch_city_aqi(fc_city)
                if r['status'] == 'ok':
                    st.metric("Live AQI", r['aqi'], r['city'])
                    st.info(f"💡 Use {r['aqi']} as actual AQI for your latest prediction above.")
                else:
                    st.error(f"❌ {r.get('message', 'Error')}")

        # Adaptive learning note
        st.markdown("---")
        st.markdown('<div class="sec-h">🧠 Adaptive Learning System</div>', unsafe_allow_html=True)
        if verified > 0 and mae is not None:
            ps = "🟢 Excellent" if mae < 30 else "🟡 Acceptable" if mae < 60 else "🟠 Needs Improvement"
            pc = "#34d399" if mae < 30 else "#fbbf24" if mae < 60 else "#fb923c"
            pm = "Within acceptable margins." if mae < 30 else "Could benefit from retraining." if mae < 60 else "Consider retraining with updated data."
        else:
            ps, pc, pm = "⚪ Awaiting Data", "#64748b", "Add actual AQI values to start monitoring."

        st.markdown(f"""
        <div class="gcard">
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px">
                <div style="font-size:1.8rem">{ps.split(' ')[0]}</div>
                <div><div style="color:{pc};font-weight:700;font-size:1.05rem">{ps}</div>
                <div style="color:#64748b;font-size:.82rem">{pm}</div></div>
            </div>
            <div style="border-top:1px solid rgba(99,102,241,.1);padding-top:14px">
                <p style="color:#94a3b8;font-size:.85rem;line-height:1.7;margin:0">
                    Our system <b style="color:#a5b4fc">stores predictions and compares them with actual data</b>,
                    enabling continuous performance evaluation. Model performance is monitored using prediction errors
                    and retraining can be performed using updated datasets via <code style="color:#818cf8">python train_model.py</code>.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 3: RESEARCH
# ══════════════════════════════════════════════
elif menu == "🔬 Research":
    st.markdown('<div class="hero"><h1 class="hero-title">Research Foundation</h1><p class="hero-sub">Athens Longitudinal Study (2001–2020) & WHO AirQ+ Model</p></div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    for v, l, c in [("20", "Years of Data", s1), ("PM10", "Primary Pollutant", s2), ("15-20", "µg/m³ Threshold", s3), ("AirQ+", "WHO Model", s4)]:
        with c:
            st.markdown(f'<div class="rstat"><div class="bn">{v}</div><div class="sl">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col_s, col_g = st.columns([3, 2])
    with col_s:
        st.markdown("""
        <div class="gcard">
            <h3 style="color:#c7d2fe;margin-top:0">📖 The Athens Study</h3>
            <p style="color:#94a3b8;line-height:1.8;font-size:.9rem">
                This project implements findings from a 20-year study on PM10 exposure in the
                <b style="color:#a5b4fc">Greater Athens Area</b> using the <b style="color:#a5b4fc">WHO AirQ+ model</b>.
            </p>
            <ul style="color:#94a3b8;line-height:2;font-size:.88rem">
                <li><b style="color:#34d399">Strong Correlation</b> — PM10 levels linked to respiratory hospital admissions</li>
                <li><b style="color:#fbbf24">Vulnerable Groups</b> — Children significantly more affected than adults</li>
                <li><b style="color:#f87171">Critical Thresholds</b> — Health risks begin at 15–20 µg/m³</li>
                <li><b style="color:#818cf8">Long-term Trend</b> — Impacts decreased after 2010</li>
                <li><b style="color:#fb923c">Spatial Variation</b> — Central Athens higher than peripheral areas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_g:
        fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=15,
            delta={'reference': 20, 'decreasing': {'color': '#34d399'}, 'increasing': {'color': '#f87171'}},
            title={'text': "Athens Baseline (µg/m³)", 'font': {'color': '#94a3b8', 'size': 13}},
            number={'font': {'color': '#e0e7ff', 'size': 40}},
            gauge={'axis': {'range': [0, 60], 'tickcolor': '#475569', 'tickfont': {'color': '#475569'}},
                   'bar': {'color': '#818cf8', 'thickness': 0.3}, 'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                   'steps': [{'range': [0, 50], 'color': 'rgba(52,211,153,.12)'}, {'range': [50, 100], 'color': 'rgba(251,191,36,.12)'},
                             {'range': [100, 150], 'color': 'rgba(251,146,60,.12)'}, {'range': [150, 200], 'color': 'rgba(248,113,113,.12)'}],
                   'threshold': {'line': {'color': '#f87171', 'width': 3}, 'thickness': 0.8, 'value': 20}}))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig, width="stretch")

        # Legend
        st.markdown("""
        <div class="gcard" style="padding:16px">
            <div style="font-size:.78rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">AQI Health Zones</div>
            <div style="display:flex;flex-direction:column;gap:6px">
                <span style="color:#94a3b8;font-size:.82rem"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#34d399;margin-right:8px"></span><b style="color:#34d399">Good</b> — 0–50</span>
                <span style="color:#94a3b8;font-size:.82rem"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#fbbf24;margin-right:8px"></span><b style="color:#fbbf24">Moderate</b> — 51–100</span>
                <span style="color:#94a3b8;font-size:.82rem"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#fb923c;margin-right:8px"></span><b style="color:#fb923c">Unhealthy (Sensitive)</b> — 101–150</span>
                <span style="color:#94a3b8;font-size:.82rem"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:8px"></span><b style="color:#ef4444">Unhealthy</b> — 151–200</span>
                <span style="color:#94a3b8;font-size:.82rem"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#a855f7;margin-right:8px"></span><b style="color:#a855f7">Hazardous</b> — 200+</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 4: ABOUT
# ══════════════════════════════════════════════
elif menu == "⚙️ About":
    st.markdown('<div class="hero"><h1 class="hero-title">Technical Architecture</h1><p class="hero-sub">Deep Learning pipeline for real-time AQI forecasting & health risk early-warning</p></div>', unsafe_allow_html=True)

    # Architecture flow
    st.markdown("""
    <div class="gcard" style="text-align:center">
        <div style="font-size:.78rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:18px">System Architecture</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap">
            <span class="pill" style="background:rgba(52,211,153,.1);border-color:rgba(52,211,153,.2);color:#34d399">📄 CSV / API</span>
            <span style="color:#475569">→</span>
            <span class="pill" style="background:rgba(99,102,241,.1);border-color:rgba(99,102,241,.2);color:#818cf8">🧹 Auto-Clean</span>
            <span style="color:#475569">→</span>
            <span class="pill" style="background:rgba(99,102,241,.1);border-color:rgba(99,102,241,.2);color:#818cf8">🔧 Scaler</span>
            <span style="color:#475569">→</span>
            <span class="pill" style="background:rgba(251,191,36,.1);border-color:rgba(251,191,36,.2);color:#fbbf24">🧠 LSTM (64)</span>
            <span style="color:#475569">→</span>
            <span class="pill" style="background:rgba(251,146,60,.1);border-color:rgba(251,146,60,.2);color:#fb923c">⚡ Dense (32)</span>
            <span style="color:#475569">→</span>
            <span class="pill" style="background:rgba(248,113,113,.1);border-color:rgba(248,113,113,.2);color:#f87171">🎯 Prediction</span>
            <span style="color:#475569">→</span>
            <span class="pill" style="background:rgba(168,85,247,.1);border-color:rgba(168,85,247,.2);color:#a855f7">🤖 AI Advisor</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Five Pillars
    st.markdown('<div class="sec-h">🏗️ System Pillars</div>', unsafe_allow_html=True)
    pillars = [
        ("🤖", "LSTM Prediction", "Time-series forecasting using 64-unit LSTM with dropout regularization"),
        ("🔬", "Health Analysis", "Research-backed risk assessment using WHO AirQ+ model & Athens study"),
        ("🌐", "Real-Time API", "Live AQI & pollutant data for any city via AQICN API"),
        ("🧠", "AI Advisor", "LLM-powered health explanations with Gemini integration"),
        ("📊", "Feedback Loop", "Prediction history, accuracy tracking, and adaptive monitoring"),
    ]
    cols = st.columns(5)
    for (icon, name, desc), col in zip(pillars, cols):
        with col:
            st.markdown(f"""
            <div class="rstat" style="padding:18px">
                <div style="font-size:1.8rem;margin-bottom:8px">{icon}</div>
                <div style="font-weight:700;color:#e0e7ff;font-size:.88rem;margin-bottom:4px">{name}</div>
                <div style="font-size:.75rem;color:#64748b">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Tech Stack
    st.markdown("")
    st.markdown('<div class="sec-h">🛠️ Technology Stack</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    for icon, name, role, col in [
        ("🧠", "TensorFlow / Keras", "LSTM neural network", t1),
        ("🖥️", "Streamlit", "Dashboard framework", t2),
        ("📊", "Plotly", "Interactive visualizations", t1),
        ("🐼", "Pandas / NumPy", "Data processing", t2),
        ("🌐", "AQICN API", "Real-time AQI data", t1),
        ("🤖", "Google Gemini", "LLM health advisor", t2),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:rgba(15,23,42,.55);border:1px solid rgba(99,102,241,.12);border-radius:14px;padding:16px;display:flex;align-items:center;gap:14px;margin-bottom:10px;transition:all .3s ease">
                <div style="font-size:1.6rem;width:44px;height:44px;display:flex;align-items:center;justify-content:center;background:rgba(99,102,241,.1);border-radius:12px;flex-shrink:0">{icon}</div>
                <div><div style="font-weight:700;color:#e0e7ff;font-size:.9rem">{name}</div><div style="font-size:.78rem;color:#64748b">{role}</div></div>
            </div>
            """, unsafe_allow_html=True)

    # Performance
    st.markdown("")
    st.markdown('<div class="sec-h">📈 Training Metrics</div>', unsafe_allow_html=True)
    for v, l, c in [("61.78", "MAE (µg/m³)", st.columns(4)[0]), ("90.81", "RMSE (µg/m³)", st.columns(4)[1]),
                     ("Epoch 33", "Best Weights", st.columns(4)[2]), ("Epoch 38", "Early Stop", st.columns(4)[3])]:
        with c:
            st.markdown(f'<div class="rstat"><div class="bn">{v}</div><div class="sl">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    <div class="gcard">
        <h4 style="color:#c7d2fe;margin-top:0">📋 Project Information</h4>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px">
            <div style="color:#64748b;font-size:.85rem"><b style="color:#94a3b8">Project:</b> Deep Learning for Real-Time AQI Forecasting & Health Risk Early-Warning</div>
            <div style="color:#64748b;font-size:.85rem"><b style="color:#94a3b8">Student:</b> Manu Kumar</div>
            <div style="color:#64748b;font-size:.85rem"><b style="color:#94a3b8">Program:</b> B.Tech CSE — 8-Credit Project</div>
            <div style="color:#64748b;font-size:.85rem"><b style="color:#94a3b8">Dataset:</b> UCI Air Quality (9,358 measurements)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
