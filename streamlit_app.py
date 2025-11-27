import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go

st.set_page_config(page_title="Cloud Anomaly Detector", layout="wide", initial_sidebar_state="expanded")

# Initialize session state for parameters
if 'cpu' not in st.session_state:
    st.session_state.cpu = 50
if 'memory' not in st.session_state:
    st.session_state.memory = 50
if 'network' not in st.session_state:
    st.session_state.network = 500
if 'power' not in st.session_state:
    st.session_state.power = 250

# Title
st.markdown("# ☁ Cloud Resource Anomaly Detection System")
st.markdown("### Real-time Anomaly Detection with Interactive Parameters")

# Train a simple anomaly detection model
@st.cache_resource
def train_model():
    np.random.seed(42)
    normal_data = np.random.normal(loc=[50, 50, 500, 250], scale=[15, 15, 150, 75], size=(500, 4))
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(normal_data)
    return iso_forest, normal_data

model, training_data = train_model()

# Sidebar for parameters
st.sidebar.markdown("## 🔫 Input Parameters")
st.sidebar.markdown("Adjust cloud resource parameters to detect anomalies")

cpu = st.sidebar.slider("CPU Usage (%)", 0, 100, st.session_state.cpu, step=1)
st.session_state.cpu = cpu

memory = st.sidebar.slider("Memory Usage (%)", 0, 100, st.session_state.memory, step=1)
st.session_state.memory = memory

network = st.sidebar.slider("Network Traffic (Mbps)", 0, 1000, st.session_state.network, step=10)
st.session_state.network = network

power = st.sidebar.slider("Power Consumption (W)", 0, 500, st.session_state.power, step=5)
st.session_state.power = power

# Make prediction
test_input = np.array([[cpu, memory, network, power]])
anomaly_score = model.decision_function(test_input)[0]
prediction = model.predict(test_input)[0]

# Main content area with columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📊 Anomaly Detection Result")
    
    # Normalize the anomaly score for display (higher negative = more anomalous)
    anomaly_percentage = min(100, max(0, int((-anomaly_score + 2) * 25)))
    
    # Create visual indicator
    if prediction == -1:
        st.markdown(f"""<div style='background-color: #ff4444; padding: 20px; border-radius: 10px; text-align: center;'>
            <h1 style='color: white; margin: 0;'>⚠️ ANOMALY DETECTED</h1>
            <p style='color: white; font-size: 18px; margin: 10px 0 0 0;'>Risk Level: {anomaly_percentage}%</p>
        </div>""", unsafe_allow_html=True)
        st.error(f"**This resource configuration is ANOMALOUS!**\n\nAnomaly Score: {-anomaly_score:.2f}")
    else:
        st.markdown(f"""<div style='background-color: #44aa44; padding: 20px; border-radius: 10px; text-align: center;'>
            <h1 style='color: white; margin: 0;'>✓ NORMAL</h1>
            <p style='color: white; font-size: 18px; margin: 10px 0 0 0;'>Risk Level: {anomaly_percentage}%</p>
        </div>""", unsafe_allow_html=True)
        st.success(f"**This resource configuration is NORMAL.**\n\nAnomaly Score: {-anomaly_score:.2f}")

with col2:
    st.markdown("## 🔍 Key Metrics")
    st.metric("CPU", f"{cpu}%")
    st.metric("Memory", f"{memory}%")
    st.metric("Network", f"{network} Mbps")
    st.metric("Power", f"{power} W")

# Visualization section
st.markdown("---")
st.markdown("## 📊 Visualization & Analysis")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.markdown("### Anomaly Score Gauge")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=anomaly_percentage,
        title={'text': "Anomaly Risk %"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightcoral"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50}}
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with viz_col2:
    st.markdown("### Current vs Normal Range")
    
    categories = ['CPU', 'Memory', 'Network', 'Power']
    current_normalized = [cpu, memory, min(network/10, 100), min(power/5, 100)]
    normal_mean = [50, 50, 50, 50]
    
    fig = go.Figure(data=[
        go.Bar(name='Normal Range', x=categories, y=normal_mean, marker_color='lightblue'),
        go.Bar(name='Current Value', x=categories, y=current_normalized, marker_color='coral')
    ])
    fig.update_layout(barmode='group', height=400)
    st.plotly_chart(fig, use_container_width=True)

# Detailed information
st.markdown("---")
st.markdown("## 📊 Detailed Information")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.info(f"""**CPU Usage**: {cpu}%
    - Normal: 30-70%
    - Status: {'🟢 OK' if 30 <= cpu <= 70 else '🔴 Alert' if cpu > 80 else '🟡 Warning'}""")

with info_col2:
    st.info(f"""**Memory Usage**: {memory}%
    - Normal: 30-70%
    - Status: {'🟢 OK' if 30 <= memory <= 70 else '🔴 Alert' if memory > 80 else '🟡 Warning'}""")

with info_col3:
    st.info(f"""**Network Traffic**: {network} Mbps
    - Normal: 300-700 Mbps
    - Status: {'🟢 OK' if 300 <= network <= 700 else '🔴 Alert' if network > 800 else '🟡 Warning'}""")

# Real-time data table
st.markdown("---")
st.markdown("## 📋 Data Summary")

data = {
    'Metric': ['CPU Usage', 'Memory Usage', 'Network Traffic', 'Power Consumption'],
    'Current Value': [f'{cpu}%', f'{memory}%', f'{network} Mbps', f'{power} W'],
    'Normal Range': ['30-70%', '30-70%', '300-700 Mbps', '200-300 W'],
    'Status': [
        '🟢 Normal' if 30 <= cpu <= 70 else '🔴 Anomaly',
        '🟢 Normal' if 30 <= memory <= 70 else '🔴 Anomaly',
        '🟢 Normal' if 300 <= network <= 700 else '🔴 Anomaly',
        '🟢 Normal' if 200 <= power <= 300 else '🔴 Anomaly'
    ]
}

df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)

st.markdown("---")
st.markdown("## 🌐 How to Use")
st.markdown("""
1. **Adjust Parameters**: Use the sliders in the sidebar to change cloud resource values
2. **View Results**: The anomaly detection result updates in real-time
3. **Analyze Data**: Check the visualizations and metrics to understand anomalies
4. **Monitor Status**: Green = Normal, Yellow = Warning, Red = Anomaly
""")
