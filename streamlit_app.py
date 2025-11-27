import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go

st.set_page_config(page_title="Cloud Anomaly Detection", layout="wide")
st.title("Cloud Resource Anomaly Detection")

st.sidebar.header("Parameters")
cpu = st.sidebar.slider("CPU (%)", 0, 100, 50)
mem = st.sidebar.slider("Memory (%)", 0, 100, 50)
net = st.sidebar.slider("Network (Mbps)", 0, 1000, 500)
pow = st.sidebar.slider("Power (W)", 0, 500, 250)

np.random.seed(42)
data = np.random.normal([50, 50, 500, 250], [15, 15, 150, 75], 300).reshape(-1, 4)
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(data)

input_val = np.array([[cpu, mem, net, pow]])
pred = model.predict(input_val)[0]
score = model.decision_function(input_val)[0]
risk = max(0, min(100, int((-score + 2) * 25)))

col1, col2 = st.columns([2, 1])
with col1:
    if pred == -1:
        st.error(f"ANOMALY DETECTED - Risk: {risk}%")
    else:
        st.success(f"NORMAL - Risk: {risk}%")

with col2:
    st.metric("CPU", f"{cpu}%")
    st.metric("Memory", f"{mem}%")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk,
    title="Risk %",
    gauge={
        "axis": {"range": [0, 100]},
        "steps": [
            {"range": [0, 30], "color": "lightgreen"},
            {"range": [30, 70], "color": "yellow"},
            {"range": [70, 100], "color": "lightcoral"}
        ]
    }
))
st.plotly_chart(fig)

df = pd.DataFrame({
    "Metric": ["CPU", "Memory", "Network", "Power"],
    "Value": [f"{cpu}%", f"{mem}%", f"{net}", f"{pow}"]
})
st.dataframe(df)
