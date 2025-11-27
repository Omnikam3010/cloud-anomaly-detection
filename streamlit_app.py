import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Cloud Anomaly Detection", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('best_cloud_anomaly_model.pkl')

model = load_model()

st.title("\u2601 Cloud Anomaly Detection System")
st.markdown("Detect anomalies in cloud resource usage using ML")

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Info"])

with tab1:
    st.header("Analyze Individual Cloud Resource")
    col1, col2 = st.columns(2)
    
    with col1:
        cpu_usage = st.slider("CPU Usage (%)", 0.0, 100.0, 50.0)
        memory_usage = st.slider("Memory Usage (%)", 0.0, 100.0, 50.0)
        network_traffic = st.slider("Network Traffic (Mbps)", 0.0, 1000.0, 500.0)
        power_consumption = st.slider("Power Consumption (W)", 0.0, 500.0, 250.0)
    
    with col2:
        num_instructions = st.slider("Num Executed Instructions", 0.0, 10000.0, 5000.0)
        execution_time = st.slider("Execution Time (ms)", 0.0, 100.0, 50.0)
        energy_efficiency = st.slider("Energy Efficiency", 0.0, 1.0, 0.5)
        task_type = st.selectbox("Task Type", ["compute", "io", "network"])
    
    task_priority = st.selectbox("Task Priority", ["low", "medium", "high"])
    task_status = st.selectbox("Task Status", ["completed", "running", "waiting"])
    hour = st.slider("Hour", 0, 23, 12)
    day = st.slider("Day", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)
    weekday = st.slider("Weekday (0=Mon)", 0, 6, 3)
    
    if st.button("Predict", key="predict_single"):
        input_data = pd.DataFrame([{
            'cpu_usage': cpu_usage, 'memory_usage': memory_usage,
            'network_traffic': network_traffic, 'power_consumption': power_consumption,
            'num_executed_instructions': num_instructions, 'execution_time': execution_time,
            'energy_efficiency': energy_efficiency, 'task_type': task_type,
            'task_priority': task_priority, 'task_status': task_status,
            'hour': hour, 'day': day, 'month': month, 'weekday': weekday
        }])
        
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                if prediction == 1:
                    st.error(f"\u26a0 ANOMALY DETECTED", icon="⚠")
                else:
                    st.success("\u2713 Normal", icon="✓")
            with col_pred2:
                st.metric("Anomaly Probability", f"{probability*100:.2f}%")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab2:
    st.header("Batch Anomaly Detection")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} records")
        
        if st.button("Predict Batch", key="predict_batch"):
            try:
                predictions = model.predict(df)
                probabilities = model.predict_proba(df)[:, 1]
                
                results_df = pd.DataFrame({
                    'Prediction': ['Anomaly' if p == 1 else 'Normal' for p in predictions],
                    'Anomaly_Probability': probabilities
                })
                
                st.dataframe(results_df)
                st.download_button("Download Results", results_df.to_csv(index=False), "results.csv")
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab3:
    st.header("Model Information")
    st.write("**Model Type:** LightGBM Classifier")
    st.write("**Features:** CPU Usage, Memory, Network Traffic, Power, Execution Time, Energy Efficiency, Task Type/Priority/Status, Time features")
    st.write("**Accuracy:** ~75%")
    st.write("**Use Case:** Detect anomalies in cloud resource configurations")
