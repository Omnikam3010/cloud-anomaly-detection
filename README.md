# Cloud Anomaly Detection - Streamlit App

☁️ **A machine learning-powered system for detecting anomalies in cloud resource usage.**

## Overview

This project deploys a trained ML model (LightGBM, Random Forest, or XGBoost) to detect anomalies in cloud resource configurations. The application is built with **Streamlit** for an interactive web interface and uses **joblib** for model loading.

## Features

- ✅ **Single Prediction**: Analyze individual cloud resource configurations in real-time
- ✅ **Batch Prediction**: Upload CSV files for bulk anomaly detection
- ✅ **Probability Scoring**: Get anomaly probability percentages with visual meters
- ✅ **User-Friendly Interface**: Clean, professional UI with intuitive controls
- ✅ **Model Pipeline**: Includes preprocessing (StandardScaler + OneHotEncoder)
- ✅ **CSV Export**: Download predictions for further analysis

## Project Structure

```
cloud-anomaly-detection/
├── streamlit_app.py                 # Main Streamlit application
├── best_cloud_anomaly_model.pkl     # Trained ML model pipeline
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Git ignore file
```

## Input Features

### Numeric Features
- `cpu_usage` - CPU utilization percentage
- `memory_usage` - Memory utilization percentage
- `network_traffic` - Network traffic in MB/s
- `power_consumption` - Power consumption in watts
- `num_executed_instructions` - Number of executed instructions (×10³)
- `execution_time` - Execution time in milliseconds
- `energy_efficiency` - Energy efficiency score
- `hour`, `day`, `month`, `weekday` - Temporal features

### Categorical Features
- `task_type` - Type of task (compute, storage, database, analytics, other, unknown)
- `task_priority` - Task priority level (low, medium, high)
- `task_status` - Current task status (running, pending, completed, failed, waiting, unknown)

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Omnikam3010/cloud-anomaly-detection.git
cd cloud-anomaly-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Single Prediction
1. Enter resource metrics and task details in the input form
2. Click **"Predict anomaly"** button
3. View the prediction result and anomaly probability

### Batch Prediction
1. Prepare a CSV file with the required feature columns
2. Upload the CSV in the "Batch Prediction" section
3. Click **"Run batch anomaly detection"**
4. Download the results with predictions and probabilities

## Model Information

- **Architecture**: LightGBM with preprocessing pipeline (StandardScaler + OneHotEncoder)
- **Input**: 13 features (10 numeric + 3 categorical)
- **Output**: Binary classification (0 = Normal, 1 = Anomaly)
- **Performance**: Includes probability scores for confidence assessment

## Deployment

### Deploy on Streamlit Cloud
1. Push the repository to GitHub
2. Visit [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app" and select this repository
4. Choose `main` branch and `streamlit_app.py` as the main file
5. Deploy!

### Deploy on Other Platforms
- **Render**: Add `Procfile` and deploy via Git
- **AWS/Azure**: Use containerization with Docker
- **Heroku**: Similar approach to Render

## Sample CSV Format

For batch predictions, prepare a CSV with this structure:

```csv
cpu_usage,memory_usage,network_traffic,power_consumption,num_executed_instructions,execution_time,energy_efficiency,hour,day,month,weekday,task_type,task_priority,task_status
50.5,45.2,100.0,250.5,5000.0,50.0,45.5,12,15,6,2,compute,medium,running
75.3,80.1,250.0,350.0,8000.0,120.0,30.0,9,20,6,1,storage,high,running
```

## Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- joblib
- scikit-learn
- lightgbm (or xgboost/random-forest depending on model)

## MLOps & DevOps Integration

This project includes:
- ✅ Trained model serialization (joblib)
- ✅ Reproducible predictions from saved pipeline
- ✅ CSV batch inference support
- ✅ Probability scoring for decision confidence
- ✅ Streamlit Cloud deployment ready

## File Descriptions

### `streamlit_app.py`
Main application file containing:
- Model loading with caching
- Single prediction interface
- Batch CSV processing
- Result visualization
- Download functionality

### `best_cloud_anomaly_model.pkl`
Serialized ML pipeline including:
- Feature preprocessing (StandardScaler + OneHotEncoder)
- Trained classifier (LightGBM/Random Forest/XGBoost)
- Feature names and transformers

### `requirements.txt`
All Python dependencies for reproducible environments

## Troubleshooting

**Model file not found?**
- Ensure `best_cloud_anomaly_model.pkl` is in the same directory as `streamlit_app.py`

**CSV prediction errors?**
- Verify CSV has all required columns
- Check column names match the training data
- Categorical values must be from the expected set

**Slow predictions?**
- First run loads the model (normal with @st.cache_resource)
- Subsequent predictions are cached and fast

## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - Feel free to use this project for educational and commercial purposes.

## Contact & Support

For questions or issues:
- Create an issue on [GitHub Issues](https://github.com/Omnikam3010/cloud-anomaly-detection/issues)
- Contact: [Your Name]
- Email: [Your Email]

---

**Last Updated**: November 2025
**Status**: ✅ Production Ready
