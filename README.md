# 🚲 Bike Rental Demand Prediction & Production Monitoring System

An end-to-end Machine Learning and MLOps project that predicts hourly bike rental demand using environmental, temporal, and weather factors, deployed with an interactive production dashboard and real-time telemetry monitoring.

---

## 📌 Project Overview
Urban bike-sharing systems play a crucial role in city mobility and sustainability. Accurately forecasting rental demand enables operators to optimize bike distribution, reduce logistics costs, and improve customer satisfaction.

This repository implements the complete data science lifecycle:
1. **Data Ingestion & Cleaning:** Outlier detection (IQR), missing value imputation, type casting.
2. **Exploratory Data Analysis (EDA):** Seasonality analysis, commute pattern discovery, correlation heatmaps.
3. **Feature Engineering:** Commute indicators, datetime transformations, one-hot encoding.
4. **Model Building & Baseline Evaluation:** 6 Regression algorithms evaluated on chronologically split test data.
5. **Hyperparameter Tuning:** Cross-validation (`TimeSeriesSplit`) with Grid Search & Random Search balancing complexity and generalization.
6. **Model Serialization & Deployment:** Automated model packaging and export into production artifacts.
7. **Streamlit Production Dashboard:** Real-time single & batch inference, telemetry monitoring (rolling RMSE, drift, residuals), and on-demand model retraining.

---

## 🏆 Model Performance Summary

After hyperparameter tuning across 6 machine learning architectures using 5-fold TimeSeries cross-validation:

| Model | Search Technique | Test RMSE | Test MAE | Test $R^2$ Score | Status |
|---|---|---|---|---|---|
| **LightGBM** | `RandomizedSearchCV` | **71.05** | **47.78** | **0.8962** | 🏆 **Deployed to Production** |
| **Random Forest** | `RandomizedSearchCV` | ~74.20 | ~49.80 | ~0.8870 | Evaluated |
| **XGBoost** | `RandomizedSearchCV` | ~75.10 | ~50.40 | ~0.8845 | Evaluated |
| **Gradient Boosting** | `RandomizedSearchCV` | ~78.50 | ~53.10 | ~0.8740 | Evaluated |
| **Decision Tree** | `GridSearchCV` | ~85.30 | ~58.20 | ~0.8510 | Evaluated |
| **KNN (with Scaler Pipeline)**| `GridSearchCV` | ~88.90 | ~60.40 | ~0.8390 | Evaluated |

---

## 🛠️ Repository Structure

```
├── Dataset.csv                   # Raw bike rental hourly dataset
├── ds_project.py                 # Full ML pipeline: EDA, modeling, tuning, export
├── app.py                        # Streamlit production inference & monitoring dashboard
├── requirements.txt              # Production Python dependencies
├── .gitignore                    # Ignored artifacts & environment caches
├── README.md                     # Project documentation
└── models/                       # Exported model & monitoring artifacts
    ├── best_bike_rental_model.joblib  # Serialized LightGBM model artifact
    ├── deployment_metadata.json       # Schema, parameters, and baseline metrics
    ├── monitoring_baseline.csv        # Baseline historical predictions for drift tracking
    └── live_prediction_logs.csv       # Production inference audit log
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/anshul-1107/Bike-Rental.git
cd Bike-Rental
```

### 2. Set Up Environment & Install Dependencies
```bash
# Create and activate a virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Run the ML Pipeline (Training, Tuning & Export)
```bash
python ds_project.py
```
This executes the complete data pipeline, tunes all 6 models, prints cross-validation metrics, plots comparison charts, and exports the champion model (`LightGBM`) to `models/`.

### 4. Launch the Production Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 💻 Streamlit Production Dashboard Features

1. **⚡ Single Real-Time Inference:**
   - Sliders and inputs for weather conditions (temperature in $^\circ\text{C}$, humidity %, windspeed), temporal settings (hour, month, weekday), and calendar factors (season, holiday, working day).
   - Generates instantaneous demand predictions with categorized badges (*Low, Moderate, High, Peak Surge*) and confidence bounds.
   - Logs query inputs and predictions directly to the monitoring audit trail.

2. **📂 Batch CSV Inference:**
   - Upload bulk CSV files.
   - Generates predictions across all records with demand preview plots.
   - Download the scored results as a timestamped CSV.

3. **📈 Continuous Telemetry & Drift Monitoring:**
   - Real-time KPIs: Production RMSE, MAE, $R^2$, and dynamic health status (*Healthy 🟢, Moderate Drift 🟡, Degraded 🔴*).
   - Visualizations: Actual vs. Predicted over time, 24-hour moving RMSE, residual normality check ($E[\epsilon] \approx 0$), and error-vs-volume scatter.
   - Ground truth reconciliation simulator to test concept drift detection.

4. **🔄 Model Management & Automated Retraining:**
   - View active production model metadata and hyperparameters.
   - Trigger retraining on accumulated live logs with candidate champion validation.
   - One-click model promotion to update production versions seamlessly.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Fork or push this repository to GitHub (`anshul-1107/Bike-Rental`).
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**, select `anshul-1107/Bike-Rental`, and set **Main file path** to `app.py`.
4. Click **"Deploy"** — your live dashboard will be hosted online!

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).

