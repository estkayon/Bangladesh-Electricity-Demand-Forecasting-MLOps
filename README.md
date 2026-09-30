# Bangladesh Electricity Demand Forecasting MLOps

An end-to-end **MLOps-powered electricity demand forecasting platform for Bangladesh**, built using real BPDB electricity demand data, weather data, machine learning, DVC, MLflow, FastAPI, Streamlit, Docker, GitHub Actions, and cloud deployment.

The system supports both **national** and **regional next-day electricity demand forecasting**, along with an **extended multi-day forecast** generated through a bridge forecasting strategy.

---

## Live Demo

- **Streamlit Dashboard:** https://bd-electricity-demand-forecast.streamlit.app/
- **FastAPI Backend:** https://bd-electricity-demand-api.onrender.com
- **FastAPI Documentation:** https://bd-electricity-demand-api.onrender.com/docs

---

## Project Overview

This project builds a complete forecasting platform that:

- Collects electricity demand data automatically from BPDB
- Collects weather data from Open-Meteo
- Cleans and preprocesses historical electricity demand
- Handles missing dates and known anomalies
- Builds time-series, weather, holiday, and calendar-based features
- Trains and evaluates multiple machine learning models
- Tracks experiments and registered models using MLflow and DagsHub
- Performs national and regional electricity demand forecasting
- Generates extended multi-day forecasts
- Serves predictions through a FastAPI backend
- Provides an interactive Streamlit dashboard
- Uses DVC for data and pipeline versioning
- Uses Docker for reproducible deployment
- Uses GitHub Actions for CI and automated data refresh
- Automatically updates production forecasts when new BPDB data becomes available

---

## Data Sources

### Bangladesh Power Development Board

Area-wise electricity demand data is collected from:

https://misc.bpdb.gov.bd/area-wise-demand

The dataset covers nine BPDB regions:

- Dhaka
- Chittagong
- Khulna
- Rajshahi
- Comilla
- Mymensingh
- Sylhet
- Barisal
- Rangpur

The collector is incremental and checks newly published BPDB dates automatically.

### Weather Data

Weather information is collected from **Open-Meteo** for Dhaka.

Weather features include:

- Maximum temperature
- Minimum temperature
- Mean temperature
- Precipitation
- Rainfall

---

## Forecasting Scope

### National Forecasting

Predicts next-day electricity demand for Bangladesh using aggregated regional demand.

### Regional Forecasting

Produces independent next-day forecasts for all nine BPDB regions.

### Extended Forecasting

The production system generates an **8-day forecast horizon**:

```text
Day 1
Real observed demand
        ↓
Anchor forecasting model
        ↓
Next-day prediction

Day 2 - Day 8
Previous forecasted values
        ↓
Bridge forecasting model
        ↓
Recursive multi-day forecast
```

---

## Machine Learning Approach

Multiple forecasting approaches were evaluated using time-series validation.

### National Production Model

The national production model uses:

```text
StandardScaler
+
Ridge Regression
```

Selected parameter:

```text
alpha = 0.01
```

### National Cross-Validation Results

| Model | Mean MAPE |
|---|---:|
| Persistence Baseline | 5.66% |
| Ridge Regression | 4.91% |
| Tuned Ridge Regression | 4.89% |
| Random Forest | 5.71% |
| XGBoost | 6.05% |

The tuned Ridge model was selected for production.

### 2026 Holdout Performance

```text
MAE   : 743.59 MW
RMSE  : 1027.79 MW
MAPE  : 5.68%
```

---

## Regional Forecasting Strategy

Each region was evaluated separately. The production system selects either **Ridge Regression** or a **Persistence Baseline** depending on validation performance.

| Region | Production Strategy |
|---|---|
| Dhaka | Ridge |
| Chittagong | Ridge |
| Comilla | Ridge |
| Khulna | Ridge |
| Mymensingh | Ridge |
| Rangpur | Ridge |
| Sylhet | Ridge |
| Rajshahi | Persistence Baseline |
| Barisal | Persistence Baseline |

This prevents forcing a machine learning model onto regions where a simple baseline is more reliable.

---

## Extended Forecast Model

A separate bridge forecasting model is used for multi-day forecasting.

Bridge features include:

- Lag 1
- Lag 2
- Lag 3
- Lag 7
- Lag 14
- Lag 21
- Lag 30
- Rolling 7-day demand
- Rolling 14-day demand
- Rolling 30-day demand
- Weekend indicators
- Holiday indicators
- Trend features

### National Extended Forecast Backtest

```text
Bridge Forecast MAPE      : 7.01%
Persistence Baseline MAPE : 8.25%
```

Production policy:

```text
Day 1   → Anchor model
Day 2-8 → Bridge model
```

### Regional Extended Forecast

The regional system uses a hybrid architecture:

```text
Day 1
Region-specific anchor strategy

Day 2-8
Region-specific bridge model
```

Overall backtest:

```text
Hybrid Forecast MAPE      : 7.86%
Persistence Baseline MAPE : 9.33%
```

---

## Feature Engineering

### Demand Features

- Demand lag 1
- Demand lag 7
- Demand lag 14
- Rolling mean 7 days
- Rolling mean 14 days
- Rolling mean 30 days
- Load-shedding lag features

### Calendar Features

- Day of week
- Month
- Year
- Weekend indicator
- Bangladesh public holiday indicator

For Bangladesh, **Friday and Saturday** are treated as weekend days.

### Weather Features

- Maximum temperature
- Minimum temperature
- Mean temperature
- Precipitation
- Rainfall

---

## Data Quality Handling

The preprocessing pipeline includes:

- Missing-date detection
- Complete regional date-grid creation
- Time-series interpolation
- Numerical type validation
- Duplicate handling
- Missing-value validation
- Known anomaly correction
- Separation of real and imputed observations

The system does **not** create artificial future actual-demand rows. Future dates exist only in the forecasting layer.

---

## Incremental BPDB Data Collection

The collector avoids redownloading the full historical dataset every day.

```text
Load existing raw data
        ↓
Find latest real BPDB date
        ↓
Probe newly available dates
        ↓
Download valid new records
        ↓
Update missing-date tracking
```

Historical missing dates are preserved but are not repeatedly requested during every daily refresh.

The parser supports BPDB table structure changes, including both older and newer layouts.

---

## Automated Live Forecast Pipeline

The complete live refresh pipeline is implemented in:

```text
src/pipeline/refresh_live_forecasts.py
```

Pipeline:

```text
1. Collect latest BPDB data
        ↓
2. Preprocess electricity demand
        ↓
3. Synchronize weather data
        ↓
4. Build live inference features
        ↓
5. Generate national extended forecast
        ↓
6. Generate regional extended forecasts
```

---

## MLOps Architecture

```text
                        BPDB
                          │
                          ▼
                    Data Collector
                          │
                          ▼
                         DVC
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
      Preprocessing               Open-Meteo
            │                           │
            └─────────────┬─────────────┘
                          ▼
                  Feature Engineering
                          │
                          ▼
                   Model Training
                          │
                          ▼
                 MLflow + DagsHub
                          │
                          ▼
                    Model Registry
                          │
                          ▼
                Production Forecasting
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
             FastAPI          Bridge Forecast
                │
                ▼
        Streamlit Dashboard
```

---

## MLflow and Model Registry

MLflow is used for:

- Experiment tracking
- Parameter logging
- Metric logging
- Model comparison
- Model artifact storage
- Model registration
- Production model aliases

DagsHub hosts the MLflow tracking server and model registry.

Production models use the:

```text
champion
```

alias.

Registered model examples:

```text
bangladesh-electricity-demand-ridge
bangladesh-electricity-demand-bridge
bangladesh-electricity-demand-dhaka
bangladesh-electricity-demand-dhaka-bridge
bangladesh-electricity-demand-chittagong
bangladesh-electricity-demand-chittagong-bridge
```

Regional models are independently registered and versioned.

---

## DVC Data Versioning

DVC is used to version:

- Raw BPDB electricity data
- Weather data
- Processed datasets
- Feature datasets
- Inference datasets
- National bridge forecasts
- Regional bridge forecasts

The DVC pipeline is defined in:

```text
dvc.yaml
```

Main stages:

```text
preprocess
build_target
build_features
build_regional_target
build_regional_features
build_inference_features
```

DVC remote storage is hosted through DagsHub.

---

## Continuous Integration

GitHub Actions automatically runs CI on pushes to `main` and pull requests.

CI checks include:

```text
Repository checkout
        ↓
Python installation
        ↓
Dependency installation
        ↓
Python syntax validation
        ↓
Core dependency import tests
        ↓
FastAPI application import test
        ↓
DVC pipeline validation
```

---

## Automated Daily Forecast Refresh

A separate GitHub Actions workflow refreshes production data and forecasts automatically every day.

Current schedule:

```text
09:17 AM Bangladesh Time
07:17 PM Bangladesh Time
```

Workflow:

```text
DVC Pull
    ↓
BPDB Data Collection
    ↓
Weather Synchronization
    ↓
Preprocessing
    ↓
Inference Feature Generation
    ↓
National Forecast
    ↓
Regional Forecasts
    ↓
DVC Update
    ↓
DVC Push
    ↓
Git Metadata Commit
```

If BPDB has not published new data, the latest real observation remains unchanged. When new data appears, the pipeline incorporates it automatically.

---

## FastAPI Backend

The production API is built with FastAPI.

### Base URL

https://bd-electricity-demand-api.onrender.com

### API Documentation

https://bd-electricity-demand-api.onrender.com/docs

### Main Endpoints

```text
GET /
GET /health
GET /regions
GET /predict/latest
GET /predict/extended
GET /predict/date
GET /history
GET /model/info
```

Example:

```text
GET /predict/latest?region=National
```

The API supports:

- National forecasting
- Regional forecasting
- Live next-day prediction
- Extended multi-day forecasting
- Historical evaluation
- Model metadata

---

## Streamlit Dashboard

Live dashboard:

https://bd-electricity-demand-forecast.streamlit.app/

Dashboard features include:

- National and regional selection
- Latest observed demand
- Next-day prediction
- Extended forecast
- Interactive charts
- Historical prediction lookup
- Actual vs predicted comparison
- Model information
- Forecast status information

---

## Docker

The FastAPI backend is containerized with Docker.

Build:

```bash
docker build -t bd-electricity-api .
```

Run:

```bash
docker run --rm -p 8000:8000 --env-file .env bd-electricity-api
```

Local API:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Cloud Deployment

### Backend

FastAPI is deployed on **Render**.

The production container pulls required DVC data from remote storage before starting the API.

### Frontend

The dashboard is deployed on **Streamlit Community Cloud**.

The dashboard communicates with the public Render API using an environment-configured `API_BASE_URL`.

---

## Project Structure

```text
Bangladesh-Electricity-Demand-Forecasting-MLOps/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── live_forecast_refresh.yml
│
├── data/
│   ├── raw/
│   │   ├── area_wise_demand.csv
│   │   ├── dhaka_weather.csv
│   │   └── missing_dates.csv
│   │
│   └── processed/
│       ├── area_wise_demand_processed.csv
│       ├── national_daily_demand.csv
│       ├── regional_daily_demand.csv
│       ├── model_features.csv
│       ├── regional_model_features.csv
│       ├── national_inference_features.csv
│       ├── regional_inference_features.csv
│       ├── bridge_forecast.csv
│       └── regional_bridge_forecast.csv
│
├── src/
│   ├── api/
│   │   └── app.py
│   ├── data/
│   │   ├── collect_data.py
│   │   ├── collect_weather.py
│   │   ├── preprocess.py
│   │   ├── check_missing_dates.py
│   │   └── retry_missing_dates.py
│   ├── features/
│   │   ├── build_target.py
│   │   ├── build_features.py
│   │   ├── build_regional_target.py
│   │   ├── build_regional_features.py
│   │   └── build_inference_features.py
│   ├── models/
│   │   ├── train.py
│   │   ├── final_train.py
│   │   ├── train_regional.py
│   │   ├── train_bridge_model.py
│   │   ├── train_regional_bridge_models.py
│   │   ├── generate_bridge_forecast.py
│   │   ├── generate_regional_bridge_forecast.py
│   │   ├── time_series_validation.py
│   │   ├── regional_time_series_validation.py
│   │   └── ...
│   ├── pipeline/
│   │   └── refresh_live_forecasts.py
│   └── logger.py
│
├── dashboard.py
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── requirements.txt
└── README.md
```

---

## Local Setup

### Clone the Repository

```bash
git clone https://github.com/estkayon/Bangladesh-Electricity-Demand-Forecasting-MLOps.git
cd Bangladesh-Electricity-Demand-Forecasting-MLOps
```

### Create the Environment

```bash
conda create -n bdpower python=3.11
conda activate bdpower
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Pull DVC Data

```bash
dvc pull -r origin
```

### Configure Environment Variables

Create a `.env` file:

```env
MLFLOW_TRACKING_URI=your_mlflow_tracking_uri
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

Do not commit `.env` to Git.

---

## Run the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Run the Dashboard

```bash
streamlit run dashboard.py
```

---

## Run the Live Forecast Pipeline

```bash
python -m src.pipeline.refresh_live_forecasts
```

This automatically performs:

```text
BPDB data collection
→ preprocessing
→ weather synchronization
→ inference feature generation
→ national forecast generation
→ regional forecast generation
```

---

## Reproduce the DVC Pipeline

```bash
dvc repro
```

Check status:

```bash
dvc status
```

Push updated data:

```bash
dvc push -r origin
```

---

## Technology Stack

### Programming
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost

### Experiment Tracking
- MLflow
- DagsHub

### Data Versioning
- DVC

### API
- FastAPI
- Uvicorn

### Dashboard
- Streamlit
- Plotly

### DevOps / MLOps
- Docker
- Git
- GitHub
- GitHub Actions
- DVC
- MLflow

### Deployment
- Render
- Streamlit Community Cloud
- DagsHub

---

## Key Engineering Highlights

This project goes beyond model training and demonstrates a complete production-oriented machine learning workflow:

- Real-world public data ingestion
- Incremental data collection
- Automated missing-date tracking
- Data anomaly correction
- Time-series validation
- Baseline comparison
- Region-specific model selection
- Model registry
- Multi-day recursive forecasting
- DVC pipeline reproducibility
- Remote DVC storage
- Containerized API deployment
- Continuous integration
- Scheduled production refresh
- Public REST API
- Interactive cloud dashboard

---

## Future Improvements

Potential future improvements include:

- Automated model retraining based on new data volume
- Model drift detection
- Forecast uncertainty intervals
- Additional weather stations
- Region-specific weather features
- Advanced ensemble forecasting
- Transformer-based time-series forecasting
- Monitoring and alerting
- Dedicated cloud database
- More granular electricity demand data

---

## Author

**Md. Estiak Rahman Ayon**

Computer Science graduate interested in:

- Machine Learning
- MLOps
- Artificial Intelligence
- Data Science
- Production ML systems
- Software Engineer

---

## Disclaimer

Electricity demand information is collected from publicly available BPDB data sources. Forecasts generated by this system should not be considered official BPDB forecasts or operational recommendations.
