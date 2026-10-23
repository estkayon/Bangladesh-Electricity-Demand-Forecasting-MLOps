<div align="center">

<!-- Animated Wave Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:0284c7,100:38bdf8&height=230&section=header&text=Bangladesh%20Electricity%20Demand%20Forecasting&fontSize=34&fontColor=ffffff&fontAlignY=38&desc=End-to-End%20Automated%20MLOps%20Forecasting%20Platform&descAlignY=58&descSize=18&animation=fadeIn" width="100%" />

<!-- Animated Typing SVG -->
<a href="https://github.com/estkayon/Bangladesh-Electricity-Demand-Forecasting-MLOps">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=21&duration=3000&pause=1000&color=38BDF8&center=true&vCenter=true&width=850&lines=National+%26+9-Region+Next-Day+Electricity+Demand+Forecasting;8-Day+Extended+Horizon+via+Hybrid+Bridge+Forecasting+Strategy;Automated+BPDB+%26+Open-Meteo+Data+Ingestion+with+DVC+Versioning;Experiment+Tracking+%26+Champion+Model+Registry+via+MLflow+%26+DagsHub;Production+FastAPI+Backend+on+Render+%2B+Interactive+Streamlit+UI" alt="Typing SVG" />
</a>

<br/>

<!-- Live Action Buttons -->
<p align="center">
  <a href="https://bd-electricity-demand-forecast.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/LIVE_DASHBOARD-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit Dashboard" />
  </a>
  <a href="https://bd-electricity-demand-api.onrender.com" target="_blank">
    <img src="https://img.shields.io/badge/LIVE_BACKEND-FastAPI_Render-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI Backend" />
  </a>
  <a href="https://bd-electricity-demand-api.onrender.com/docs" target="_blank">
    <img src="https://img.shields.io/badge/API_DOCS-Swagger_UI-85EA2D?style=for-the-badge&logo=swagger&logoColor=black" alt="Swagger Docs" />
  </a>
</p>

<!-- Tech Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-15B5EA?style=flat-square" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/DVC-945DD6?style=flat-square&logo=dvc&logoColor=white" />
  <img src="https://img.shields.io/badge/DagsHub-181717?style=flat-square" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=github-actions&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
</p>

</div>

An end-to-end **MLOps-powered electricity demand forecasting platform for Bangladesh**, built using real BPDB electricity demand data, weather data, machine learning, DVC, MLflow, FastAPI, Streamlit, Docker, GitHub Actions, and cloud deployment.

The system supports both **national** and **regional next-day electricity demand forecasting**, along with an **extended multi-day forecast** generated through a bridge forecasting strategy.

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🌐 Live Demo

| Component | Platform | Live Link |
| :--- | :--- | :--- |
| 📊 **Streamlit Dashboard** | Streamlit Community Cloud | [https://bd-electricity-demand-forecast.streamlit.app/](https://bd-electricity-demand-forecast.streamlit.app/) |
| ⚡ **FastAPI Backend** | Render | [https://bd-electricity-demand-api.onrender.com](https://bd-electricity-demand-api.onrender.com) |
| 📑 **FastAPI Documentation** | Swagger UI | [https://bd-electricity-demand-api.onrender.com/docs](https://bd-electricity-demand-api.onrender.com/docs) |

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🚀 Project Overview

This project builds a complete forecasting platform that:

- 📥 **Collects electricity demand data** automatically from BPDB
- 🌦️ **Collects weather data** from Open-Meteo
- 🧹 **Cleans and preprocesses** historical electricity demand
- 🛠️ **Handles missing dates** and known anomalies
- ⚙️ **Builds time-series, weather, holiday, and calendar-based features**
- 🧠 **Trains and evaluates** multiple machine learning models
- 📈 **Tracks experiments and registered models** using MLflow and DagsHub
- 🗺️ **Performs national and regional** electricity demand forecasting
- 🔮 **Generates extended multi-day forecasts**
- ⚡ **Serves predictions** through a FastAPI backend
- 🖥️ **Provides an interactive** Streamlit dashboard
- 📦 **Uses DVC** for data and pipeline versioning
- 🐳 **Uses Docker** for reproducible deployment
- 🔄 **Uses GitHub Actions** for CI and automated data refresh
- ⏱️ **Automatically updates production forecasts** when new BPDB data becomes available

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 📡 Data Sources

<table>
<tr>
<td width="50%" valign="top">

### ⚡ Bangladesh Power Development Board
Area-wise electricity demand data is collected from:  
🔗 [https://misc.bpdb.gov.bd/area-wise-demand](https://misc.bpdb.gov.bd/area-wise-demand)

The dataset covers **nine BPDB regions**:
- `Dhaka` • `Chittagong` • `Khulna`
- `Rajshahi` • `Comilla` • `Mymensingh`
- `Sylhet` • `Barisal` • `Rangpur`

*The collector is incremental and checks newly published BPDB dates automatically.*

</td>
<td width="50%" valign="top">

### 🌦️ Weather Data
Weather information is collected from **Open-Meteo** for Dhaka.

**Weather features include:**
- 🌡️ Maximum temperature
- ❄️ Minimum temperature
- 📊 Mean temperature
- 🌧️ Precipitation
- 💧 Rainfall

</td>
</tr>
</table>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🎯 Forecasting Scope

### 1️⃣ National Forecasting
Predicts next-day electricity demand for Bangladesh using aggregated regional demand.

### 2️⃣ Regional Forecasting
Produces independent next-day forecasts for all nine BPDB regions.

### 3️⃣ Extended Forecasting
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

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🧠 Machine Learning Approach

Multiple forecasting approaches were evaluated using time-series validation.

<table>
<tr>
<td width="45%" valign="top">

### 🏆 National Production Model
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

### 📊 2026 Holdout Performance
```text
MAE   : 743.59 MW
RMSE  : 1027.79 MW
MAPE  : 5.68%
```

</td>
<td width="55%" valign="top">

### 📉 National Cross-Validation Results

| Model | Mean MAPE | Status |
| :--- | :---: | :---: |
| Persistence Baseline | 5.66% | Baseline |
| Ridge Regression | 4.91% | Evaluated |
| **Tuned Ridge Regression** | **4.89%** | ✅ **Selected for Production** |
| Random Forest | 5.71% | Evaluated |
| XGBoost | 6.05% | Evaluated |

*The tuned Ridge model was selected for production.*

</td>
</tr>
</table>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🗺️ Regional Forecasting Strategy

Each region was evaluated separately. The production system selects either **Ridge Regression** or a **Persistence Baseline** depending on validation performance.

| Region | Production Strategy | Region | Production Strategy |
| :--- | :--- | :--- | :--- |
| **Dhaka** | `Ridge` | **Rangpur** | `Ridge` |
| **Chittagong** | `Ridge` | **Sylhet** | `Ridge` |
| **Comilla** | `Ridge` | **Rajshahi** | `Persistence Baseline` |
| **Khulna** | `Ridge` | **Barisal** | `Persistence Baseline` |
| **Mymensingh** | `Ridge` | | |

> 💡 **Engineering Note:** This prevents forcing a machine learning model onto regions where a simple baseline is more reliable.

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🌉 Extended Forecast Model

A separate bridge forecasting model is used for multi-day forecasting.

**Bridge features include:**
- `Lag 1`, `Lag 2`, `Lag 3`, `Lag 7`, `Lag 14`, `Lag 21`, `Lag 30`
- `Rolling 7-day demand`, `Rolling 14-day demand`, `Rolling 30-day demand`
- `Weekend indicators`, `Holiday indicators`, `Trend features`

<table>
<tr>
<td width="50%" valign="top">

### 🇧🇩 National Extended Forecast Backtest
```text
Bridge Forecast MAPE      : 7.01%
Persistence Baseline MAPE : 8.25%
```
**Production policy:**
```text
Day 1   → Anchor model
Day 2-8 → Bridge model
```

</td>
<td width="50%" valign="top">

### 🧭 Regional Extended Forecast
The regional system uses a hybrid architecture:
```text
Day 1
Region-specific anchor strategy

Day 2-8
Region-specific bridge model
```
**Overall backtest:**
```text
Hybrid Forecast MAPE      : 7.86%
Persistence Baseline MAPE : 9.33%
```

</td>
</tr>
</table>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🛠️ Feature Engineering & Data Quality

<table>
<tr>
<td width="33%" valign="top">

### ⚡ Demand Features
- Demand lag 1
- Demand lag 7
- Demand lag 14
- Rolling mean 7 days
- Rolling mean 14 days
- Rolling mean 30 days
- Load-shedding lag features

</td>
<td width="33%" valign="top">

### 📅 Calendar Features
- Day of week
- Month
- Year
- Weekend indicator
- Bangladesh public holiday indicator

*For Bangladesh, **Friday and Saturday** are treated as weekend days.*

</td>
<td width="34%" valign="top">

### 🌦️ Weather Features
- Maximum temperature
- Minimum temperature
- Mean temperature
- Precipitation
- Rainfall

</td>
</tr>
</table>

### 🛡️ Data Quality Handling
The preprocessing pipeline includes:
- Missing-date detection & complete regional date-grid creation
- Time-series interpolation & numerical type validation
- Duplicate handling & missing-value validation
- Known anomaly correction & separation of real and imputed observations

> ⚠️ The system does **not** create artificial future actual-demand rows. Future dates exist only in the forecasting layer.

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔄 Automated Pipelines & Data Collection

<table>
<tr>
<td width="50%" valign="top">

### 📥 Incremental BPDB Data Collection
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
Historical missing dates are preserved but are not repeatedly requested during every daily refresh. The parser supports BPDB table structure changes, including both older and newer layouts.

</td>
<td width="50%" valign="top">

### ⚙️ Automated Live Forecast Pipeline
The complete live refresh pipeline is implemented in:
`src/pipeline/refresh_live_forecasts.py`

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

</td>
</tr>
</table>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🏗️ MLOps Architecture

```text
                        BPDB
                         │
                         ▼
                   Data Collector
                         │
                         ▼
                        DVC
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
      Preprocessing              Open-Meteo
            │                         │
            └────────────┬────────────┘
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
                ┌────────┴────────┐
                ▼                 ▼
             FastAPI       Bridge Forecast
                │
                ▼
       Streamlit Dashboard
```

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 📦 MLflow Registry & DVC Data Versioning

<table>
<tr>
<td width="50%" valign="top">

### 🧪 MLflow and Model Registry
MLflow (hosted on **DagsHub**) is used for:
- Experiment tracking, Parameter & Metric logging
- Model comparison & Artifact storage
- Model registration & Production model aliases

Production models use the `champion` alias.

**Registered model examples:**
```text
bangladesh-electricity-demand-ridge
bangladesh-electricity-demand-bridge
bangladesh-electricity-demand-dhaka
bangladesh-electricity-demand-dhaka-bridge
bangladesh-electricity-demand-chittagong
bangladesh-electricity-demand-chittagong-bridge
```
*Regional models are independently registered and versioned.*

</td>
<td width="50%" valign="top">

### 🗂️ DVC Data Versioning
DVC (remote storage hosted through **DagsHub**) versions:
- Raw BPDB electricity data & Weather data
- Processed, Feature & Inference datasets
- National & Regional bridge forecasts

The DVC pipeline is defined in `dvc.yaml`.

**Main stages:**
```text
preprocess
build_target
build_features
build_regional_target
build_regional_features
build_inference_features
```

</td>
</tr>
</table>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🤖 CI/CD & Automated Daily Refresh

<table>
<tr>
<td width="45%" valign="top">

### ✅ Continuous Integration
GitHub Actions automatically runs CI on pushes to `main` and pull requests.

**CI checks include:**
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

</td>
<td width="55%" valign="top">

### ⏰ Automated Daily Forecast Refresh
A separate GitHub Actions workflow refreshes production data and forecasts automatically every day.

**Current schedule:**
- `09:17 AM Bangladesh Time`
- `07:17 PM Bangladesh Time`

**Workflow:**
```text
DVC Pull → BPDB Data Collection → Weather Synchronization
→ Preprocessing → Inference Feature Generation → National Forecast
→ Regional Forecasts → DVC Update → DVC Push → Git Metadata Commit
```
*If BPDB has not published new data, the latest real observation remains unchanged. When new data appears, the pipeline incorporates it automatically.*

</td>
</tr>
</table>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚡ FastAPI Backend & 🖥️ Streamlit Dashboard

<table>
<tr>
<td width="50%" valign="top">

### ⚡ FastAPI Backend
- **Base URL:** [https://bd-electricity-demand-api.onrender.com](https://bd-electricity-demand-api.onrender.com)
- **API Docs:** [https://bd-electricity-demand-api.onrender.com/docs](https://bd-electricity-demand-api.onrender.com/docs)

**Main Endpoints:**
```http
GET /
GET /health
GET /regions
GET /predict/latest
GET /predict/extended
GET /predict/date
GET /history
GET /model/info
```
**Example:** `GET /predict/latest?region=National`

**The API supports:**
- National & Regional forecasting
- Live next-day & Extended multi-day forecasting
- Historical evaluation & Model metadata

</td>
<td width="50%" valign="top">

### 🖥️ Streamlit Dashboard
- **Live Dashboard:** [https://bd-electricity-demand-forecast.streamlit.app/](https://bd-electricity-demand-forecast.streamlit.app/)

**Dashboard features include:**
- National and regional selection
- Latest observed demand
- Next-day prediction
- Extended forecast
- Interactive charts
- Historical prediction lookup
- Actual vs predicted comparison
- Model information
- Forecast status information

### ☁️ Cloud Deployment
- **Backend:** Deployed on **Render** (pulls required DVC data from remote storage before starting the API).
- **Frontend:** Deployed on **Streamlit Community Cloud** (communicates with the public Render API using an environment-configured `API_BASE_URL`).

</td>
</tr>
</table>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 📂 Project Structure

<details open>
<summary><b>Click to expand/collapse repository file tree</b></summary>

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
</details>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 💻 Local Setup & Execution

### 1. Clone the Repository
```bash
git clone [https://github.com/estkayon/Bangladesh-Electricity-Demand-Forecasting-MLOps.git](https://github.com/estkayon/Bangladesh-Electricity-Demand-Forecasting-MLOps.git)
cd Bangladesh-Electricity-Demand-Forecasting-MLOps
```

### 2. Create the Environment & Install Dependencies
```bash
conda create -n bdpower python=3.11
conda activate bdpower
pip install -r requirements.txt
```

### 3. Pull DVC Data
```bash
dvc pull -r origin
```

### 4. Configure Environment Variables
Create a `.env` file (Do not commit `.env` to Git):
```env
MLFLOW_TRACKING_URI=your_mlflow_tracking_uri
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

### 5. Run the Application & Pipelines

<table>
<tr>
<td width="50%" valign="top">

**Run the API:**
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
Open: `http://127.0.0.1:8000/docs`

**Run the Dashboard:**
```bash
streamlit run dashboard.py
```

**Docker Build & Run:**
```bash
docker build -t bd-electricity-api .
docker run --rm -p 8000:8000 --env-file .env bd-electricity-api
```
- Local API: `http://127.0.0.1:8000`
- Swagger documentation: `http://127.0.0.1:8000/docs`

</td>
<td width="50%" valign="top">

**Run the Live Forecast Pipeline:**
```bash
python -m src.pipeline.refresh_live_forecasts
```
*Automatically performs:*
`BPDB data collection → preprocessing → weather synchronization → inference feature generation → national forecast generation → regional forecast generation`

**Reproduce the DVC Pipeline:**
```bash
dvc repro
dvc status
dvc push -r origin
```

</td>
</tr>
</table>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🧰 Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Programming** | Python |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Experiment Tracking** | MLflow, DagsHub |
| **Data Versioning** | DVC |
| **API** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit, Plotly |
| **DevOps / MLOps** | Docker, Git, GitHub, GitHub Actions, DVC, MLflow |
| **Deployment** | Render, Streamlit Community Cloud, DagsHub |

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🌟 Key Engineering Highlights & Future Improvements

<table>
<tr>
<td width="50%" valign="top">

### 🌟 Key Engineering Highlights
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

</td>
<td width="50%" valign="top">

### 🔮 Future Improvements
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

</td>
</tr>
</table>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 👨‍💻 Author

**Md. Estiak Rahman Ayon**  
*Computer Science graduate interested in:*
- Machine Learning • MLOps • Artificial Intelligence • Data Science • Production ML systems • Software Engineer

---

## ⚠️ Disclaimer

> Electricity demand information is collected from publicly available BPDB data sources. Forecasts generated by this system should not be considered official BPDB forecasts or operational recommendations.

<!-- Animated Footer -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:0284c7,100:38bdf8&height=120&section=footer" width="100%" />