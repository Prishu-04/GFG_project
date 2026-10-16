# 🚦 Dynamic Urban Traffic Prediction

## 📌 Project Overview

**Dynamic Urban Traffic Prediction** is a machine learning-based project designed to predict traffic congestion across roads and intersections using historical traffic data.

The system analyzes historical traffic information such as vehicle count, average speed, vehicle density, road capacity, traffic ratios, and congestion-related measurements to identify and predict traffic conditions.

The basic system focuses on historical traffic data and machine learning. **Weather and event information can be added as optional features**, while advanced spatial modelling using **Graph Neural Networks (GNN)** can be considered as a future extension.

The project is designed to start with a simple working prediction system and gradually extend it with a database, backend API, frontend dashboard, and map-based visualization.

---

# 🎯 Project Objective

The main objective of this project is to develop a system that can **predict traffic congestion levels across roads or intersections using historical traffic data**.

The system aims to:

* Analyze historical urban traffic data.
* Understand traffic patterns and congestion conditions.
* Identify important traffic-related features.
* Use machine learning to predict congestion.
* Classify traffic conditions into different congestion levels.
* Provide a foundation for displaying predictions through a web application.
* Support future integration of weather and event information.
* Provide an optional foundation for spatial traffic modelling using GNN.

---

# 🚦 Problem Statement

Urban traffic congestion is a common problem caused by increasing vehicle volume, limited road capacity, changing vehicle speeds, and variations in traffic conditions throughout the day.

Traditional traffic monitoring mainly describes the **current** traffic situation. A prediction system can instead use historical traffic patterns to estimate future congestion conditions.

This project aims to use historical traffic data and machine learning techniques to develop a system capable of predicting traffic congestion levels.

---

# 💡 Proposed Solution

The proposed system follows a step-by-step machine learning workflow:

```text
Historical Traffic Data
          ↓
Data Understanding
          ↓
Data Cleaning
          ↓
Feature Preparation
          ↓
Machine Learning Model
          ↓
Congestion Prediction
          ↓
Backend API
          ↓
Frontend Dashboard
          ↓
Map-based Visualization
```

The initial machine learning approach will focus on **XGBoost**, with **LSTM** considered as an optional advanced time-series model.

---

# 📊 Dataset

The project uses the **Urban Traffic Congestion Data** dataset.

### Dataset Information

| Property              | Details                        |
| --------------------- | ------------------------------ |
| Dataset Name          | `traffic_dataset`              |
| Source                | Kaggle                         |
| File Format           | CSV                            |
| Number of Rows        | 4,354                          |
| Number of Columns     | 14                             |
| Prediction Objective  | Traffic Congestion             |
| Congestion Categories | Low, Moderate, High, Very High |

### Dataset Source

The dataset is available on Kaggle:

`https://www.kaggle.com/datasets/chanchal27/urban-traffic-congestion-data`

---

# 📋 Dataset Columns

The dataset contains the following 14 columns:

| No. | Column Name                                  |
| --: | -------------------------------------------- |
|   1 | `Timestamp`                                  |
|   2 | `IR Presence (Lane 1-4)`                     |
|   3 | `Vehicle Count`                              |
|   4 | `Avg Speed (km/h)`                           |
|   5 | `Vehicle Types Detected`                     |
|   6 | `Vehicle Density (%)`                        |
|   7 | `Saturation Flow Rate(veh/hr/lane)`          |
|   8 | `Volume to Saturation Lane Traffic ratio(%)` |
|   9 | `FreeFlowSpeed (km/h)`                       |
|  10 | `TSR`                                        |
|  11 | `VLSR`                                       |
|  12 | `Speed Factor`                               |
|  13 | `CI`                                         |
|  14 | `Congestion Level`                           |

---

# 📖 Dataset Column Description

| Column No. | Column Name                                    | Description                                                                                                                                                          |
| ---------: | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|          1 | **Timestamp**                                  | The exact date and time when the traffic data was recorded.                                                                                                          |
|          2 | **IR Presence (Lane 1-4)**                     | Indicates whether vehicles were detected by infrared (IR) sensors in each of the four lanes. `1` indicates vehicle detection and `0` indicates no vehicle detection. |
|          3 | **Vehicle Count**                              | The total number of vehicles detected at the recorded timestamp.                                                                                                     |
|          4 | **Avg Speed (km/h)**                           | The average speed of vehicles at the recorded time, measured in kilometers per hour.                                                                                 |
|          5 | **Vehicle Types Detected**                     | Represents the types and counts of vehicles detected, such as cars, bikes, trucks, buses, and other vehicle categories.                                              |
|          6 | **Vehicle Density (%)**                        | Represents how crowded the road is as a percentage. A higher value generally indicates greater traffic density.                                                      |
|          7 | **Saturation Flow Rate(veh/hr/lane)**          | Represents the maximum number of vehicles that can pass through one lane in one hour under ideal traffic conditions.                                                 |
|          8 | **Volume to Saturation Lane Traffic ratio(%)** | Compares the current traffic volume with the available traffic capacity. A higher value indicates heavier traffic relative to the available capacity.                |
|          9 | **FreeFlowSpeed (km/h)**                       | Represents the expected vehicle speed when there is little or no traffic congestion.                                                                                 |
|         10 | **TSR**                                        | **Traffic Saturation Ratio** — represents how heavily the road is being used compared with its available traffic capacity.                                           |
|         11 | **VLSR**                                       | **Volume-to-Lane Saturation Ratio** — represents traffic volume relative to the available lane capacity.                                                             |
|         12 | **Speed Factor**                               | Compares the current average speed with the free-flow speed. A lower value generally indicates slower traffic and greater congestion.                                |
|         13 | **CI**                                         | **Congestion Index** — a numerical measure representing the level of traffic congestion. Higher values generally indicate greater congestion.                        |
|         14 | **Congestion Level**                           | Represents the overall traffic condition as a categorical congestion level. The dataset contains categories such as Low, Moderate, High, and Very High.              |

---

# 🚥 Congestion Levels

The dataset contains four congestion categories:

| Congestion Level | General Meaning                                                            |
| ---------------- | -------------------------------------------------------------------------- |
| **Low**          | Traffic conditions are relatively normal with little congestion.           |
| **Moderate**     | Traffic is increasing and some congestion is present.                      |
| **High**         | Significant congestion is present and traffic movement is more restricted. |
| **Very High**    | Severe congestion with very heavy traffic conditions.                      |

The exact distribution and relationship between these categories and the numerical traffic measurements will be analyzed during the data analysis and machine learning stages.

---

# 🤖 Machine Learning

The project will use machine learning to learn patterns from historical traffic data and predict congestion conditions.

The primary planned machine learning approach is **XGBoost**.

## XGBoost

XGBoost is a tree-based machine learning algorithm that works particularly well with structured or tabular datasets.

For this project, XGBoost can learn relationships between traffic-related features and congestion conditions.

Possible input features include:

```text
Vehicle Count
Avg Speed
Vehicle Density
Saturation Flow Rate
Volume to Saturation Lane Traffic ratio
FreeFlowSpeed
TSR
VLSR
Speed Factor
CI
Time-related features
```

The final set of features will be determined during the data preprocessing and feature-selection stages.

---

# 🎯 Prediction Target

The main prediction objective of the project is **traffic congestion**.

The dataset contains a `Congestion Level` column with four categories:

```text
Low
Moderate
High
Very High
```

The exact machine learning target representation and feature selection will be documented after the model development stage.

---

# 🧠 LSTM — Optional Extension

Traffic data is time-dependent because traffic conditions can change according to time and previous traffic patterns.

**Long Short-Term Memory (LSTM)** is a type of neural network designed to work with sequential and time-series data.

LSTM may be explored as an optional extension after the basic XGBoost implementation is completed.

The project does not require LSTM for the basic working version.

---

# 🕸️ GNN — Optional Advanced Extension

Traffic conditions can also be influenced by nearby roads and connected intersections.

A **Graph Neural Network (GNN)** can represent a road network as a graph:

```text
Intersection A ───── Intersection B
       │                    │
       │                    │
Intersection C ───── Intersection D
```

In this representation:

* **Nodes** can represent intersections or traffic locations.
* **Edges** can represent roads connecting those locations.

GNN can potentially learn how traffic conditions at connected locations influence one another.

GNN is considered an **optional advanced extension** and is not required for the basic traffic prediction system.

---

# 🌦️ Optional Weather and Event Data

The basic project focuses on historical traffic data.

Additional information can be incorporated later, including:

### Weather

* Temperature
* Rainfall
* Weather conditions

### Events

* Public holidays
* Festivals
* Sports events
* Large public gatherings

These features may help the model understand unusual changes in traffic patterns.

Weather and event information are considered **optional features**, not mandatory requirements for the basic system.

---

# 🛠️ Technology Stack

| Technology     | Purpose                                  |
| -------------- | ---------------------------------------- |
| **Python**     | Main programming language                |
| **Pandas**     | Data loading, cleaning, and analysis     |
| **XGBoost**    | Primary machine learning approach        |
| **LSTM**       | Optional time-series prediction approach |
| **PostgreSQL** | Database management                      |
| **FastAPI**    | Backend API development                  |
| **React**      | Frontend development                     |
| **Mapbox**     | Map-based traffic visualization          |
| **GNN**        | Optional advanced spatial modelling      |

---

# 🏗️ High-Level System Architecture

The planned system architecture is:

```text
                  Historical Traffic Data
                           │
                           ▼
                  ┌───────────────────┐
                  │ Data Preprocessing │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Feature Engineering│
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Machine Learning  │
                  │     XGBoost       │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Congestion        │
                  │ Prediction        │
                  └─────────┬─────────┘
                            │
                            ▼
                      FastAPI Backend
                            │
                    ┌───────┴───────┐
                    │               │
                    ▼               ▼
               PostgreSQL       React Frontend
                                    │
                                    ▼
                                  Mapbox
                                    │
                                    ▼
                         Traffic Visualization
```

---

# 📁 Planned Project Structure

The project can be organized using the following structure:

```text
dynamic-urban-traffic-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── model/
│
├── notebooks/
│
├── backend/
│
├── frontend/
│
├── requirements.txt
│
└── README.md
```

### Folder Purpose

| Folder/File        | Purpose                                                   |
| ------------------ | --------------------------------------------------------- |
| `data/`            | Stores datasets                                           |
| `data/raw/`        | Stores original datasets                                  |
| `data/processed/`  | Stores cleaned/processed datasets                         |
| `model/`           | Stores trained machine learning models                    |
| `notebooks/`       | Stores Jupyter notebooks for analysis and experimentation |
| `backend/`         | Contains backend/API code                                 |
| `frontend/`        | Contains React application code                           |
| `requirements.txt` | Contains required Python libraries                        |
| `README.md`        | Project documentation                                     |

---

# 🔄 Development Workflow

The project will be developed incrementally:

```text
Phase 1
Project Understanding
        ↓
Phase 2
Dataset Understanding
        ↓
Phase 3
Data Cleaning & Analysis
        ↓
Phase 4
Feature Engineering
        ↓
Phase 5
XGBoost Model
        ↓
Phase 6
Model Evaluation
        ↓
Phase 7
PostgreSQL
        ↓
Phase 8
FastAPI
        ↓
Phase 9
React
        ↓
Phase 10
Mapbox
        ↓
Optional LSTM
        ↓
Optional GNN
```

---

# 📈 Model Evaluation

The model will be evaluated using appropriate machine learning evaluation metrics.

For numerical traffic prediction, metrics such as:

* **MAE — Mean Absolute Error**
* **RMSE — Root Mean Squared Error**
* **R² — R-squared**

can be used.

For congestion-level classification, appropriate classification metrics can also be considered.

**Actual model evaluation results will be added after model training is completed.**

---

# 🗄️ PostgreSQL

PostgreSQL is planned as the database for storing structured traffic-related information.

Potential database information may include:

* Traffic records
* Timestamps
* Traffic measurements
* Prediction results
* Congestion levels

The final database schema will be documented after PostgreSQL implementation.

---

# 🔌 FastAPI

FastAPI is planned as the backend framework.

The backend will provide a connection between the machine learning model and the frontend application.

The planned workflow is:

```text
React Frontend
      ↓
FastAPI API
      ↓
Machine Learning Model
      ↓
Prediction
      ↓
FastAPI Response
      ↓
React Frontend
```

The exact API endpoints will be documented after the backend is implemented.

---

# 🖥️ React

React is planned for developing the frontend dashboard.

The frontend may allow users to:

* Enter traffic-related information.
* Request a prediction.
* View predicted congestion.
* View traffic information.
* View traffic conditions on a map.

The final frontend functionality will be documented after implementation.

---

# 🗺️ Mapbox

Mapbox is planned for map-based visualization.

The system may use the map to display roads, intersections, and predicted congestion conditions.

A possible visualization approach is:

```text
🟢 Low
🟡 Moderate
🟠 High
🔴 Very High
```

The final map implementation will be documented after development.

---

# ⚙️ Installation

Installation instructions will be added once the project dependencies and final application structure are finalized.

The project is expected to use Python and the libraries specified in the final `requirements.txt` file.

---

# ▶️ Running the Project

Detailed commands for:

* Data analysis
* Model training
* Backend execution
* Frontend execution
* Database setup

will be added after each component has been implemented and tested.

---

# 📊 Results

Model performance results will be added after training and evaluation.

The final results section will include information such as:

```text
Model Used:
Training Data:
Testing Data:
MAE:
RMSE:
R²:
Accuracy / Classification Metrics:
```

Visualizations and screenshots can also be added here.

---

# ⚠️ Current Limitations

Based on the current project definition, possible limitations include:

* The system primarily relies on historical traffic data.
* Unexpected traffic events may be difficult to predict.
* Weather and event information are not mandatory in the basic version.
* Traffic patterns can vary between different roads and intersections.
* Advanced spatial modelling is not required for the initial implementation.
* Model performance will depend on the quality and characteristics of the available dataset.

Additional limitations will be added after the complete system is implemented and evaluated.

---

# 🚀 Future Scope

Possible future improvements include:

* Real-time traffic data integration.
* Weather data integration.
* Event and holiday information.
* LSTM-based time-series prediction.
* GNN-based spatial traffic modelling.
* Prediction across multiple intersections.
* Real-time traffic visualization.
* Improved map-based monitoring.
* Cloud deployment.
* Automatic model retraining.

---

# 📌 Project Status

**Status: 🚧 In Development**

### Current project foundation

* [x] Project idea defined
* [x] Dataset selected
* [x] Dataset identified
* [x] Dataset structure identified
* [x] Congestion categories identified
* [ ] Data preprocessing
* [ ] Exploratory Data Analysis
* [ ] Feature engineering
* [ ] XGBoost model
* [ ] Model evaluation
* [ ] PostgreSQL integration
* [ ] FastAPI backend
* [ ] React frontend
* [ ] Mapbox visualization
* [ ] Optional LSTM
* [ ] Optional GNN

The checklist will be updated as development progresses.

---

# 🎓 Academic Purpose

This project is developed as an academic project to demonstrate the application of:

* Data analysis
* Machine learning
* Traffic prediction
* Database management
* Backend development
* Frontend development
* Data visualization

The project follows a gradual development approach, starting with a basic machine learning model and extending toward a complete traffic prediction application.

---

# 👥 Team

Team member information will be added here.

```text
Team Members:
1. __________________
2. __________________
3. __________________
4. __________________
5. __________________

Project Guide:
______________________

Institution:
______________________

Department:
______________________

Academic Year:
______________________
```

---

# 📜 License

This project is developed for **educational and academic purposes**.

Additional licensing information can be added according to the dataset and project requirements.
