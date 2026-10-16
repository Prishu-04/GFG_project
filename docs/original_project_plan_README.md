# Dynamic Urban Traffic Prediction

A beginner-friendly, group-project system that predicts future traffic volume from historical traffic data and classifies the expected congestion level as **Low**, **Moderate**, or **High**.

---

## 📌 Project Objective

The system learns traffic patterns from historical data. Given inputs such as time, day, previous traffic, average speed, waiting time, and an optional event indicator, the model predicts the expected number of vehicles. The predicted volume is then converted into a congestion category.

---

## 🧩 Main Inputs (Features)

| Feature | Purpose |
|---|---|
| Hour | Traffic changes by time of day. |
| Day of week | Weekdays and weekends can have different patterns. |
| Vehicle count | Measures traffic volume in the historical record. |
| Average speed | Helps describe how freely traffic is moving. |
| Waiting time | Useful for identifying intersection congestion. |
| Previous traffic | Recent traffic helps predict upcoming traffic. |
| Event (optional) | Festivals/special events can change normal traffic. |

---

## 🔄 Prediction Flow

```
Historical Traffic Data → Pandas Data Cleaning → Feature Preparation → XGBoost → Predicted Vehicle Count → Congestion Classification
```

**Example**
- **Input:** 9 AM, Monday, previous traffic = 850, average speed = 18 km/h, waiting time = 2 minutes, event = Yes.
- **Output:** predicted traffic ≈ 1050 vehicles → **High Congestion**.

---

## 🚦 Congestion Classification

The ML model's main target is traffic **volume**, not a hard-coded congestion label. After prediction, suitable traffic-volume thresholds for the selected road/dataset are used to label the result as Low, Moderate, or High. Thresholds are justified from the dataset rather than presented as universal traffic rules.

---

## 📋 Recommended Scope

| Keep in Basic Version | Future / Optional |
|---|---|
| Historical traffic data | Real-time CCTV vehicle detection |
| XGBoost prediction | LSTM |
| Vehicle count, speed, waiting time | GNN / spatial modelling |
| Hour, day, previous traffic | Advanced weather features |
| Simple event indicator | Complex probability/Bayesian model |

---

## 🛠️ Technology Stack

| Technology | Use in Project |
|---|---|
| Python | Main programming language. |
| Pandas | Load, inspect, clean, and prepare traffic data. |
| Scikit-learn | Train/test splitting and evaluation utilities. |
| XGBoost | Main machine-learning model for traffic-volume prediction. |
| Matplotlib | Simple charts for data analysis and model results. |
| Jupyter Notebook | Beginner-friendly environment for ML development. |
| PostgreSQL | Store historical traffic and application data. |
| FastAPI | Backend API; receives inputs and returns predictions. |
| React | Simple frontend/dashboard for users. |
| Mapbox | Optional map showing roads/intersections and congestion. |

---

## 🏗️ System Architecture

```
Traffic Dataset
      ↓
Pandas — cleaning & feature preparation
      ↓
XGBoost — traffic-volume prediction
      ↓
Predicted vehicle count
      ↓
Low / Moderate / High congestion
      ↓
FastAPI backend ↔ PostgreSQL database
      ↓
React dashboard → Mapbox visualization
```

---

## 🗺️ Development Plan

1. Create a small traffic dataset and understand every column.
2. Load and clean the dataset with Pandas.
3. Split data into training and testing sets.
4. Train the first XGBoost model.
5. Evaluate with MAE, RMSE, and R².
6. Improve features using time, previous traffic, speed, waiting time, and events.
7. Save the trained model.
8. Add PostgreSQL for storing data.
9. Build a FastAPI `/predict` endpoint.
10. Build a simple React dashboard.
11. Add Mapbox visualization if time permits.
12. Keep CCTV, LSTM, GNN, and advanced weather as future enhancements.

---

## ✅ Why This Project Is Suitable

It is small enough to build and understand as a student group, but it still demonstrates the complete ML application pipeline:

**data → preprocessing → machine learning → evaluation → backend API → database → frontend → visualization**

The project can be expanded later without making the basic version unnecessarily complicated.

---

## 🎤 One-Sentence Viva Explanation

> "Our system uses historical traffic data and factors such as time, vehicle count, average speed, waiting time, previous traffic, and special events to predict traffic volume using XGBoost and classify the expected congestion as Low, Moderate, or High."

---

## 📁 Suggested Project Structure

```
dynamic-urban-traffic-prediction/
├── data/                   # Raw and cleaned traffic datasets
├── notebooks/              # Jupyter notebooks for exploration & model training
├── model/                  # Saved XGBoost model artifacts
├── backend/                # FastAPI application (/predict endpoint)
├── frontend/               # React dashboard
├── requirements.txt        # Python dependencies
└── README.md
```

## ⚙️ Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd dynamic-urban-traffic-prediction

# Install Python dependencies
pip install -r requirements.txt

# Run the FastAPI backend
uvicorn backend.main:app --reload

# In a separate terminal, start the React frontend
cd frontend
npm install
npm start
```

## 📊 Evaluation Metrics

Model performance is evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

## 🔮 Future Enhancements

- Real-time CCTV-based vehicle detection
- LSTM for sequential/time-series modelling
- GNN-based spatial modelling across intersections
- Advanced weather feature integration
- Bayesian/probabilistic congestion estimation
