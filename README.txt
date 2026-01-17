Predictive Maintenance System — Machine Failure & Anomaly Detection

This project is a complete end-to-end Predictive Maintenance System built using real industrial-style sensor data.

It includes:

✔ Machine Learning (Random Forest Classification)
✔ Anomaly Detection (Isolation Forest)
✔ Full Preprocessing + Feature Engineering Pipeline
✔ Streamlit UI Dashboard
✔ Failure probability predictions
✔ Visualizations & insights
✔ Exportable outputs

The goal is to detect machine failures early, reduce downtime, and assist maintenance teams in taking proactive actions.


Project Structure

Predictive_Maintenance_Project/
│
├── app.py                               # Streamlit UI Application
│
├── notebooks/
│   └── Predictive_Maintenance.ipynb     # EDA, Feature Engineering, Model Training
│
├── models/
│   ├── Final_Failure_model.pkl          # Full Failure Prediction Pipeline (preprocessing + model)
│   ├── Isolation_Forest_Model.pkl       # Anomaly Detection Model
│   └── iso_features.json                # Sensor columns used for Isolation Forest
│
├── data/
│   ├── raw/
│   │   └── sensor.csv                   # Raw Sensor Data
│   └── processed/
│       └── processed_data.csv           # Dataset after feature engineering
│
├── figures/
│   ├── boxplots/
│   ├── histograms/
│   ├── ui_screenshots/
│   │   ├── dashboard.png
│   │   ├── prediction_page.png
│   │   ├── visualization_page1.png
│   │   ├── visualization_page2.png
│   │   └── about_page.png
│   ├── correlation_heatmap.png
│   ├── machine_failure.png
│   ├── pairplot.png
│   └── scatterplot.png
│
│
└── README.md


Machine Learning Workflow

1 Data Cleaning & Preprocessing

Missing-value handling

Outlier detection

Column standardization

Categorical encoding

Pipeline built using ColumnTransformer

2 Feature Engineering

Created intelligent features that improve ML accuracy:

Temperature difference

Torque per RPM

Interaction feature: TempDiff_x_ToolWear

Z-score based anomaly indicators

Binary tool wear classification

3 Failure Prediction Model

Algorithm used:

✔ Random Forest Classifier

Why?

Handles non-linear sensor relationships

Robust to noise

Works well with high-dimensional engineered features

This model predicts:

Failure Probability (0–1)

Failure Prediction (0 or 1)

4 Anomaly Detection

Model used:

✔ Isolation Forest

Detects unusual patterns in:

Torque

RPM

Temperature

Tool wear

Outputs:

Anomaly Score

Anomaly Flag (0 = normal, 1 = anomaly)

How to Run the App (Streamlit UI)

1 Install dependencies

pip install -r requirements.txt


2 Run the Streamlit UI

streamlit run app.py


3 Upload a processed CSV

The UI will automatically show:

Failure probability

Failure prediction

Anomaly score

Visual insights

Downloadable output CSV

Key Features

✔ End-to-End System

From raw data → model training → UI deployment.

✔ Real-Time Predictions

Uses full preprocessing pipeline internally

Works directly on processed dataset

Shows high-risk points

✔ Industrial-Level Insights

Flag anomalies

View RPM–Torque behavior

Analyze temperature distribution

✔ Complete Explainability

Every feature and step is well-documented in the notebook.
✔ Friendly UI

A clean, interactive dashboard presenting:

Dataset overview

ML predictions

Anomaly detection

Visual analytics

Goal of the Project


This system helps industries achieve:


Early detection of machine failure

Reduced downtime

Better maintenance planning

Increased reliability of operations


Technologies Used


| Category      | Tools                           |
| ------------- | ------------------------------- |
| Programming   | Python                          |
| ML Models     | Random Forest, Isolation Forest |
| Visualization | Matplotlib, Seaborn             |
| UI            | Streamlit                       |
| Packaging     | Joblib                          |
| Data          | Pandas, NumPy                   |



Future Improvements

Allow raw sensor CSV prediction (auto-run feature engineering)

Add LSTM-based sequence modeling

Deploy to Cloud (AWS / Streamlit Cloud)

Add real-time Kafka streaming support


Author

Muhammed Hijas

Mechatronics Engineer | Data Science & Machine Learning Trainee

muhammedhijas04@gmail.com