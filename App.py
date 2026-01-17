#================================================================
#  SIMPLE STREAMLIT UI FOR PREDICTIVE MAINTENANCE 
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

# ====================================
# PAGE SETTINGS
# ====================================
st.set_page_config(page_title="Predictive Maintenance App", layout="wide") # title of the app and full wide space 

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "Prediction", "Visualizations", "About Project"]) # creating sidebar to 4 diff space

# Upload CSV
st.sidebar.title("Upload Processed CSV")
uploaded_file = st.sidebar.file_uploader("Upload Processed CSV", type=["csv"]) # slot for uplod file to the model 



#=====================================
# Load Models + Data
#=====================================

def load_models():
    final_model = joblib.load("Final_Failure_model.pkl")
    iso_model = joblib.load("Isolation_Forest_Model.pkl")

    with open("iso_features.json", "r") as f:
        iso_features = json.load(f)
    
    return final_model , iso_model, iso_features

final_model, iso_model, iso_features = load_models()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV Loaded!")
else:
    st.warning("Please Uploade the csv to continue.")
    st.stop()


#=====================================
# PAGE 1 — Dashboard
#=====================================

if page == "Dashboard":

    st.header("Dashboard — Dataset Overview")

    if df is None:
        st.error("Upload a CSV first.")
        st.stop()

    st.subheader("First 25 Rows")
    st.dataframe(df.head(25))

    st.subheader(" Data Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing Values", df.isna().sum().sum())

    st.subheader(" Column Information")
    st.write(df.dtypes)


#=====================================
# PAGE 2 — Prediction Page
#=====================================

if page == "Prediction":
    st.header("Predictions (Failure Probability)")

    if uploaded_file is None:
        st.warning("Upload a CSV file to generate predictions.")
        st.stop()

    st.subheader("Generating Predictions...")

    try:
        df_pred = df.copy()

        failure_model = joblib.load("Final_Failure_model.pkl")

        failure_prob = failure_model.predict_proba(df_pred)[:, 1]

        df_pred["Failure Probability"] = failure_prob
        df_pred["Failure Prediction"] = (failure_prob > 0.5).astype(int)

        st.success("Predictions generated!")

        st.subheader("Prediction Results")
        st.dataframe(df_pred[[
            "anomaly_score", "anomaly_flag",
            "Failure Probability", "Failure Prediction"
        ]].head(20))

        # KPI metrics
        col1, col2 = st.columns(2)
        col1.metric("High Risk Count", (df_pred["Failure Prediction"] == 1).sum())
        col2.metric("Max Failure Probability", round(df_pred["Failure Probability"].max(), 3))

        # Anomaly KPIs
        col3, col4 = st.columns(2)
        col3.metric("Total Anomalies", df_pred["anomaly_flag"].sum())
        col4.metric("Avg Anomaly Score", round(df_pred["anomaly_score"].mean(), 3))

        st.subheader("Failure Probability Trend")
        st.line_chart(df_pred["Failure Probability"])

        st.subheader("Download Prediction Output")

        csv_data = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv_data, "failure_predictions.csv")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


#=====================================
# PAGE 3 — Visualizations Page
#=====================================

if page == "Visualizations":
    st.header("Visualizations")

    # Safety check — show columns for debugging
    st.write("CSV Columns:", df.columns.tolist())
    
      
    # ---- RPM VS TORQUE SCATTER ----
    
    st.subheader("RPM vs Torque Scatter Plot")

    fig, ax = plt.subplots()
    ax.scatter(df["Rotational speed [rpm]"], df["Torque [Nm]"], s=10, alpha=0.7)
    ax.set_xlabel("RPM")
    ax.set_ylabel("Torque (Nm)")
    st.pyplot(fig)

   

    # ---- AIR TEMPERATURE HISTOGRAM ----
    
    st.subheader("Air Temperature Histogram")

    fig, ax = plt.subplots()
    ax.hist(df["Air temperature [K]"], bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel("Air Temperature (K)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)    




# ============================================================
# PAGE 4 — ABOUT PROJECT
# ============================================================
if page == "About Project":

    st.header(" About This Predictive Maintenance Project")

    st.write("""
This is an end-to-end **Predictive Maintenance System** built using:

### ✔ Machine Learning (Random Forest Classifier)  
Predicts whether a machine will fail.

### ✔ Anomaly Detection (Isolation Forest)  
Detects unusual behavior in sensor readings (temperature, torque, RPM).

### ✔ Feature Engineering  
Created additional intelligent features such as:
- Temperature difference  
- Torque per RPM  
- Z-scores for anomaly detection  
- Interaction features  

### ✔ ML Pipeline  
A clean preprocessing + model pipeline saved as `.pkl`.

### ✔ Streamlit Dashboard  
User can upload sensor CSV → get predictions → view visualizations.

---

### Goal
Reduce downtime and improve machine reliability by:
- Identifying failures early  
- Detecting abnormal patterns  
- Helping maintenance teams take proactive action  
""" )
