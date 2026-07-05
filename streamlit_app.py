import streamlit as st
import joblib
import pandas as pd
import json

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="FedGT Malicious Client Detection",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")

with open("metrics.json", "r") as f:
    metrics = json.load(f)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🛡️ FedGT System")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction", "Dashboard"]
)

# -----------------------------
# Home
# -----------------------------
if page == "Home":

    st.title("Identification of Malicious Clients")
    st.subheader("Federated Learning with Secure Aggregation")

    st.write("""
This application predicts whether a client is malicious
using a trained Random Forest model.

### Features

- Client Prediction
- Dashboard
- Model Accuracy
- Feature Importance
""")

# -----------------------------
# Prediction
# -----------------------------
elif page == "Prediction":

    st.title("Client Prediction")

    col1, col2 = st.columns(2)

    with col1:
        packet = st.number_input("Packet Loss Rate", value=10.0)
        latency = st.number_input("Latency (ms)", value=100.0)
        upload = st.number_input("Upload Throughput", value=25.0)

    with col2:
        download = st.number_input("Download Throughput", value=80.0)
        cpu = st.number_input("CPU Usage (%)", value=50.0)
        memory = st.number_input("Memory Usage (%)", value=60.0)

    if st.button("Predict"):

        data = [[
            packet,
            latency,
            upload,
            download,
            cpu,
            memory
        ]]

        prediction = model.predict(data)[0]

        st.success(f"Prediction: **{prediction}**")

# -----------------------------
# Dashboard
# -----------------------------
elif page == "Dashboard":

    st.title("Analytics Dashboard")

    col1, col2 = st.columns(2)

    col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("Samples", metrics["total_samples"])

    st.subheader("Dataset Distribution")

    df = pd.DataFrame(
        metrics["label_counts"].items(),
        columns=["Attack", "Count"]
    )

    st.bar_chart(df.set_index("Attack"))

    st.subheader("Feature Importance")

    fi = pd.DataFrame(
        metrics["feature_importance"].items(),
        columns=["Feature", "Importance"]
    )

    st.bar_chart(fi.set_index("Feature"))