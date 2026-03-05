import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model, feature_names = joblib.load("churn_model.pkl")

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction System")

st.write("Predict whether a telecom customer will churn.")

st.divider()

# Input section
st.header("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges", 20, 120, 70)

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )

st.divider()

# Build input data
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Contract": contract,
    "InternetService": internet_service,
    "PaymentMethod": payment_method
}

input_df = pd.DataFrame([input_dict])

# Convert categorical variables
input_df = pd.get_dummies(input_df)

# Add missing columns
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure correct column order
input_df = input_df[feature_names]

# Prediction
if st.button("Predict Churn"):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    if prediction == 1:
        col1.error("⚠️ Customer likely to churn")
    else:
        col1.success("✅ Customer will stay")

    col2.metric("Churn Probability", f"{probability*100:.2f}%")