import joblib

# Find the index of the best model in the sorted DataFrame
best_model = model_mapping[best_model_name]
best_model_name = best_model_row['Model']

# Map the model name back to the actual model object
model_mapping = {
    lg.__class__.__name__: lg,
    knn.__class__.__name__: knn,
    rf.__class__.__name__: rf,
    gb.__class__.__name__: gb,
    ada.__class__.__name__: ada,
    xgb.__class__.__name__: xgb,
    nb.__class__.__name__: nb,
    svm.__class__.__name__: svm
}

best_model = model_mapping[best_model_name]

joblib.dump(best_model, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")


import streamlit as st
import joblib
import numpy as np

model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Risk Prediction App")
st.write("Write customer details to predict loan default risk")

age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Annual Income")
loan_amount = st.number_input("Loan Amount")
credit_utilization = st.number_input("Credit Utilization (%)")
payment_history = st.number_input("Number of Late Payments")

if st.button("Predict Default Risk"):

    input_data = np.array([[age, income, loan_amount, credit_utilization, payment_history]])

    input_scaled = scaler.transform(input_data)

    probability = model.predict_proba(input_scaled)[0][1]

    if probability >= 0.4:
        st.error(f"High Risk of Default ({probability:.2f})")
    else:
        st.success(f"Low Risk of Default ({probability:.2f})")


