import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -----------------------------
# Load the saved model and scaler
# -----------------------------
model = load_model('churn_ann_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

st.title("📊 Customer Churn Predictor")
st.write("Enter customer details below to predict whether they are likely to churn.")

# -----------------------------
# User inputs (must match the features used in training)
# -----------------------------
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)

geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

gender = st.selectbox("Gender", ["Female", "Male"])

age = st.slider("Age", 18, 100, 35)

tenure = st.slider("Tenure (years with bank)", 0, 10, 3)

balance = st.number_input("Account Balance", min_value=0.0, value=50000.0, step=1000.0)

num_products = st.slider("Number of Products", 1, 4, 1)

has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])

is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])

estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

# -----------------------------
# Encode inputs EXACTLY like training
# Column order must match X_train columns:
#  CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard,
#  IsActiveMember, EstimatedSalary, balance_to_salary, Tenure_by_age,
#  Geography_Germany, Geography_Spain
#
# Gender: LabelEncoder -> Female=0, Male=1
# balance_to_salary = Balance / EstimatedSalary
# Tenure_by_age      = Balance / Age   (note: uses Balance, not Tenure, despite the name)
# -----------------------------
gender_val = 1 if gender == "Male" else 0
geography_germany = 1 if geography == "Germany" else 0
geography_spain = 1 if geography == "Spain" else 0
has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_val = 1 if is_active_member == "Yes" else 0

balance_to_salary = balance / estimated_salary if estimated_salary != 0 else 0
tenure_by_age = balance / age if age != 0 else 0

input_data = np.array([[
    credit_score,
    gender_val,
    age,
    tenure,
    balance,
    num_products,
    has_cr_card_val,
    is_active_val,
    estimated_salary,
    balance_to_salary,
    tenure_by_age,
    geography_germany,
    geography_spain
]])

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Churn"):
    input_scaled = scaler.transform(input_data)
    prediction_prob = model.predict(input_scaled)[0][0]
    prediction = prediction_prob > 0.5

    st.subheader("Result")
    if prediction:
        st.error(f"⚠️ Likely to CHURN (probability: {prediction_prob:.2%})")
    else:
        st.success(f"✅ Likely to STAY (churn probability: {prediction_prob:.2%})")

    st.progress(float(prediction_prob))
