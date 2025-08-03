import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Heart Attack Risk Predictor")
st.markdown("Enter your health details below to predict heart attack risk:")

# User inputs
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["Yes", "No"])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina?", ["Yes", "No"])
oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible)", [1, 2, 3])

# Encode categorical inputs
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])
scaled_input = scaler.transform(user_input)

if st.button("Predict"):
    prob = model.predict_proba(scaled_input)[0][1]
    risk_perc = round(prob * 100, 2)
    st.write(f"**Risk of Heart Attack: {risk_perc}%**")
    if risk_perc > 75:
        st.error("🚨 High Risk! Please consult a doctor immediately.")
    elif risk_perc > 40:
        st.warning("⚠️ Moderate Risk. Consider further check-ups.")
    else:
        st.success("✅ Low Risk. Maintain healthy habits!")
