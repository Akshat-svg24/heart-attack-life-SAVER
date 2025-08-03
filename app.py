import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Heart Attack Risk Predictor")
st.write("Enter your health details:")

age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
trestbps = st.slider("Resting BP (mm Hg)", 80,200,120)
chol = st.slider("Cholesterol (mg/dL)", 100,600,200)
fbs = st.selectbox("Fasting Blood Sugar >120 mg/dL?", ["Yes","No"])
thalach = st.slider("Max Heart Rate Achieved", 70,210,150)
exang = st.selectbox("Exercise Induced Angina?", ["Yes","No"])
oldpeak = st.slider("ST depression", 0.0,6.0,1.0,step=0.1)
slope = st.selectbox("Slope ST Segment (0–2)", [0,1,2])
ca = st.selectbox("Major Vessels Colored (0–3)", [0,1,2,3])
thal = st.selectbox("Thalassemia (1=normal,2=fixed,3=reversible)", [1,2,3])

# Encode
sex = 1 if sex=="Male" else 0
fbs = 1 if fbs=="Yes" else 0
exang = 1 if exang=="Yes" else 0

user_input = np.array([[age,sex,cp,trestbps,chol,fbs,thalach,exang,oldpeak,slope,ca,thal]])
scaled = scaler.transform(user_input)

if st.button("Predict"):
    prob = model.predict_proba(scaled)[0][1]
    risk = round(prob*100,2)
    st.write(f"**Risk of heart attack: {risk}%**")
    if risk>75:
        st.error("🚨 High risk – please consult a doctor.")
    elif risk>40:
        st.warning("⚠️ Moderate risk – consider a health check-up.")
    else:
        st.success("✅ Low risk – keep up healthy habits!")
