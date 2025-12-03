# ======================================================
# 1️⃣ Import Libraries
# ======================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ======================================================
# 2️⃣ Page Configuration
# ======================================================
st.set_page_config(
    page_title="Life Expectancy Predictor 🌍",
    page_icon="💉",
    layout="wide"
)

st.title("🌟 Life Expectancy Predictor")
st.markdown("""
Predict life expectancy based on **health**, **economy**, and **education** indicators.  
Adjust the values in the sidebar and click **Predict** to see the result.
""")

# ======================================================
# 3️⃣ Load Pre-trained Model
# ======================================================
# Make sure your trained model is saved as 'life_model.pkl'
best_model = joblib.load("life_model.pkl")

# ======================================================
# 4️⃣ Default Values for Features
# ======================================================
default_values = {
    "Status": 1,
    "Year": 2015,
    "Adult Mortality": 150,
    "infant deaths": 10,
    "Alcohol": 1.0,
    "percentage expenditure": 60.0,
    "Hepatitis B": 90,
    "Measles ": 100,
    " BMI ": 23.5,
    "under-five deaths ": 15,
    "Polio": 90,
    "Total expenditure": 5.0,
    "Diphtheria ": 90,
    " HIV/AIDS": 0.1,
    "GDP": 4000,
    "Population": 1000000,
    " thinness  1-19 years": 15.0,
    " thinness 5-9 years": 14.0,
    "Income composition of resources": 0.5,
    "Schooling": 12.5
}

# ======================================================
# 5️⃣ Sidebar - User Input
# ======================================================
st.sidebar.header("Enter Country Indicators")

status = st.sidebar.selectbox("Status (0=Developing, 1=Developed)", [0, 1], index=default_values["Status"])
year = st.sidebar.slider("Year", 2000, 2025, default_values["Year"])
adult_mortality = st.sidebar.slider("Adult Mortality", 0, 500, default_values["Adult Mortality"])
infant_deaths = st.sidebar.slider("Infant Deaths", 0, 500, default_values["infant deaths"])
alcohol = st.sidebar.slider("Alcohol Consumption (liters per capita)", 0.0, 20.0, default_values["Alcohol"])
bmi = st.sidebar.slider("BMI (Body Mass Index)", 10.0, 50.0, default_values[" BMI "])
hepb = st.sidebar.slider("Hepatitis B Coverage (%)", 0, 100, default_values["Hepatitis B"])
polio = st.sidebar.slider("Polio Vaccine Coverage (%)", 0, 100, default_values["Polio"])
diphtheria = st.sidebar.slider("Diphtheria Vaccine Coverage (%)", 0, 100, default_values["Diphtheria "])
hiv = st.sidebar.slider("HIV/AIDS (%)", 0.0, 10.0, default_values[" HIV/AIDS"])
gdp = st.sidebar.number_input("GDP (in USD)", 0, 100000, default_values["GDP"])
schooling = st.sidebar.slider("Schooling Years", 0.0, 20.0, default_values["Schooling"])

# ======================================================
# 6️⃣ Prepare Input for Prediction
# ======================================================
input_data = default_values.copy()
input_data["Status"] = status
input_data["Year"] = year
input_data["Adult Mortality"] = adult_mortality
input_data["infant deaths"] = infant_deaths
input_data["Alcohol"] = alcohol
input_data[" BMI "] = bmi
input_data["Hepatitis B"] = hepb
input_data["Polio"] = polio
input_data["Diphtheria "] = diphtheria
input_data[" HIV/AIDS"] = hiv
input_data["GDP"] = gdp
input_data["Schooling"] = schooling

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# ⚡ Fix Feature Names & Order (must match training)
feature_columns = best_model.feature_names_in_
input_df = input_df[feature_columns]

# ======================================================
# 7️⃣ Prediction
# ======================================================
if st.button("Predict Life Expectancy"):
    prediction = best_model.predict(input_df)
    st.success(f"🎯 Predicted Life Expectancy: {prediction[0]:.2f} years")

# ======================================================
# 8️⃣ Feature Importance
# ======================================================
st.subheader("Feature Importance")
importances = best_model.feature_importances_
feat_importance = pd.Series(importances, index=feature_columns).sort_values(ascending=True)

plt.figure(figsize=(10,6))
feat_importance.plot(kind="barh", color="skyblue")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
st.pyplot(plt)
