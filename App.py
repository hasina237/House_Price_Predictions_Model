import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("Advanced House Price Prediction App")
st.markdown("Enter house features to predict the sale price.")

# LOAD MODEL (replace with your model path)

model = joblib.load('House_Price_Model.pkl')

# ---------------------------
# FEATURE INPUTS
# ---------------------------
st.sidebar.header("Input Features")

def user_input_features():
    Gr_Liv_Area = st.sidebar.slider("Ground Living Area", 300, 5000, 1500)
    Overall_Qual = st.sidebar.slider("Overall Quality", 1, 10, 5)
    Garage_Area = st.sidebar.slider("Garage Area", 0, 1500, 500)
    Full_Bath = st.sidebar.slider("Full Bathrooms", 0, 4, 2)
    Half_Bath = st.sidebar.slider("Half Bathrooms", 0, 2, 1)
    Bedroom_AbvGr = st.sidebar.slider("Bedrooms Above Ground", 0, 8, 3)
    Year_Built = st.sidebar.slider("Year Built", 1900, 2026, 2000)
    Year_Remod = st.sidebar.slider("Year Remodeled", 1900, 2026, 2010)
    Total_Bsmt_SF = st.sidebar.slider("Basement Area", 0, 3000, 800)
    First_Flr_SF = st.sidebar.slider("1st Floor SF", 300, 3000, 1000)
    Second_Flr_SF = st.sidebar.slider("2nd Floor SF", 0, 2000, 300)
    Garage_Cars = st.sidebar.slider("Garage Cars", 0, 5, 2)
    TotRms_AbvGrd = st.sidebar.slider("Total Rooms", 2, 14, 6)
    Lot_Area = st.sidebar.slider("Lot Area", 1000, 20000, 8000)
    Overall_Cond = st.sidebar.slider("Overall Condition", 1, 10, 5)
    Mas_Vnr_Area = st.sidebar.slider("Masonry Veneer Area", 0, 1500, 100)
    Bsmt_Full_Bath = st.sidebar.slider("Basement Full Bath", 0, 3, 1)
    Open_Porch_SF = st.sidebar.slider("Open Porch Area", 0, 500, 50)
    Fireplaces = st.sidebar.slider("Fireplaces", 0, 3, 1)

    Remod_age = st.sidebar.slider("Remodel Age", 0, 150, 20)
    Garage_Score = st.sidebar.slider("Garage Score", 0, 100, 50)
    Total_Rooms = st.sidebar.slider("Total Rooms (Engineered)", 2, 20, 7)
    Total_Bathrooms = st.sidebar.slider("Total Bathrooms", 1, 6, 2)
    House_Age = st.sidebar.slider("House Age", 0, 150, 25)
    Total_Area = st.sidebar.slider("Total Area", 500, 6000, 2000)

    data = {
        "Gr Liv Area": Gr_Liv_Area,
        "Overall Qual": Overall_Qual,
        "Garage Area": Garage_Area,
        "Full Bath": Full_Bath,
        "Half Bath": Half_Bath,
        "Bedroom AbvGr": Bedroom_AbvGr,
        "Year Built": Year_Built,
        "Year Remod/Add": Year_Remod,
        "Total Bsmt SF": Total_Bsmt_SF,
        "1st Flr SF": First_Flr_SF,
        "2nd Flr SF": Second_Flr_SF,
        "Garage Cars": Garage_Cars,
        "TotRms AbvGrd": TotRms_AbvGrd,
        "Lot Area": Lot_Area,
        "Overall Cond": Overall_Cond,
        "Mas Vnr Area": Mas_Vnr_Area,
        "Bsmt Full Bath": Bsmt_Full_Bath,
        "Open Porch SF": Open_Porch_SF,
        "Fireplaces": Fireplaces,
        "Remod_age": Remod_age,
        "Garage_Score": Garage_Score,
        "Total_Rooms": Total_Rooms,
        "Total_Bathrooms": Total_Bathrooms,
        "House_Age": House_Age,
        "Total_Area": Total_Area
    }

    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# ---------------------------
# DISPLAY INPUT DATA
# ---------------------------
st.subheader("Input Summary")
st.write(input_df)

# ---------------------------
# PREDICTION
# ---------------------------
st.subheader("Prediction")

if st.button("Predict House Price"):
    if model is not None:
        prediction = model.predict(input_df)
        st.success(f"🏷 Estimated Sale Price: ${prediction[0]:,.2f}")
    else:
        st.error("Model not loaded. Please attach your trained model (.pkl file).")

# ---------------------------
# OPTIONAL INSIGHTS SECTION
# ---------------------------
st.markdown("---")
st.subheader("Tips")
st.info("Higher Overall Quality, Living Area, and Garage Size strongly increase price prediction accuracy.")