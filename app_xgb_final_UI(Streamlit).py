import streamlit as st
import pandas as pd
import numpy as np
import joblib
import webbrowser

# =================================================
# CONSTANTS (MATCH TRAINING)
# =================================================
REFERENCE_YEAR = 2025

# =================================================
# LOAD MODEL & FEATURE ORDER
# =================================================
model = joblib.load("car_price_xgboost_final_model.joblib")
feature_columns = joblib.load("model_features.joblib")

make_columns  = [c for c in feature_columns if c.startswith("make_cleaned_")]
model_columns = [c for c in feature_columns if c.startswith("model_cleaned_")]

# =================================================
# BRAND POPULARITY (FROM DATASET)
# =================================================
brand_popularity_map = {
    "Kia": 0.04005,
    "Mazda": 0.04004,
    "Subaru": 0.04023,
    "Tesla": 0.04018,
    "Nissan": 0.04012,
    "Porsche": 0.04010,
    "Ram": 0.04009,
    "Acura": 0.04008,
    "Land Rover": 0.04007,
    "Chrysler": 0.04006,
    "Volkswagen": 0.04005,
    "Dodge": 0.04004,
    "Audi": 0.03990,
    "Honda": 0.04001,
    "Jeep": 0.04000,
    "Chevrolet": 0.03998,
    "Lexus": 0.03992,
    "Cadillac": 0.03985,
    "Volvo": 0.03984,
    "Ford": 0.03984,
    "Other": 0.03950
}

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(page_title="Car Price Prediction", layout="centered")
st.title("🚗 Car Price Prediction System")
st.write("Predict used car prices using a trained Machine Learning model.")

# =================================================
# SIDEBAR
# =================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose an option:", ("Car Price Prediction", "Dashboard"))

# =================================================
# UI MAKE & MODEL
# =================================================
makes = list(brand_popularity_map.keys())

models_by_make = {
    "Honda": ["Accord", "CR-V", "Civic", "Pilot", "Odyssey"],
    "Volkswagen": ["Jetta", "Atlas", "Passat", "Tiguan"],
    "Other": ["Other_Model"]
}

# =================================================
# PAGE 1: PREDICTION
# =================================================
if page == "Car Price Prediction":

    st.subheader("Enter Car Details")

    make = st.selectbox("Make", makes)
    model_name = st.selectbox(
        "Model",
        models_by_make.get(make, ["Other_Model"])
    )

    condition = st.selectbox("Condition", ["Fair", "Good", "Excellent"])
    condition_map = {"Fair": 1, "Good": 2, "Excellent": 3}

    fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric"])

    year = st.number_input("Manufacturing Year", 1995, REFERENCE_YEAR, 2018)
    vehicle_age = max(REFERENCE_YEAR - year, 1)
    st.number_input("Vehicle Age (years)", value=vehicle_age, disabled=True)

    mileage = st.number_input("Mileage (km)", 0, value=50000)
    engine_hp = st.number_input("Engine Horsepower", 50, value=120)
    owner_count = st.selectbox("Number of Owners", [1, 2, 3, 4, 5])

    mileage_per_year = mileage / vehicle_age

    # =================================================
    # CREATE INPUT VECTOR (EXACT TRAINING FORMAT)
    # =================================================
    input_data = pd.DataFrame([[0]*len(feature_columns)], columns=feature_columns)

    # ---- Numeric features
    input_data.at[0, "year"] = year
    input_data.at[0, "mileage"] = mileage
    input_data.at[0, "engine_hp"] = engine_hp
    input_data.at[0, "owner_count"] = owner_count
    input_data.at[0, "vehicle_age"] = vehicle_age
    input_data.at[0, "mileage_per_year"] = mileage_per_year
    input_data.at[0, "condition_encoded"] = condition_map[condition]

    # ---- Brand popularity (LOG TRANSFORMED – VERY IMPORTANT)
    bp = brand_popularity_map.get(make, brand_popularity_map["Other"])
    input_data.at[0, "brand_popularity"] = np.log1p(bp)

    # ---- Make encoding
    make_col = next(
        (c for c in make_columns if c.replace("make_cleaned_", "").lower() == make.lower()),
        "make_cleaned_Other"
    )
    input_data.at[0, make_col] = 1

    # ---- Model encoding
    model_col = next(
        (c for c in model_columns if c.replace("model_cleaned_", "").lower() == model_name.lower()),
        "model_cleaned_Other_Model"
    )
    input_data.at[0, model_col] = 1

    # ---- Fuel type
    fuel_col = f"fuel_type_{fuel_type}"
    if fuel_col in input_data.columns:
        input_data.at[0, fuel_col] = 1

    # ---- Transmission (your dataset has Manual encoded)
    input_data.at[0, "transmission_Manual"] = 1

    # =================================================
    # PREDICT
    # =================================================
    if st.button("🔮 Predict Price"):
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Car Price: ${prediction:,.2f}")
        st.caption("Prediction pipeline exactly matches training pipeline.")

# =================================================
# PAGE 2: DASHBOARD
# =================================================
elif page == "Dashboard":
    st.subheader("📊 Analytics Dashboard")
    webbrowser.open_new_tab("https://public.tableau.com/views/Book1_17700872354460/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link")
