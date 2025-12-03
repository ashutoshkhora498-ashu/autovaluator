import streamlit as st
from PIL import Image
import pandas as pd
import joblib

# ✅ Load model and dataset
model = joblib.load("car_price_model.pkl")
data = pd.read_csv("New_cleaned vehicles.csv")

# ✅ Load model accuracy
with open("model_score.txt", "r") as f:
    model_accuracy = float(f.read())

# ✅ Show in sidebar
st.sidebar.markdown(f"**📊 Model R² Score:** `{model_accuracy:.2f}`")


# ✅ Optional: background or style
#page_bg_img = ''' ... '''
#st.markdown(page_bg_img, unsafe_allow_html=True)

# ✅ Logo (optional)
logo = Image.open("logo.png")
st.image(logo, width=200)

# ✅ Title
#st.markdown("<h1 style='text-align: center;'>🚗 Car Price Prediction App</h1>", unsafe_allow_html=True)

# ✅ Sidebar info
st.sidebar.markdown("""
### 🧠 What This App Does

- Predicts resale price of used cars  
- Useful for buyers, sellers & dealerships  
- Built using Machine Learning and Streamlit
""")

# ✅ 👇👇👇 PASTE THIS BLOCK HERE 👇👇👇

# 🚘 Step 1: Select Car Brand
selected_brand = st.selectbox("Select Car Brand", ["Select a brand"] + sorted(data['company'].unique()))

# 🏷️ Step 2: Show Models Based on Selected Brand
if selected_brand != "Select a brand":
    models = sorted(data[data['company'] == selected_brand]['name'].unique())
    selected_model = st.selectbox("Select Car Model", ["Select a model"] + models)
else:
    selected_model = st.selectbox("Select Car Model", ["Select a model"])

# 📅 Year
year = st.selectbox("Year", ["Select year"] + sorted(data['year'].unique(), reverse=True))

# 🛢️ Fuel Type
fuel = st.selectbox("Fuel Type", ["Select fuel type"] + sorted(data['fuel_type'].unique()))

# 🚗 Kms Driven
kms = st.number_input("Kilometers Driven", min_value=0, step=1000, format="%d")

import datetime

if st.button("Predict Price"):
    if (
        selected_brand != "Select a brand" and
        selected_model != "Select a model" and
        year != "Select year" and
        fuel != "Select fuel type"
    ):
        # Step 1: Create input DataFrame
        input_df = pd.DataFrame({
            'name': [selected_model],
            'company': [selected_brand],
            'year': [year],
            'kms_driven': [kms],
            'fuel_type': [fuel]
        })

        # Step 2: Preprocess to match training format

        # 🔹 Derive 'car_age'
        current_year = datetime.datetime.now().year
        input_df['car_age'] = current_year - input_df['year']
        input_df.drop(['year'], axis=1, inplace=True)

        # 🔹 One-hot encode
        input_df_encoded = pd.get_dummies(input_df)

        # 🔹 Align columns with training set
        model_columns = model.feature_names_in_  # Automatically gets expected columns
        for col in model_columns:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0  # Add missing columns with 0

        input_df_encoded = input_df_encoded[model_columns]  # Ensure correct column order

        # Step 3: Predict
        predicted_price = model.predict(input_df_encoded)[0]
        st.success(f"💰 Estimated Resale Price: ₹ {predicted_price:,.2f}")
    else:
        st.warning("⚠️ Please select all fields before predicting.")
