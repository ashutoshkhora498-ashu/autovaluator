import streamlit as st
from PIL import Image
import pandas as pd
import joblib


model = joblib.load("car_price_model.pkl")
data = pd.read_csv("New_cleaned vehicles.csv")


with open("model_score.txt", "r") as f:
    model_accuracy = float(f.read())


st.sidebar.markdown(f"**📊 Model R² Score:** `{model_accuracy:.2f}`")



logo = Image.open("logo.png")
st.image(logo, width=200)


st.sidebar.markdown("""
### 🧠 What This App Does

- Predicts resale price of used cars  
- Useful for buyers, sellers & dealerships  
- Built using Machine Learning and Streamlit
""")


selected_brand = st.selectbox("Select Car Brand", ["Select a brand"] + sorted(data['company'].unique()))


if selected_brand != "Select a brand":
    models = sorted(data[data['company'] == selected_brand]['name'].unique())
    selected_model = st.selectbox("Select Car Model", ["Select a model"] + models)
else:
    selected_model = st.selectbox("Select Car Model", ["Select a model"])


year = st.selectbox("Year", ["Select year"] + sorted(data['year'].unique(), reverse=True))

fuel = st.selectbox("Fuel Type", ["Select fuel type"] + sorted(data['fuel_type'].unique()))


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

       
        current_year = datetime.datetime.now().year
        input_df['car_age'] = current_year - input_df['year']
        input_df.drop(['year'], axis=1, inplace=True)

        
        input_df_encoded = pd.get_dummies(input_df)

       
        model_columns = model.feature_names_in_  
        for col in model_columns:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0  

        input_df_encoded = input_df_encoded[model_columns]  
        
        predicted_price = model.predict(input_df_encoded)[0]
        st.success(f"💰 Estimated Resale Price: ₹ {predicted_price:,.2f}")
    else:
        st.warning("⚠️ Please select all fields before predicting.")
