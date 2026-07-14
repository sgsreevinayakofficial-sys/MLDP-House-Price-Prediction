import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏠 House Price Prediction")

st.write("Enter the house details below.")

area = st.number_input("Area", value=5000)
bedrooms = st.number_input("Bedrooms", value=3)
bathrooms = st.number_input("Bathrooms", value=2)
stories = st.number_input("Stories", value=2)

mainroad = st.selectbox("Main Road", ["No","Yes"])
guestroom = st.selectbox("Guest Room", ["No","Yes"])
basement = st.selectbox("Basement", ["No","Yes"])
hotwaterheating = st.selectbox("Hot Water Heating", ["No","Yes"])
airconditioning = st.selectbox("Air Conditioning", ["No","Yes"])

parking = st.number_input("Parking", value=1)

prefarea = st.selectbox("Preferred Area", ["No","Yes"])

furnishing = st.selectbox(
    "Furnishing Status",
    ["furnished","semi-furnished","unfurnished"]
)

# Convert Yes/No to 1/0
mainroad = 1 if mainroad=="Yes" else 0
guestroom = 1 if guestroom=="Yes" else 0
basement = 1 if basement=="Yes" else 0
hotwaterheating = 1 if hotwaterheating=="Yes" else 0
airconditioning = 1 if airconditioning=="Yes" else 0
prefarea = 1 if prefarea=="Yes" else 0

furnished = 1 if furnishing=="furnished" else 0
semi = 1 if furnishing=="semi-furnished" else 0
unfurnished = 1 if furnishing=="unfurnished" else 0

data = pd.DataFrame({
    "area":[area],
    "bedrooms":[bedrooms],
    "bathrooms":[bathrooms],
    "stories":[stories],
    "mainroad":[mainroad],
    "guestroom":[guestroom],
    "basement":[basement],
    "hotwaterheating":[hotwaterheating],
    "airconditioning":[airconditioning],
    "parking":[parking],
    "prefarea":[prefarea],
    "furnishingstatus_furnished":[furnished],
    "furnishingstatus_semi-furnished":[semi],
    "furnishingstatus_unfurnished":[unfurnished]
})

# Scale numerical features
num_cols = ["area","bedrooms","bathrooms","stories","parking"]
data[num_cols] = scaler.transform(data[num_cols])

if st.button("Predict Price"):
    prediction = model.predict(data)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")