import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title(" House Price Prediction System")

st.write(
    "Estimate the selling price of a house using a trained "
    "**Linear Regression** model."
)

st.divider()

st.subheader("Property Information")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", 1000, 20000, 5000, 100)
    bathrooms = st.number_input("Bathrooms", 1, 10, 2)
    parking = st.number_input("Parking", 0, 5, 1)

with col2:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    stories = st.number_input("Stories", 1, 5, 2)

    furnishing = st.selectbox(
        "Furnishing Status",
        ["furnished", "semi-furnished", "unfurnished"]
    )

st.divider()

st.subheader("Amenities")

c1, c2, c3 = st.columns(3)

with c1:
    mainroad = st.checkbox("Main Road")
    basement = st.checkbox("Basement")

with c2:
    guestroom = st.checkbox("Guest Room")
    airconditioning = st.checkbox("Air Conditioning")

with c3:
    hotwaterheating = st.checkbox("Hot Water Heating")
    prefarea = st.checkbox("Preferred Area")

# Encoding
mainroad = int(mainroad)
guestroom = int(guestroom)
basement = int(basement)
hotwaterheating = int(hotwaterheating)
airconditioning = int(airconditioning)
prefarea = int(prefarea)

furnished = 1 if furnishing == "furnished" else 0
semi = 1 if furnishing == "semi-furnished" else 0
unfurnished = 1 if furnishing == "unfurnished" else 0

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

num_cols = ["area","bedrooms","bathrooms","stories","parking"]
data[num_cols] = scaler.transform(data[num_cols])

st.divider()

if st.button("🔍 Predict House Price", use_container_width=True):

    prediction = model.predict(data)[0]

    st.success("Prediction Completed")

    st.metric(
        "Estimated House Price",
        f"${prediction:,.2f}"
    )

    st.subheader("Property Summary")

    summary = pd.DataFrame({
        "Feature":[
            "Area",
            "Bedrooms",
            "Bathrooms",
            "Stories",
            "Parking",
            "Furnishing"
        ],
        "Value":[
            area,
            bedrooms,
            bathrooms,
            stories,
            parking,
            furnishing
        ]
    })

    st.table(summary)

st.divider()

st.caption(
    "House Price Prediction System • "
    "Machine Learning using Linear Regression"
)