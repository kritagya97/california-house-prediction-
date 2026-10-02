import streamlit as st
import joblib
import pandas as pd
from tensorflow.keras.models import load_model


# Page configuration
st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="",
    layout="centered"
)


# Load model and scaler
@st.cache_resource
def load_resources():
    scaler = joblib.load("scaler.pkl")
    model = load_model("california_model.keras", compile=False)
    return scaler, model


scaler, model = load_resources()


# Initialize prediction result
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# App title
st.title("California House Price Prediction")
st.write("Enter the house and location details to predict the house price.")


# Input fields
MedInc = st.number_input(
    "Median Income (MedInc)",
    min_value=0.0,
    value=3.0,
    step=0.1
)

HouseAge = st.number_input(
    "House Age",
    min_value=0.0,
    value=20.0,
    step=1.0
)

AveRooms = st.number_input(
    "Average Rooms (AveRooms)",
    min_value=0.0,
    value=5.0,
    step=0.1
)

AveBedrms = st.number_input(
    "Average Bedrooms (AveBedrms)",
    min_value=0.0,
    value=1.0,
    step=0.1
)

Population = st.number_input(
    "Population",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

AveOccup = st.number_input(
    "Average Occupancy (AveOccup)",
    min_value=0.0,
    value=3.0,
    step=0.1
)

Latitude = st.number_input(
    "Latitude",
    value=35.0,
    step=0.1
)

Longitude = st.number_input(
    "Longitude",
    value=-119.0,
    step=0.1
)


# Buttons
col1, col2 = st.columns(2)

with col1:
    predict_button = st.button(
        "Predict House Price",
        type="primary",
        use_container_width=True
    )

with col2:
    clear_button = st.button(
        "Clear Result",
        use_container_width=True
    )


# Clear previous prediction
if clear_button:
    st.session_state.prediction = None
    st.rerun()


# Prediction
if predict_button:

    try:
        # Create DataFrame
        input_data = pd.DataFrame([[
            MedInc,
            HouseAge,
            AveRooms,
            AveBedrms,
            Population,
            AveOccup,
            Latitude,
            Longitude
        ]], columns=[
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ])


        # Scale input using the SAVED scaler
        input_scaled = scaler.transform(input_data)


        # Make prediction
        prediction = model.predict(
            input_scaled,
            verbose=0
        )[0][0]


        # Convert from $100,000 units to dollars
        prediction = prediction * 100000


        # Store prediction
        st.session_state.prediction = prediction


    except Exception as e:
        st.error(f"Prediction error: {e}")


# Display prediction
if st.session_state.prediction is not None:

    st.success(
        f"Predicted House Price: "
        f"${st.session_state.prediction:,.2f}"
    )