import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🏠 House Price Prediction using SVR")

st.write(
    "This app predicts California house prices using "
    "Support Vector Regressor (SVR)."
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

data = pd.read_csv("housing.csv")

# ---------------------------------------------------
# SELECT FEATURES
# ---------------------------------------------------

data = data[
    [
        'median_income',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_house_value'
    ]
]

# ---------------------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------------------

data = data.fillna(data.median())

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------

X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

model = SVR(kernel='rbf')

model.fit(X_train, y_train)

# ---------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------
# USER INPUT SECTION
# ---------------------------------------------------

st.header("Enter House Details")

median_income = st.slider(
    "Median Income",
    1.0,
    15.0,
    5.0
)

housing_median_age = st.slider(
    "House Age",
    1,
    50,
    20
)

total_rooms = st.number_input(
    "Total Rooms",
    min_value=100,
    value=2000
)

total_bedrooms = st.number_input(
    "Total Bedrooms",
    min_value=50,
    value=400
)

population = st.number_input(
    "Population",
    min_value=100,
    value=1000
)

households = st.number_input(
    "Households",
    min_value=50,
    value=300
)

# ---------------------------------------------------
# CREATE INPUT DATAFRAME
# ---------------------------------------------------

input_data = pd.DataFrame({
    'median_income': [median_income],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households]
})

# ---------------------------------------------------
# SCALE INPUT
# ---------------------------------------------------

input_scaled = scaler.transform(input_data)

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

if st.button("Predict House Price"):

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted House Price: ${prediction[0]:,.2f}"
    )