import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }

    h1 {
        color: #00ffd5;
        text-align: center;
        font-size: 50px;
    }

    h2, h3 {
        color: #00ffd5;
    }
     .stButton>button {
        background-color: #00ffd5;
        color: black;
        font-size: 20px;
        border-radius: 12px;
        padding: 12px 28px;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #00c9a7;
        transform: scale(1.05);
    }

    .metric-box {
        background-color: rgba(255,255,255,0.08);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
     </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

data = pd.read_csv("data.csv")

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------

data = data.drop(columns=["id", "Unnamed: 32"])

# Encode target
data["diagnosis"] = data["diagnosis"].map({
    "M": 1,
    "B": 0
})

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------

X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# ---------------------------------------------------
# SCALING
# ---------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------
# SPLIT DATA
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

model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown("<h1>🩺 Breast Cancer Prediction using SVC</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='metric-box'>
        <h2>Machine Learning Healthcare Project</h2>
        <p>Predict whether cancer is Malignant or Benign using Support Vector Classifier.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📊 Model Information")

st.sidebar.success(f"Model Accuracy: {accuracy:.2f}")

st.sidebar.info("Algorithm: Support Vector Classifier (SVC)")

st.sidebar.write("Kernel Used: RBF")

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.subheader("🔬 Enter Medical Details")

col1, col2 = st.columns(2)

with col1:

    radius_mean = st.slider(
        "Radius Mean",
        5.0,
        30.0,
        14.0
    )

    texture_mean = st.slider(
        "Texture Mean",
        5.0,
        40.0,
        20.0
    )

    perimeter_mean = st.slider(
        "Perimeter Mean",
        40.0,
        200.0,
        90.0
    )

with col2:
    area_mean = st.slider(
        "Area Mean",
        100.0,
        2500.0,
        600.0
    )

    smoothness_mean = st.slider(
        "Smoothness Mean",
        0.05,
        0.20,
        0.10
    )

# ---------------------------------------------------
# INPUT ARRAY
# ---------------------------------------------------

input_data = np.zeros((1, X.shape[1]))

input_data[0][0] = radius_mean
input_data[0][1] = texture_mean
input_data[0][2] = perimeter_mean
input_data[0][3] = area_mean
input_data[0][4] = smoothness_mean

# ---------------------------------------------------
# SCALE INPUT
# ---------------------------------------------------

input_scaled = scaler.transform(input_data)

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

if st.button("🔍 Predict Cancer Type"):

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:

        st.error("⚠️ Malignant Cancer Detected")

        st.progress(float(probability[0][1]))
        st.write(f"Confidence: {probability[0][1]*100:.2f}%")

    else:

        st.success("✅ Benign Cancer Detected")

        st.progress(float(probability[0][0]))

        st.write(f"Confidence: {probability[0][0]*100:.2f}%")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    "<center><h4>Made with ❤️ using Streamlit and Machine Learning</h4></center>",
    unsafe_allow_html=True
)