import streamlit as st
import joblib
import numpy as np
import time

# Load trained model
model = joblib.load("model.joblib")

# Page config
st.set_page_config(page_title="Iris Flower Classifier", layout="centered")

# Add custom background color
st.markdown("""
    <style>
    .stApp {
        background-color: aquamarine !important;
    }
    .main {
       background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        text-align: center;  /* ✅ Discover the iris species instantly */
    }
    .stButton > button {
        background-color: #4a90e2;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #2c70c9;
    }
    </style>
""", unsafe_allow_html=True)

# Custom title
st.markdown("<h1 style='text-align:center;'>🌸 Iris Flower Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>A Machine Learning Demo with Streamlit</p>", unsafe_allow_html=True)

# Input form area with white card
with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sepal Measurements")
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)

    with col2:
        st.subheader("Petal Measurements")
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

    if st.button("Predict 🌟"):
        with st.spinner("Analyzing..."):
            time.sleep(1)
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(input_data)
        st.success(f"🌼 The predicted Iris species is: **{prediction[0]}**")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; font-size: 13px; color: gray;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
