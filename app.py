# -------------------------------------------------------------
# app.py
# Streamlit Web App for AI-Based Drug Repurposing
# By: Digvijay Yadav
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

# -------------------------------------------------------------
# Load Model and Encoders
# -------------------------------------------------------------
@st.cache_resource
def load_all():
    model = joblib.load("drug_model.pkl")
    le_drug = joblib.load("le_drug.pkl")
    le_condition = joblib.load("le_condition.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, le_drug, le_condition, tfidf

model, le_drug, le_condition, tfidf = load_all()

# -------------------------------------------------------------
# Streamlit UI Setup
# -------------------------------------------------------------
st.set_page_config(page_title="AI Drug Repurposing", page_icon="💊", layout="centered")

st.title("💊 AI for Drug Repurposing")
st.markdown("### Predict the effectiveness of existing drugs for different conditions")

st.markdown("---")

# Sidebar Info
st.sidebar.header("🔧 Input Settings")
st.sidebar.info("Select or enter drug details to predict effectiveness.")

# -------------------------------------------------------------
# User Inputs
# -------------------------------------------------------------
drug_name = st.text_input("Enter Drug Name", placeholder="e.g. Valsartan")
condition_name = st.text_input("Enter Condition", placeholder="e.g. Left Ventricular Dysfunction")
review_text = st.text_area("Enter Drug Review", placeholder="Write how the drug performed or user feedback...")
useful_count = st.slider("Number of Helpful Votes", 0, 500, 50)

# -------------------------------------------------------------
# Prediction Logic
# -------------------------------------------------------------
if st.button("🔮 Predict Effectiveness"):
    if drug_name and condition_name and review_text:
        if drug_name in le_drug.classes_ and condition_name in le_condition.classes_:
            d_enc = le_drug.transform([drug_name])[0]
            c_enc = le_condition.transform([condition_name])[0]
            review_vec = tfidf.transform([review_text])
            review_length = len(review_text)
            
            num_features = np.array([[d_enc, c_enc, useful_count, review_length]])
            features = hstack([num_features, review_vec])
            
            prediction = model.predict(features)[0]
            st.success(f"💡 Predicted Effectiveness Rating: **{prediction:.1f} / 10**")
            
            if prediction > 8:
                st.balloons()
                st.markdown("✅ This drug seems **highly effective** for the given condition.")
            elif prediction > 5:
                st.markdown("⚖️ This drug seems **moderately effective**.")
            else:
                st.markdown("⚠️ This drug seems **less effective** for this condition.")
        else:
            st.warning("⚠️ The entered drug or condition is not found in the model’s database.")
    else:
        st.error("Please fill all fields before predicting.")

st.markdown("---")
st.markdown("""
👨‍⚕️ **Developed by:** Digvijay Yadav  
🏫 **Project Title:** AI for Drug Repurposing  
📘 **Tech Stack:** Python, Scikit-learn, TF-IDF, Streamlit  
""")
