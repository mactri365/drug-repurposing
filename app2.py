# -------------------------------------------------------------
# app.py
# Streamlit Web App - AI for Drug Repurposing
# By: Digvijay Yadav
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------------
st.set_page_config(page_title="AI for Drug Repurposing", page_icon="💊", layout="wide")

st.title("💊 AI for Drug Repurposing")
st.markdown("### Discover how existing drugs can be repurposed for new medical conditions using AI and Machine Learning.")
st.markdown("---")

# -------------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("./datasets/drugsComTrain_raw.tsv", sep="\t")
    df = df[['drugName', 'condition', 'review', 'rating', 'usefulCount']].dropna()
    return df

df = load_data()
st.sidebar.header("📘 Dataset Information")
st.sidebar.write(f"**Total Records:** {len(df):,}")
st.sidebar.write(f"**Unique Drugs:** {df['drugName'].nunique()}")
st.sidebar.write(f"**Unique Conditions:** {df['condition'].nunique()}")

# -------------------------------------------------------------
# MODEL TRAINING (LIGHTWEIGHT)
# -------------------------------------------------------------
@st.cache_resource
def train_model(df):
    le_drug = LabelEncoder()
    le_cond = LabelEncoder()
    df['drug_encoded'] = le_drug.fit_transform(df['drugName'])
    df['condition_encoded'] = le_cond.fit_transform(df['condition'])
    df['usefulCount'] = df['usefulCount'].fillna(0)
    df['review_length'] = df['review'].apply(lambda x: len(str(x)))

    # TF-IDF text vectorization
    tfidf = TfidfVectorizer(max_features=150, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['review'])
    X_num = df[['drug_encoded', 'condition_encoded', 'usefulCount', 'review_length']].values
    X = hstack([X_num, tfidf_matrix])
    y = df['rating']

    # Simple model (fast + explainable)
    model = LinearRegression()
    model.fit(X, y)

    df['predicted_rating'] = model.predict(hstack([X_num, tfidf_matrix]))
    df['predicted_rating'] = df['predicted_rating'].clip(0, 10)

    return model, le_drug, le_cond, tfidf, df

st.info("⏳ Training model (this takes ~15 seconds)...")
model, le_drug, le_cond, tfidf, df = train_model(df)
st.success("✅ Model trained successfully!")

# -------------------------------------------------------------
# FIND REPURPOSING CANDIDATES
# -------------------------------------------------------------
drug_condition_effect = (
    df.groupby(['drugName', 'condition'])['predicted_rating']
    .mean()
    .reset_index()
)

repurpose_candidates = (
    drug_condition_effect[drug_condition_effect['predicted_rating'] >= 8]
    .groupby('drugName')['condition']
    .apply(list)
    .reset_index()
)

repurpose_candidates = repurpose_candidates[repurpose_candidates['condition'].map(len) > 1]
repurpose_candidates['No_of_Conditions'] = repurpose_candidates['condition'].apply(len)

# -------------------------------------------------------------
# MAIN UI
# -------------------------------------------------------------
st.subheader("🔍 Potential Drug Repurposing Insights")

drug_selected = st.selectbox(
    "Search or Select a Drug to See Repurposing Possibilities:",
    sorted(repurpose_candidates['drugName'].unique())
)

if drug_selected:
    conditions = repurpose_candidates[repurpose_candidates['drugName'] == drug_selected]['condition'].values[0]
    main_use = conditions[0]
    new_uses = conditions[1:]
    st.markdown(
        f"💡 **{drug_selected}** (commonly used for **{main_use}**) is also predicted to be effective for: **{', '.join(new_uses)}.**"
    )
    st.success("This indicates potential for *drug repurposing*!")

# -------------------------------------------------------------
# VISUALIZE TOP REPURPOSED DRUGS
# -------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Top 10 Potentially Repurposed Drugs")

top10 = repurpose_candidates.sort_values(by='No_of_Conditions', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='No_of_Conditions', y='drugName', data=top10, ax=ax, palette='cool')
plt.title("Top 10 Repurposed Drugs (Predicted Effective in Multiple Conditions)")
plt.xlabel("Number of Effective Conditions")
plt.ylabel("Drug Name")
st.pyplot(fig)

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.markdown("""
👨‍⚕️ **Developed by:** Digvijay Yadav  
🏫 **Project:** AI for Drug Repurposing  
🧠 **Tech Stack:** Python, Scikit-learn, TF-IDF, Streamlit  
""")
