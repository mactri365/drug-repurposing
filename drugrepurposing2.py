# -------------------------------------------------------------
# AI for Drug Repurposing - Explainable Results Version
# By: Digvijay Yadav
# -------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# STEP 1: LOAD DATA
# -----------------------
train = pd.read_csv("./datasets/drugsComTrain_raw.tsv", sep='\t')
print("[SUCCESS] Dataset Loaded:", train.shape)

# -----------------------
# STEP 2: CLEAN AND PREPROCESS
# -----------------------
df = train[['drugName', 'condition', 'review', 'rating', 'usefulCount']].dropna()

# Encode text columns
le_drug = LabelEncoder()
le_cond = LabelEncoder()
df['drug_encoded'] = le_drug.fit_transform(df['drugName'])
df['condition_encoded'] = le_cond.fit_transform(df['condition'])

# Handle numeric missing values and compute review length
df['usefulCount'] = df['usefulCount'].fillna(0)
df['review_length'] = df['review'].apply(lambda x: len(str(x)))

# Convert review text into TF-IDF features
tfidf = TfidfVectorizer(max_features=150, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['review'])

# Combine numeric and text features
X_numeric = df[['drug_encoded', 'condition_encoded', 'usefulCount', 'review_length']].values
X = hstack([X_numeric, tfidf_matrix])
y = df['rating']

# -----------------------
# STEP 3: TRAIN MODEL
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("[SUCCESS] Model Training Complete")

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"[INFO] MAE: {mae:.2f} | R²: {r2:.2f}")

# -----------------------
# STEP 4: PREDICT ON FULL DATA
# -----------------------
df['predicted_rating'] = model.predict(hstack([X_numeric, tfidf_matrix]))
df['predicted_rating'] = df['predicted_rating'].clip(0, 10)

# -----------------------
# STEP 5: IDENTIFY REPURPOSING CANDIDATES
# -----------------------
# Get average predicted rating for each drug-condition pair
drug_condition_effect = (
    df.groupby(['drugName', 'condition'])['predicted_rating']
    .mean()
    .reset_index()
)

# Drugs that are highly effective (≥8) for more than one condition
repurpose_candidates = (
    drug_condition_effect[drug_condition_effect['predicted_rating'] >= 8]
    .groupby('drugName')['condition']
    .apply(list)
    .reset_index()
)

# Keep only drugs with multiple effective conditions
repurpose_candidates = repurpose_candidates[repurpose_candidates['condition'].map(len) > 1]

# -----------------------
# STEP 6: DISPLAY RESULTS
# -----------------------
print("\n[DRUG] Potential Drug Repurposing Insights:\n")
for idx, row in repurpose_candidates.iterrows():
    drug = row['drugName']
    conditions = row['condition']
    main_use = conditions[0]
    new_uses = conditions[1:]
    print(f"[PREDICTION] {drug} (used for {main_use}) is also predicted to be effective for {', '.join(new_uses)}.")

# -----------------------
# STEP 7: VISUALIZE TOP DRUGS
# -----------------------
repurpose_candidates['No_of_Conditions'] = repurpose_candidates['condition'].apply(len)
top10 = repurpose_candidates.sort_values(by='No_of_Conditions', ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(x='No_of_Conditions', y='drugName', data=top10)
plt.title("Top 10 Potentially Repurposed Drugs")
plt.xlabel("Number of Effective Conditions")
plt.ylabel("Drug Name")
plt.savefig("top_repurposed_drugs.png")
print("[SUCCESS] Saved: top_repurposed_drugs.png")
plt.close()

# -----------------------
# STEP 8: SAVE REPORT
# -----------------------
repurpose_candidates.to_csv("repurposed_drugs.csv", index=False)
print("\n[SUCCESS] Results saved as 'repurposed_drugs.csv'")
