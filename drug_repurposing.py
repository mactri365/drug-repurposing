# -------------------------------------------------------------
# AI for Drug Repurposing - Lightweight Final Version
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
test = pd.read_csv("./datasets/drugsComTest_raw.tsv", sep='\t')

print("[SUCCESS] Train Data Loaded:", train.shape)
print("[SUCCESS] Test Data Loaded:", test.shape)
print("\nTraining Data Sample:")
print(train.head())

# -----------------------
# STEP 2: SAMPLE SMALLER DATASET (FASTER TRAINING)
# -----------------------
train = train.sample(5000, random_state=42)
print("\n[SUCCESS] Using a sample of 5000 rows for faster training")

# -----------------------
# STEP 3: PREPROCESSING
# -----------------------
df = train[['drugName', 'condition', 'review', 'rating', 'usefulCount']].dropna()

# Encode categorical features
le_drug = LabelEncoder()
le_cond = LabelEncoder()
df['drug_encoded'] = le_drug.fit_transform(df['drugName'])
df['condition_encoded'] = le_cond.fit_transform(df['condition'])

# Handle missing numeric values and add review length
df['usefulCount'] = df['usefulCount'].fillna(0)
df['review_length'] = df['review'].apply(lambda x: len(str(x)))

# -----------------------
# STEP 4: TF-IDF FOR TEXT REVIEWS
# -----------------------
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['review'])

# Combine numeric and text features
X_numeric = df[['drug_encoded', 'condition_encoded', 'usefulCount', 'review_length']].values
X = hstack([X_numeric, tfidf_matrix])
y = df['rating']

# -----------------------
# STEP 5: SPLIT DATA
# -----------------------
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n[SUCCESS] Data Split Successful:")
print("Training set:", X_train.shape)
print("Validation set:", X_valid.shape)

# -----------------------
# STEP 6: TRAIN FAST MODEL
# -----------------------
model = LinearRegression()
model.fit(X_train, y_train)
print("\n[SUCCESS] Model Training Complete")

# -----------------------
# STEP 7: EVALUATE MODEL
# -----------------------
y_pred = model.predict(X_valid)
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print("\n[INFO] Model Evaluation:")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("R² Score:", round(r2, 2))

# -----------------------
# STEP 8: PREDICTION EXAMPLE
# -----------------------
print("\n[INFO] Available Drugs:", list(df['drugName'].unique())[:10])
print("[INFO] Available Conditions:", list(df['condition'].unique())[:10])

sample_drug = "Valsartan"
sample_condition = "Left Ventricular Dysfunction"
sample_review = "This medicine works really well with almost no side effects."

if sample_drug in le_drug.classes_ and sample_condition in le_cond.classes_:
    d_enc = le_drug.transform([sample_drug])[0]
    c_enc = le_cond.transform([sample_condition])[0]
    review_vec = tfidf.transform([sample_review])
    num_features = np.array([[d_enc, c_enc, 50, len(sample_review)]])
    features = hstack([num_features, review_vec])
    pred_rating = model.predict(features)[0]
    print(f"\n[DRUG] Predicted Effectiveness for {sample_drug} on {sample_condition}: {pred_rating:.1f}/10")
else:
    print("\n[WARNING] The given drug or condition was not found in the training data.")

# -----------------------
# STEP 9: VISUALIZATION
# -----------------------
plt.figure(figsize=(8,5))
sns.histplot(df['rating'], bins=20, kde=True)
plt.title("Distribution of Drug Effectiveness Ratings")
plt.xlabel("Rating (Effectiveness)")
plt.ylabel("Frequency")
plt.savefig("rating_distribution.png")
print("\n[SUCCESS] Saved: rating_distribution.png")
plt.close()

top_drugs = df['drugName'].value_counts().head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=top_drugs.values, y=top_drugs.index)
plt.title("Top 10 Most Reviewed Drugs")
plt.xlabel("Number of Reviews")
plt.ylabel("Drug Name")
plt.savefig("top_drugs.png")
print("[SUCCESS] Saved: top_drugs.png")
plt.close()

print("\n[SUCCESS] Script completed successfully!")

import joblib

# Save trained model and encoders
joblib.dump(model, "drug_model.pkl")
joblib.dump(le_drug, "le_drug.pkl")
joblib.dump(le_cond, "le_condition.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\n[SUCCESS] Model and encoders saved successfully!")
