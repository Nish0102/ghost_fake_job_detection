# -------------------------------
# 1️⃣ Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# -------------------------------
# 2️⃣ Load Dataset
# -------------------------------
# Make sure the CSV is in the same folder
df = pd.read_csv("fake_job_postings.csv")

# -------------------------------
# 3️⃣ Combine Text Columns
# -------------------------------
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
df[text_cols] = df[text_cols].fillna("")  # fill missing text
df['text'] = df[text_cols].agg(" ".join, axis=1)

# -------------------------------
# 4️⃣ Prepare Features and Labels
# -------------------------------
X_text = df['text']
y = df['fraudulent']

# -------------------------------
# 5️⃣ Vectorize Text
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(X_text)

# -------------------------------
# 6️⃣ Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 7️⃣ Train Logistic Regression Model (Balanced)
# -------------------------------
model = LogisticRegression(max_iter=3000, class_weight='balanced')
model.fit(X_train, y_train)

# -------------------------------
# 8️⃣ Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 9️⃣ Test Custom Job Input
# -------------------------------
def predict_job(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "FAKE JOB ⚠️" if pred == 1 else "REAL JOB ✅"

sample_job = """
Work from home. Earn $50,000 per week.
No experience required. Apply now!
"""

print("\nSample Prediction:")
print(predict_job(sample_job))

# -------------------------------
#  🔹 Optional: Save Model & Vectorizer
# -------------------------------
joblib.dump(model, "fake_job_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nModel and vectorizer saved as 'fake_job_model.pkl' and 'tfidf_vectorizer.pkl'")
