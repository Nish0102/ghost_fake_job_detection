# ghost_fake_job_detector.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# 1️⃣ Load Dataset
# -------------------------------
df = pd.read_csv("fake_job_postings.csv")

# -------------------------------
# 2️⃣ Preprocess Text
# -------------------------------
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
df[text_cols] = df[text_cols].fillna("")
df['text'] = df[text_cols].agg(" ".join, axis=1)

# -------------------------------
# 3️⃣ TF-IDF & Train/Test Split
# -------------------------------
X_text = df['text']
y = df['fraudulent']

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 4️⃣ Logistic Regression Model
# -------------------------------
model = LogisticRegression(max_iter=3000, class_weight='balanced')
model.fit(X_train, y_train)

# -------------------------------
# 5️⃣ Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 6️⃣ Predict Fake/Real Job
# -------------------------------
def predict_job(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "FAKE JOB ⚠️" if pred == 1 else "REAL JOB ✅"

# -------------------------------
# 7️⃣ Ghost Detection Functions
# -------------------------------
def ghost_score(job):
    score = 0
    if pd.isna(job['company_profile']) or job['company_profile'].strip() == "":
        score += 2
    if pd.isna(job['salary_range']) or job['salary_range'].strip() == "":
        score += 1
    if pd.isna(job['description']) or len(job['description'].strip()) < 50:
        score += 1
    if pd.isna(job['benefits']) or job['benefits'].strip() == "":
        score += 1
    if job['has_company_logo'] == 0:
        score += 1
    ghost_keywords = ["work from home", "earn money", "quick cash"]
    if any(word in str(job['title']).lower() for word in ghost_keywords):
        score += 1
    if job['telecommuting'] == 1:
        score += 1
    return score

def is_ghost(job):
    return "GHOST JOB 👻" if ghost_score(job) > 3 else "NOT GHOST ✅"

# -------------------------------
# 8️⃣ Apply Combined Detection to Dataset
# -------------------------------
def classify_job_row(job):
    fake_pred = predict_job(job['text'])
    ghost_pred = is_ghost(job)
    return pd.Series([fake_pred, ghost_pred])

df[['Fake/Real', 'Ghost']] = df.apply(classify_job_row, axis=1)

# -------------------------------
# 9️⃣ Save Results
# -------------------------------
df.to_csv("job_classification_results.csv", index=False)
print("✅ Full dataset predictions saved to 'job_classification_results.csv'")

# -------------------------------
# 10️⃣ Sample Check
# -------------------------------
print("\nSample 10 predictions:")
print(df[['title', 'Fake/Real', 'Ghost']].sample(10, random_state=42))
