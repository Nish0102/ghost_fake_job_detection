# 🕵️ Ghost & Fake Job Detector

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/) 
[![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-orange?logo=google-colab)](https://colab.research.google.com/) 
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-brightgreen?logo=scikit-learn)]

---

## 🚀 Project Overview
A machine learning system to **detect fake and ghost job postings** using NLP techniques.  
The project uses textual features (job description, title, company profile, requirements, benefits) and metadata signals to classify jobs as **Real or Fake**, and can be extended to detect ghost jobs.

---

## 🌟 Highlights

- TF-IDF based text vectorization  
- Logistic Regression classifier  
- Handles class imbalance for better fraud detection  
- Achieves high recall (~82%) for fraudulent postings  
- Supports text + metadata features for improved accuracy  

---

## 📊 Dataset
- Source: Fake Job Postings Dataset (Kaggle)  
- Records: ~17,880 job postings  
- Labels: `fraudulent` (0 = Real, 1 = Fake)  

> Dataset not included due to size and licensing.

---

## 🛠️ Tech Stack
- Python 3  
- Pandas, NumPy  
- Scikit-learn  
- TF-IDF Vectorizer  
- Google Colab  

---

## ▶️ How to Run
1. Upload `fake_job_postings.csv` to Colab or your local environment.  
2. Run notebook cells sequentially.  
3. Train the model and test predictions.  
4. Extend with ghost detection rules if desired.  

---

## 📌 Future Improvements
- Ghost job detection logic  
- Deep learning models (BERT)  
- Streamlit web interface for live testing  
- Model explainability  

---

⭐ If you find this useful, feel free to star this repo!
