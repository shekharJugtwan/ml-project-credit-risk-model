# 📊 Credit Risk Modeling – Machine Learning Project

## 🚀 Overview
This project is an end-to-end **Credit Risk Modeling system** that predicts the probability of loan default using machine learning.  
It helps financial institutions assess borrower risk and make better lending decisions.

The model is trained using **XGBoost** and optimized using **Optuna**, then deployed as an interactive web app using **Streamlit**.

🔗 Live Demo: https://ml-project-credit-risk-model-sj.streamlit.app/

---

## 🎯 Problem Statement
Banks and financial institutions need to evaluate whether a loan applicant is likely to default.  
This project builds a classification model to predict credit risk based on applicant financial and personal attributes.

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- XGBoost ⚡
- Optuna (Hyperparameter Tuning)
- Matplotlib & Seaborn (EDA & Visualization)
- Joblib (Model Saving)
- Streamlit (Deployment)

---

## 📂 Project Workflow

### 1. Data Understanding
- Explored dataset structure
- Identified key features affecting credit risk

### 2. Data Preprocessing
- Handled missing values
- Treated outliers
- Feature engineering
- Encoding categorical variables

### 3. Exploratory Data Analysis (EDA)
- Analyzed distributions
- Identified patterns in defaulters vs non-defaulters
- Correlation analysis

### 4. Model Building
- Trained multiple classification models
- Selected **XGBoost** as the best-performing model

### 5. Hyperparameter Tuning
- Used **Optuna** for optimization
- Improved model performance and stability

### 6. Model Deployment
- Built interactive web app using **Streamlit**
- Deployed trained model for real-time predictions

---

## 📊 Model Performance
- Optimized using AUC / Accuracy / F1-score (based on evaluation setup)
- XGBoost performed best after tuning

---

## 📁 How to Run Locally

```bash
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model

pip install -r requirements.txt

streamlit run main.py
