# 🚀 Employee Promotion Prediction – Streamlit App

A machine learning-powered **Streamlit web application** that predicts whether an employee is likely to be **Promoted**, **Needs Review**, or **Not Promoted** based on their profile. This tool supports HR professionals by providing data-driven and unbiased promotion recommendations.

---

## 📄 Project Description

Employee Promotion Prediction is an interactive app designed to assist HR departments in forecasting employee promotions. By analyzing attributes such as:

- Years of experience
- Job satisfaction
- Number of projects
- Training hours
- Performance ratings
- Education level
- Overtime status

...the model predicts the promotion outcome based on historical data.

The app uses a **Logistic Regression model** trained on a labeled dataset. With its user-friendly Streamlit UI, HR managers can quickly input employee details and receive predictions instantly.

---

## 🎯 Objectives

- ✅ Automate and enhance the employee promotion decision-making process  
- ✅ Reduce bias and increase transparency in evaluations  
- ✅ Provide a lightweight, web-based predictor using Streamlit

---

## 🧠 Key Features

- 📈 **Logistic Regression Model**: Trained for multi-class prediction
- 🔢 **Categorical Encoding**: Transforms values like education level and satisfaction into numerical form
- 🧮 **Real-Time Prediction**: Instant results based on input attributes
- 🖥️ **Streamlit UI**: Clean, easy-to-use interface — runs directly in the browser
- 📁 **Model & Encoder Files**: Saved with `joblib` for reuse without retraining

---

## 🗂️ Prediction Classes

- ✅ **Promoted**
- ⚠️ **Needs Review**
- ❌ **Not Promoted**

---

## 🏗️ Tech Stack

| Component     | Technology           |
|---------------|----------------------|
| Language       | Python 3.x           |
| ML Libraries   | scikit-learn, NumPy, pandas |
| Web Framework  | Streamlit            |
| Model Storage  | joblib               |
| Encoding       | LabelEncoder         |

---

## 📦 How to Run the App Locally

### 🔧 Prerequisites

- Python 3.7+
- Required libraries:  
  ```bash
  pip install streamlit pandas scikit-learn joblib

### ▶️ Run the App

python app.py

Then open your browser and go to:
 http://127.0.0.1:5000

### 📁 Project Structure
├── app.py                        # Flask web app
├── train_model.py                # Script to train and save the model
├── employee_promotion_3000_rows.csv
├── promotion_knn_model.pkl       # Trained KNN model
├── job_satisfaction_encoder.pkl  # Encoder for satisfaction level
├── education_encoder.pkl         # Encoder for education level
├── scaler.pkl                    # StandardScaler (optional)
├── requirements.txt              # Project dependencies
└── templates/
    ├── index.html                # Input form
    └── result.html               # Result display

### 📄 License
This project is licensed under the MIT License.

