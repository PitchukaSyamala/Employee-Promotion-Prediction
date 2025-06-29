# 🚀 Employee Promotion Prediction – Flask Web App

A machine learning-based **Flask web application** that predicts whether an employee is likely to be **Promoted**, **Needs Review**, or **Not Promoted**. This tool assists HR departments in making more **data-driven and unbiased** decisions using historical employee data.

---

## 📄 Project Description

The Employee Promotion Prediction app uses features like years of service, job satisfaction, education level, training hours, and performance ratings to predict an employee’s promotion status.

Built with **Flask** and **scikit-learn**, it offers a lightweight, browser-accessible interface that runs locally through the terminal.

---

## 🎯 Objectives

- ✅ Automate and enhance employee promotion evaluations
- ✅ Reduce bias and subjectivity in decision-making
- ✅ Provide a browser-based interface using Flask

---

## 🧠 Key Features

- 🤖 **Machine Learning Model** (Logistic Regression / KNN)
- 🔤 **Categorical Encoding**: Converts strings to machine-readable formats
- 🌐 **Flask Interface**: Simple HTML forms for input and output
- 📊 **Real-Time Predictions**: Get results instantly in the browser
- 💾 **Encoders & Model Saved**: Using `joblib` for fast reuse


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
```bash 
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

```

### 📄 License
This project is licensed under the MIT License.

