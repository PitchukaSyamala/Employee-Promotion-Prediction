import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("employee_promotion_500_rows.csv")

# Encode categorical columns
label_enc_job = LabelEncoder()
label_enc_edu = LabelEncoder()
df["Job Satisfaction"] = label_enc_job.fit_transform(df["Job Satisfaction"])
df["Education Level"] = label_enc_edu.fit_transform(df["Education Level"])

# Define features and target variable
X = df.drop(columns=["Promotion Status"])
y = df["Promotion Status"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Prediction Labels
prediction_labels = {0: "Promoted ✅", 1: "Needs Review ⚠️", 2: "Not Promoted ❌"}

# Streamlit UI

# Title
st.title("📊 Employee Promotion Predictor")

# Input Fields
st.subheader("🔍 Enter Employee Details")

# Form for input fields
years = st.number_input("Years at Company:", min_value=0, max_value=50, step=1)
projects = st.number_input("Projects Completed:", min_value=0, max_value=100, step=1)
ratings = st.slider("Manager Ratings (1-5):", 1, 5)
overtime = st.number_input("Overtime Hours per Week:", min_value=0, max_value=100, step=1)

# Dropdowns for categorical variables
job_satisfaction = st.selectbox("Job Satisfaction:", ["Bad", "Average", "Good", "Very Good", "Excellent"])
training = st.selectbox("Training Received?", ["No", "Yes"])
education = st.selectbox("Education Level:", ["High School", "Bachelor's", "Master's", "PhD"])

# Predict Button
if st.button("🔮 Predict Promotion"):
    try:
        # Encode categorical inputs
        job_satisfaction_encoded = label_enc_job.transform([job_satisfaction])[0]
        education_encoded = label_enc_edu.transform([education])[0]
        training_val = 1 if training == "Yes" else 0

        # Create input array
        user_input = np.array([[years, projects, job_satisfaction_encoded,
                                training_val, ratings, overtime, education_encoded]])

        # Make prediction
        prediction = model.predict(user_input)[0]
        result = prediction_labels[prediction]

        # Display prediction result
        st.success(f"🎯 Prediction: {result}")

        # Check if prediction is positive (Promoted ✅)
        if prediction == 0:  # If predicted as Promoted (0)
            st.markdown("### 🔥 Positive Prediction: The employee is promoted!")
        else:
            st.markdown("### ❌ Negative Prediction: The employee is not promoted.")

        # Show a randomly generated test set for fun
      

        

        # Display prediction in a table
        st.subheader("📄 Prediction Results")
        result_data = {
            "Years": years,
            "Projects": projects,
            "Ratings": ratings,
            "Overtime": overtime,
            "Satisfaction": job_satisfaction,
            "Training": training,
            "Education": education,
            "Result": result
        }

        st.write(pd.DataFrame([result_data]))

    except Exception as e:
        st.error("Error: Invalid input! Please check your entries.")
