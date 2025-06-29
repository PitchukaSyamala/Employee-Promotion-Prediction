from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load("promotion_knn_model.pkl")
js_encoder = joblib.load("job_satisfaction_encoder.pkl")
edu_encoder = joblib.load("education_encoder.pkl")

@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        years = int(request.form['years'])
        projects = int(request.form['projects'])
        satisfaction = js_encoder.transform([request.form['satisfaction']])[0]
        training = int(request.form['training'])
        ratings = int(request.form['ratings'])
        overtime = int(request.form['overtime'])
        education = edu_encoder.transform([request.form['education']])[0]

        features = np.array([[years, projects, satisfaction, training, ratings, overtime, education]])

        # Predict using the model
        prediction = model.predict(features)[0]
        
        # Redirect to result page with the prediction
        return render_template("result.html", prediction=prediction)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
