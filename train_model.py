import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("📥 Loading dataset...")
df = pd.read_csv("employee_promotion_3000_rows.csv")

# Feature selection: drop noisy or low-informative columns (if any)
# For now, we keep all; modify based on feature importance if needed

# Label Encoding
print("🔠 Encoding categorical variables...")
le_js = LabelEncoder()
le_edu = LabelEncoder()

df['Job Satisfaction'] = le_js.fit_transform(df['Job Satisfaction'])
df['Education Level'] = le_edu.fit_transform(df['Education Level'])

# Save encoders
joblib.dump(le_js, "job_satisfaction_encoder.pkl")
joblib.dump(le_edu, "education_encoder.pkl")

# Features and target
X = df.drop("Promotion Status", axis=1)
y = df["Promotion Status"]

# Feature Scaling
print("⚖️ Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# Split data
print("✂️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# GridSearchCV for KNN
print("🔍 Performing hyperparameter tuning...")
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['distance'],  # 'distance' usually performs better
    'metric': ['euclidean', 'manhattan']
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# Best model
knn_model = grid.best_estimator_
print(f"✅ Best Hyperparameters: {grid.best_params_}")

# Evaluate
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Final Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(knn_model, "promotion_knn_model.pkl")
print("💾 Model saved as 'promotion_knn_model.pkl'") 