import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean TotalCharges (some are blank)
print("Cleaning data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Handle No internet service and No phone service - convert to No for consistency
for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies']:
    df[col] = df[col].replace('No internet service', 'No')
    
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

# Encode binary columns
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# One-hot encode multi-category columns
categorical_cols = [
    'InternetService', 'Contract', 'PaymentMethod', 'MultipleLines',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies'
]

print("Encoding categorical features...")
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"Dataset shape: {X.shape}")
print(f"Features: {len(X.columns)}")
print(f"Target distribution: {y.value_counts(normalize=True).round(2).to_dict()}")

# Train-test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train XGBoost model
print("Training XGBoost model...")
model = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss', 
    random_state=42,
    n_estimators=100,
    max_depth=5
)
model.fit(X_train, y_train)

# Save model and feature column names
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("Model trained and saved!")
print(f"Training Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")
print(f"Model uses {len(X.columns)} features")

# Save feature names for reference
feature_names = X.columns.tolist()
with open("models/feature_names.txt", "w") as f:
    for i, feature in enumerate(feature_names):
        f.write(f"{i}: {feature}\n")
        
print("Feature names saved to models/feature_names.txt")
print("Ready to run churn_app.py!")