# ================= Smart Health Risk Model Trainer (MongoDB) =================

from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ---------------- 1. CONNECT TO MONGODB ----------------
client = MongoClient("mongodb://localhost:27017/")
db = client["smart_health_risk_advisor"]
dataset_collection = db["dataset"]

# ---------------- 2. LOAD DATA ----------------
df = pd.DataFrame(list(dataset_collection.find()))
if df.empty:
    print("❌ Dataset in MongoDB is empty! Please add data.")
    exit()

# Drop MongoDB _id column if exists
if "_id" in df.columns:
    df = df.drop("_id", axis=1)

# ---------------- 3. PREPROCESS ----------------
df.columns = df.columns.str.strip()
df['risk_label'] = df['cardio']  # 0=Low, 1=High
df['gender'] = df['gender'].map({1:0, 2:1})  # Map Male=0, Female=1

feature_cols = [
    'age', 'weight', 'height',
    'ap_hi', 'ap_lo',
    'cholesterol', 'gluc',
    'smoke', 'alco', 'active',
    'gender'
]

X = df[feature_cols]
y = df['risk_label']

# ---------------- 4. TRAIN-TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------------- 5. SCALE FEATURES ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- 6. TRAIN RANDOM FOREST ----------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# ---------------- 7. CHECK ACCURACY ----------------
train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

print(f"🔹 Train Accuracy: {train_acc*100:.2f}%")
print(f"🔹 Test Accuracy: {test_acc*100:.2f}%")

# ---------------- 8. SAVE MODEL & SCALER ----------------
joblib.dump(model, "health_risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model saved as health_risk_model.pkl")
print("✅ Scaler saved as scaler.pkl")
