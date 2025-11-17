from flask import Flask, render_template, request, redirect, flash, session, url_for
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import bcrypt
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# ---------------- MONGODB CONNECTION ----------------
client = MongoClient("mongodb://localhost:27017/")
db = client["smart_health_risk_advisor"]
users_collection = db["users"]
dataset_collection = db["dataset"]
history_collection = db["history"]

# ---------------- MODEL PATHS ----------------
MODEL_PATH = 'health_risk_model.pkl'
SCALER_PATH = 'scaler.pkl'


# ---------------- TRAIN MODEL ----------------
def train_model():
    df = pd.DataFrame(list(dataset_collection.find()))

    if df.empty:
        print("Dataset is EMPTY! Insert rows in MongoDB Compass.")
        return None, None

    if '_id' in df.columns:
        df = df.drop(columns=['_id'])

    df.columns = df.columns.str.strip()

    df['risk_label'] = df['cardio']  # 0=Low, 1=High

    # Dataset gender: 1=Male, 2=Female
    df['gender'] = df['gender'].map({1: 0, 2: 1})

    feature_cols = [
        'age', 'weight', 'height',
        'ap_hi', 'ap_lo',
        'cholesterol', 'gluc',
        'smoke', 'alco', 'active',
        'gender'
    ]

    X = df[feature_cols]
    y = df['risk_label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Model trained & saved successfully!")
    return model, scaler


# ---------------- LOAD MODEL ----------------
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    model, scaler = train_model()


# ---------------- MAIN ROUTES ----------------
@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))


# ---------------- REGISTER ----------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_pw = request.form['confirm_pw']

        if users_collection.find_one({'email': email}):
            flash("Email already registered!")
            return redirect(url_for('login'))

        if password != confirm_pw:
            flash("Passwords do not match!")
            return redirect(url_for('register'))

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

        users_collection.insert_one({
            "name": name,
            "email": email,
            "password": hashed_pw
        })

        flash("Registration successful!")
        return redirect(url_for('login'))

    return render_template("register.html")


# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = users_collection.find_one({'email': email})

        if user and bcrypt.checkpw(password.encode(), user['password']):
            session['user'] = email
            return redirect(url_for('home'))
        else:
            flash("Invalid login details")

    return render_template("login.html")


# ---------------- HOME ----------------
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("home.html", user=session['user'])


# ---------------- HEALTH FORM + PREDICTION ----------------
@app.route('/form', methods=['GET', 'POST'])
def form():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        age = int(request.form['Age'])
        raw_gender = request.form['Gender']  # Male/Female

        # Dataset style gender → 1/2 → mapped to 0/1
        gender = 0 if raw_gender == "Male" else 1

        weight = float(request.form['Weight'])
        height = float(request.form['Height'])

        # Get BP Values safely
        sys_bp = request.form.get('Systolic')
        dia_bp = request.form.get('Diastolic')

        # Convert None → default values
        if not sys_bp:
            sys_bp = 120
        if not dia_bp:
            dia_bp = 80

        ap_hi = int(sys_bp)
        ap_lo = int(dia_bp)


        # Cholesterol mapping
        chol = int(request.form['Cholesterol'])
        if chol < 200:
            cholesterol = 1
        elif 200 <= chol <= 239:
            cholesterol = 2
        else:
            cholesterol = 3

        # Sugar mapping
        sugar = int(request.form.get('Sugar', 0))
        if sugar < 140:
            gluc = 1
        elif 140 <= sugar <= 199:
            gluc = 2
        else:
            gluc = 3

        # Lifestyle (each selected individually)
        lifestyle = request.form['Lifestyle']
        smoke = 1 if lifestyle == "Smoker" else 0
        alco = 1 if lifestyle == "Alcoholic" else 0
        active = 1 if lifestyle == "Active" else 0

        # ---------------- MATCH DATASET ORDER ----------------
        input_values = np.array([[
            age, weight, height,
            ap_hi, ap_lo,
            cholesterol, gluc,
            smoke, alco, active,
            gender
        ]])

        input_scaled = scaler.transform(input_values)

        pred = model.predict(input_scaled)[0]
        prediction = "Low Risk" if pred == 0 else "High Risk"

        # Save history
        history_collection.insert_one({
            "user": session['user'],
            "prediction": prediction,
            "form_data": {
                "age": age,
                "gender": raw_gender,
                "weight": weight,
                "height": height,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": smoke,
                "alco": alco,
                "active": active
            }
        })

        return render_template("prediction.html", prediction=prediction)

    return render_template("form.html")


# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully!")
    return redirect(url_for('login'))


# ---------------- RUN APP ----------------
if __name__ == '__main__':
    app.run(debug=True)
