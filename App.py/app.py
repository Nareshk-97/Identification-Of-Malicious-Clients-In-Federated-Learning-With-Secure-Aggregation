from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import joblib
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)
app.secret_key = "fedgt_secret_key"

# Load model and metrics
model = joblib.load('model.pkl')
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        if username.lower() == "fedgt" and password == "Secure@3":
            session['user'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from form
        data = [float(request.form[f'field_{i}']) for i in range(6)]
        
        # Predict
        prediction = model.predict([data])[0]
        
        return render_template('index.html', prediction=prediction, inputs=data)
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', metrics=metrics)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
