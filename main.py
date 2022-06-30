from math import ldexp
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open('LogModel.pkl','rb'))

scaler = pickle.load(open('scaler.pkl','rb'))

@app.route("/")
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations = locations)

@app.route('/predict', methods = ['POST'])
def predict():
    age = request.form.get('age')
    gender = request.form.get('gender')
    pain_type = request.form.get('pain_type') 
    bps = int(request.form.get('bps'))
    chol = int(request.form.get('chol'))
    fbs = float(request.form.get('fbs'))
    ecg = float(request.form.get('ecg'))
    thalach =  float(request.form.get('thalach')) 
    exang =  float(request.form.get('exang'))
    oldpeak =  float(request.form.get('oldpeak'))
    slope =  float(request.form.get('slope'))
    ca =  float(request.form.get('ca')) 
    thal =  float(request.form.get('thal')) 


    print(age, gender, pain_type,bps,chol, fbs, ecg, thalach, exang, oldpeak, slope, ca, thal)
    input = np.array([[age, gender, pain_type,bps,chol, fbs, ecg, thalach, exang, oldpeak, slope, ca, thal]])
    print(input)
    input = scaler.transform(input)
    # print("Input : ",input)

    # prediction = pipe.predict(input)[0] 
    proba = pipe.predict_proba(input)[0]
    print(proba)
    print(np.round(proba[1],5))
    return str(np.round(proba[1]*100,5))

if __name__ == '__main__':
    app.run(debug = True, host = "0.0.0.0", port = 9696)