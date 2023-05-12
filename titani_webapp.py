# Importing the necessary libraries
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

# Creating a Flask web application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the saved model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Get the input values from the user
    PassengerId = int(request.form['PassengerId'])
    Pclass = int(request.form['Pclass'])
    Age = int(request.form['Age'])
    SibSp = int(request.form['SibSp'])
    Parch = int(request.form['Parch'])
    Fare = float(request.form['Fare'])
    male = int(request.form['male'])
    Q = int(request.form['Q'])
    S = int(request.form['S'])

    # Make a prediction using the loaded model
    data = np.array([[PassengerId, Pclass, Age, SibSp, Parch, Fare, male, Q, S]])
  

    #load the scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    data = scaler.transform(data)
    prediction = model.predict(data)
    prediction = prediction[0][0]
    if prediction >= 0.5:
        prediction = 'Died'
    else: 
        prediction = 'Survived'  

    # Return the prediction to the user
    return render_template('results.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
