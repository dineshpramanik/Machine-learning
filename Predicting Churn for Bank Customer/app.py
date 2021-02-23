from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


model = pickle.load(open('Bank_Customer_Churn.pkl', 'rb'))

app = Flask(__name__)
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        CreditScore = int(request.form['CreditScore'])
        Age = int(request.form['Age'])
        Tenure = int(request.form['Tenure'])
        Balance = float(request.form['Balance'])
        NumOfProducts = int(request.form['NumOfProducts'])
        HasCrCard = int(request.form['HasCrCard'])
        IsActiveMember = int(request.form['IsActiveMember'])
        EstimatedSalary = float(request.form['EstimatedSalary'])

        Geography_Germany = request.form['Geography_Germany']
        if (Geography_Germany == 'Germany'):
            Geography_Germany = 1
            Geography_Spain = 0
            Geography_France = 0

        elif (Geography_Germany == 'Spain'):
            Geography_Germany = 0
            Geography_Spain = 1
            Geography_France = 0

        else:
            Geography_Germany = 0
            Geography_Spain = 0
            Geography_France = 1

        Gender_Male = request.form['Gender_Male']
        if (Gender_Male == 'Male'):
            Gender_Male = 1
            Gender_Female = 0
        else:
            Gender_Male = 0
            Gender_Female = 1

        my_prediction = model.predict([[CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard,
                                     IsActiveMember, EstimatedSalary, Geography_Germany, Geography_Spain, Gender_Male]])

    return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)