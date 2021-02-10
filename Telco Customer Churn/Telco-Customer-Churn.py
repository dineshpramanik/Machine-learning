#Importing neccessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as po
import plotly.graph_objs as go
import pickle

#Import Customer Churn Dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

#Convert string values (Yes/No) of Churn column into (1/0)
churn = {'Yes': 1, 'No': 0}
df['Churn'].replace(churn, inplace= True)

#Replacing blank value with 'Nan' in 'TotalCharges' column
df['TotalCharges'].replace(' ', np.nan, inplace=True)

#Removing Null values from 'TotalCharges' column
df = df[df['TotalCharges'].notnull()]

#Reset the index
df = df.reset_index()[df.columns]

#Convert object datatype into float for 'TotalCharges' column
df['TotalCharges'] = df['TotalCharges'].astype(float)

#Convert 'No internet service' to 'No' for the below columns
replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for i in replace_cols:
    df[i].replace({'No internet service': 'No'}, inplace=True)

#Convert 'No phone service' to 'No' for the 'MultipleLines' columns
df['MultipleLines'].replace({'No phone service' : 'No'}, inplace=True)

categorical_features = df[['gender',  'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']]
numerical_features = df[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']]
target = df['Churn']

churn_labels = df['Churn'].value_counts().keys().tolist()
churn_values = df['Churn'].value_counts().values.tolist()

df.drop('customerID', axis=1, inplace=True)

#Performing OneHotEncoding using get_dummies method
df = pd.get_dummies(data=df, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
                                      'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                     'Contract', 'PaperlessBilling', 'PaymentMethod', 'InternetService'], drop_first=True)


#Importing StandardScaler library and creating an object of this
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()

#Perform feature scaling on 'tenure', 'MonthlyCharges', 'TotalCharges' in order to bring them in same scale
cols_for_scaling = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[cols_for_scaling]= sc.fit_transform(df[cols_for_scaling])

#Create Feature variable 'X' and Target variable 'y'

X= df.drop('Churn', axis=1)
y= df['Churn']

#Split the data into Training and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)

#Import SVM classification library
from sklearn.svm import SVC
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

#Importing various metrics libraries to evaluate the accuracy of models
from sklearn.metrics import confusion_matrix, accuracy_score

#Find Confusion_matrix and Accuracy_score
cm_svm = confusion_matrix(y_test, y_pred)
print('SVM: \n', cm_svm, '\n')

accuracy_svm = round(accuracy_score(y_test, y_pred) * 100, 2)
print('Accuracy score on SVM: ', accuracy_svm, '%')

filename = 'customer-churn-svm.pkl'
pickle.dump(svm, open(filename, 'wb'))
