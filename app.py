import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st


KNN=pickle.load(open("./models/KNN.sav","rb"))
LR=pickle.load(open("./models/LR.sav","rb"))
RF=pickle.load(open("./models/RF.sav","rb"))

st.set_option('deprecation.showPyplotGlobalUse', False)
#Create Title
st.write("""
# Diabetes Detection 
Predict if someone has diabetes or not using Machine Learning
""")

#image
image=Image.open('image.jpg')
st.image(image, use_column_width=True)

#Load Data
df=pd.read_csv("diabetes.csv")
#clean missing values with reference to their distribution
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].median(), inplace = True)



#Subheader
st.write('## Dataset Information:')
st.write('This dataset was derived from Kaggle. It is originally from the **National Institute of Diabetes and Digestive and Kidney Diseases**. This will be used to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.')
#Show data as table
st.write('## Exploratory Data Analysis:')
st.write('### Dataset Head')
st.dataframe(df[0:5])
#Show stats
st.write('### Descriptive Statistics')
st.write(df.describe())
#Chart
chart=st.bar_chart(df)



st.write('### Distributions of each feature')
histogram=df.hist(figsize=(20,20))
plt.show
st.pyplot()

st.write('### Outcomes of the Dataset')
st.write("**Legend** ")
st.write("0 - No Diabetes ")
st.write("1 - With Diabetes")
p=df.Outcome.value_counts().plot(kind="bar")
plt.show()
st.pyplot()
st.write('The graph above shows that the data is biased towards datapoints having outcome value as 0 where it means that diabetes was not present actually. The number of non-diabetics is almost twice the number of diabetic patient.')


st.sidebar.header('User Input')
st.sidebar.write('Predict whether you have diabetes or not by entering the parameters. The results are located at the bottom of the page')
option = st.sidebar.selectbox('Select your Machine Learning Model', ('K Nearest Neighbors', 'Logistic Regression', 'Random Forest'))



#Get User input
def getUserInfo():
    pregnancies=st.sidebar.slider('Pregnancies had',0,17,3)
    glucose=st.sidebar.slider('Plasma Glucose Concentration (mg/dl)',0,199,117)
    bloodPressure=st.sidebar.slider('Diastolic Blood Pressure (mm Hg)',0,122,72)
    skinThickness=st.sidebar.slider('Triceps skin fold thickness (mm)',0,99,23)
    insulin=st.sidebar.slider('Serum Insulin (U/ml)',0.0,846.0,30.0)
    bmi=st.sidebar.slider('Body Mass Index (BMI)',0.0,67.1,32.0)
    diabetesPedigreeFunction=st.sidebar.slider('Diabetes Pedigree Function',0.078,2.42,0.3725)
    age=st.sidebar.slider('Age',21,81,29)
    
    #Store into dictionary
    userData={'pregnancies':pregnancies,
    'glucose':glucose,
    'bloodPressure':bloodPressure,
    'skinThickness':skinThickness,
    'insulin':insulin,
    'bmi':bmi,
    'diabetesPedigreeFunction':diabetesPedigreeFunction,
    'age':age}

    #Transform to DF
    features=pd.DataFrame(userData,index=[0])
    return features

#Store user input to variable
userInput=getUserInfo()


if option=='K Nearest Neighbors':
    accuracy='71%'
    prediction=KNN.predict(userInput)
    predictionProbability=KNN.predict_proba(userInput)
elif option=='Logistic Regression':
    accuracy='73%'
    prediction=LR.predict(userInput)
    predictionProbability=LR.predict_proba(userInput)
elif option=='Random Forest':
    accuracy='74%'
    prediction=RF.predict(userInput)
    predictionProbability=RF.predict_proba(userInput)





#Set subheader and display user input
st.header('User Input and Prediction:')
st.write(userInput)

st.subheader(f'Model Selected: {option}')
st.write(f"Model Accuracy: {accuracy}")
#Subheader classification display
st.subheader('Prediction: ')
st.write(prediction)
st.subheader('Prediction Probability: ')
st.write(predictionProbability)