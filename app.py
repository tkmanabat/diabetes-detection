import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st

#Create Title
st.write("""
# Diabetes Detection 
Predict if someone has diabetes or not using Machine Learning
""")

#Load Data
df=pd.read_csv("diabetes.csv")
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


#Train test split
X=df.iloc[:,0:8].values
Y=df['Outcome']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

#Get User input
def getUserInfo():
    pregnancies=st.sidebar.slider('pregnancies',0,17,3)
    glucose=st.sidebar.slider('glucose',0,199,117)
    bloodPressure=st.sidebar.slider('bloodPressure',0,122,72)
    skinThickness=st.sidebar.slider('skinThickness',0,99,23)
    insulin=st.sidebar.slider('insulin',0.0,846.0,30.0)
    bmi=st.sidebar.slider('bmi',0.0,67.1,32.0)
    diabetesPedigreeFunction=st.sidebar.slider('diabetesPedigreeFunction',0.078,2.42,0.3725)
    age=st.sidebar.slider('age',21,81,29)

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

#Set subheader and display user input
st.subheader('User Input:')
st.write(userInput)

#Model Training(RandomForrestClassifier)
RFC=RandomForestClassifier()
RFC.fit(X_train,Y_train)

#Show model metrics
st.subheader("Model Test Accuracy Score:")
st.write(str(accuracy_score(Y_test,RFC.predict(X_test))*100)+'%')

#Store model predictions
prediction=RFC.predict(userInput)
predictionProbability=RFC.predict_proba(userInput)

#Subheader classification display
st.subheader('Classification: ')
st.write(prediction)
st.write(predictionProbability)