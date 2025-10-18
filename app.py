
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the dataset
df = pd.read_csv('heart_disease_data.csv')

# Separate features and target
X = df.drop(columns='target', axis=1)
Y = df['target']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit app
st.title('Heart Disease Prediction')

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=25)

with col2:
    sex = st.selectbox('Sex', ('Male', 'Female'))

with col3:
    cp = st.selectbox('Chest Pain Type', (0, 1, 2, 3))

with col1:
    trestbps = st.number_input('Resting Blood Pressure', min_value=1, max_value=300, value=120)

with col2:
    chol = st.number_input('Cholesterol', min_value=1, max_value=600, value=200)

with col3:
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ('True', 'False'))

with col1:
    restecg = st.selectbox('Resting Electrocardiographic Results', (0, 1, 2))

with col2:
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=1, max_value=250, value=150)

with col3:
    exang = st.selectbox('Exercise Induced Angina', ('Yes', 'No'))

with col1:
    oldpeak = st.number_input('ST depression induced by exercise relative to rest', min_value=0.0, max_value=10.0, value=1.0)

with col2:
    slope = st.selectbox('Slope of the peak exercise ST segment', (0, 1, 2))

with col3:
    ca = st.selectbox('Number of major vessels colored by flourosopy', (0, 1, 2, 3, 4))

with col1:
    thal = st.selectbox('Thal', (0, 1, 2, 3))

# Prediction
if st.button('Predict'):
    sex = 1 if sex == 'Male' else 0
    fbs = 1 if fbs == 'True' else 0
    exang = 1 if exang == 'Yes' else 0

    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)

    if prediction[0] == 0:
        st.success('The person does not have a Heart Disease')
    else:
        st.error('The person has Heart Disease')
