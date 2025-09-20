import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title('Bank Customer Churn Prediction')

st.header('Enter Customer Details:')

# Input fields for customer details
credit_score = st.slider('Credit Score', 350, 850, 650)
country = st.selectbox('Country', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Female', 'Male'])
age = st.slider('Age', 18, 92, 39)
tenure = st.slider('Tenure (years)', 0, 10, 5)
balance = st.number_input('Balance', value=0.0)
products_number = st.slider('Number of Products', 1, 4, 1)
credit_card = st.selectbox('Has Credit Card?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
active_member = st.selectbox('Is Active Member?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
estimated_salary = st.number_input('Estimated Salary', value=0.0)

# Create a DataFrame from user input
input_data = pd.DataFrame([[credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary]],
                          columns=['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary'])

# Apply one-hot encoding to match the training data
input_data = pd.get_dummies(input_data, columns=['country', 'gender'], drop_first=True)

# Ensure all columns from training data are present and in the same order
# This assumes you have a list of columns from your training data X_train
# For demonstration, let's assume the order based on your previous steps:
# 'customer_id', 'credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary', 'country_Germany', 'country_Spain', 'gender_Male'
# We need to handle the 'customer_id' column which was in X but not used for prediction
# Let's create a dummy customer_id for the input data
input_data['customer_id'] = 0 # Add a placeholder customer_id

# Define the expected column order based on your training data X_train (excluding churn)
# You might want to load the actual X_train columns from a file or store them
expected_columns = ['customer_id', 'credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary', 'country_Germany', 'country_Spain', 'gender_Male'] # Replace with actual X_train columns

# Reindex the input data to match the training data columns and fill missing columns with 0
input_data = input_data.reindex(columns=expected_columns, fill_value=0)


# Scale the numerical features using the loaded scaler
numerical_cols = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])


# Make prediction
if st.button('Predict Churn'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.error(f'The customer is likely to churn (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'The customer is unlikely to churn (Probability: {prediction_proba[0]:.2f})')
