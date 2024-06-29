import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('MedIns.pkl', 'rb'))

# Streamlit app title
st.title("Medical Insurance Predictor")


# Function to handle input and conversion
def get_input(prompt, dtype):
    value = st.text_input(prompt)
    try:
        return dtype(value)
    except ValueError:
        st.error(f"Invalid input for {prompt}. Expected {dtype.__name__}.")
        return None


# Collect user inputs with error handling
input_age = get_input("Enter age:", int)
input_sex = get_input("Enter sex: (male: 0; female: 1)", int)
input_bmi = get_input("Enter body mass index:", float)
input_children = get_input("Enter number of children:", int)
input_smoker = get_input("Enter 0 if smoker, 1 if not:", int)
input_region = get_input("Enter region: (southeast: 0; southwest: 1; northeast: 2; northwest: 3)", int)

# Check if all inputs are valid before making a prediction
if None not in (input_age, input_sex, input_bmi, input_children, input_smoker, input_region):
    input_data = (input_age, input_sex, input_bmi, input_children, input_smoker, input_region)

    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make the prediction
    prediction = model.predict(input_data_reshaped)

    # Display the prediction result
    st.success(f'The insurance cost is USD {prediction[0]:.2f}')
else:
    st.warning("Please provide valid inputs for all fields.")
