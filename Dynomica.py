# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 05:42:11 2024

@author: mashhadcom.com
"""


import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor  # Use LGBMRegressor for regression

# Create a file upload button
uploaded_file = st.file_uploader("Choose a trained LightGBM regression model (.sav or .joblib):", type=["sav", "joblib"])

# If a file is uploaded, load the ML model
if uploaded_file is not None:
    try:
        # Load the ML model directly from the BytesIO object
        model = joblib.load(uploaded_file)

        # Check if the model is an LGBMRegressor model
        if not isinstance(model, LGBMRegressor):
            raise TypeError("The uploaded model is not a trained LightGBMRegressor model.")

        # Get the feature names from the model
        var_names = model.feature_name_

        # Create a box to show model information
        st.header("Model Information")
        st.write(f"Model Parameters: {model.get_params()}")
        st.write(f"Features: {var_names}")

        # Create sliders for each feature
        new_data = {}
        for var_name in var_names:
            point_val = st.slider(f'Select a point for {var_name} within the range:', 0, 100, 10)
            new_data[var_name] = point_val

        # Create a button to predict
        if st.button('Predict'):
            # Create DataFrame with selected values
            new_df = pd.DataFrame([new_data], index=[0])

            # Predict using the model
            prediction = model.predict(new_df)

            # Display the prediction
            st.write(f"Predicted Profit: {prediction[0]}")

    except Exception as e:
        st.write(f"Error loading the model: {e}")
else:
    st.write("Please upload a trained LightGBM regression model (.sav or .joblib).")


# Function to generate scenarios based on user inputs
def generate_scenarios(target_grade, controllers, controller_ranges, fixed_values, model):
    scenarios = []

    # Generate combinations of controller values
    for controller_values in np.array(np.meshgrid(*[range_vals for _, _, range_vals in controller_ranges])).T.reshape(-1, len(controllers)):
        scenario = {controller: value for controller, value in zip(controllers, controller_values)}
        scenario.update(fixed_values)
        # Convert scenario to DataFrame and predict using the model
        new_df = pd.DataFrame([scenario])
        predicted_grade = model.predict(new_df)[0]

        # Check if the predicted grade meets the target
        if predicted_grade >= target_grade:
            scenario['Predicted Grade'] = predicted_grade
            scenarios.append(scenario)

    return scenarios

# Check if a file is uploaded and the model is loaded
if uploaded_file is not None:
    # Your existing code for loading the model and displaying model information

    # Add checkbox for selecting control variables
    st.subheader("Select Control Variables:")
    selected_controllers = st.multiselect("Select control variables:", var_names)

    # Create sliders for setting ranges and automatically generating intermediate values
    controller_ranges = {}
    for controller in selected_controllers:
        min_val, max_val = st.slider(f"Select range for {controller}:", 0, 100, (0, 100))
        step_values = list(range(min_val, max_val + 1))  # Generate intermediate values directly
        controller_ranges[controller] = (min_val, max_val, step_values)

    # Create input boxes for setting fixed values for other variables
    fixed_values = {}
    for i, var_name in enumerate(var_names):
        if var_name not in selected_controllers:
            value = st.slider(f"Select value for {var_name}:", 0, 100, 50, key=f"fixed_slider_{i}")  # Add unique key
            fixed_values[var_name] = value

    # Create input box for setting target grade
    target_grade = st.number_input("Desired target grade:", min_value=0, max_value=100, step=1)

    # Button to generate scenarios
    if st.button("Generate Scenarios"):
        scenarios = generate_scenarios(target_grade, selected_controllers, controller_ranges.values(), fixed_values, model)

        # Display scenarios
        st.subheader(f"Scenarios with predicted grade >= {target_grade}")
        for scenario in scenarios:
            st.write(scenario)

    # Display scenarios in a table
    if scenarios:
        st.subheader(f"Scenarios with predicted grade >= {target_grade}")
        df_scenarios = pd.DataFrame(scenarios)
        st.table(df_scenarios)
    else:
        st.write("No scenarios found with predicted grade >= target grade.")
