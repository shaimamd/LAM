import streamlit as st
import numpy as np
import joblib
import os

# Define paths for models and scalers (relative paths)
current_dir = os.path.dirname(__file__)
gpr_model_path = os.path.join(current_dir, 'gpr_model.pkl')
scaler_X_path = os.path.join(current_dir, 'scaler_X.pkl')
scaler_Y_path = os.path.join(current_dir, 'scaler_Y.pkl')

# Load the saved models and scalers
gpr = joblib.load(gpr_model_path)
scaler_X = joblib.load(scaler_X_path)
scaler_Y = joblib.load(scaler_Y_path)

# Title for the Streamlit app
st.title('Gaussian Process Regression for Laser Welding')

# Create columns to organize the sliders in a nice layout
col1, col2 = st.columns(2)

# First column - Laser Power P_L (KW)
with col1:
    P_L = st.slider("Laser Power $P_{L}$ (KW)", min_value=4.0, max_value=8.0, step=0.5)

# Second column - Welding Speed V_W (m/min)
with col2:
    V_W = st.slider("Welding Speed $V_{W}$ (m/min)", min_value=2.0, max_value=5.0, step=0.5)

# Third column - Laser Diameter D_L (mm)
with col1:
    D_L = st.slider("Laser Diameter $D_{L}$ (mm)", min_value=0.5, max_value=0.8, step=0.01)

# Fourth column - Focus Position f_p (mm)
with col2:
    f_p = st.slider("Focus position $f_{p}$ (mm)", min_value=-6.0, max_value=0.0, step=1.0)

# Prepare the user inputs into a numpy array for prediction
user_input = np.array([[P_L, V_W, D_L, f_p]])

# Scale the input using the loaded scaler
user_input_scaled = scaler_X.transform(user_input)

# Make the prediction using the loaded Gaussian Process model
y_pred_scaled, y_std_scaled = gpr.predict(user_input_scaled, return_std=True)

# Inverse the scaling for the target variable (Porosity %)
y_pred = scaler_Y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]
y_std = y_std_scaled[0] * scaler_Y.scale_[0]

# Display the prediction results
st.write(f"**Predicted Porosity**: {y_pred:.2f}%")
st.write(f"**Standard Deviation**: {y_std:.2f}%")

# Add a reference to the article or data
st.write("Reference: The model was trained using data from : Toward prediction and insight of porosity formation in laser welding: A physics-informed deep learning framework.")

