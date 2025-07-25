

from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- Load the trained models ---
rf_model = None
lr_model = None
try:
    rf_model = joblib.load("rf_model.pkl")
    print("RandomForestClassifier model loaded successfully.")
except FileNotFoundError:
    print("Error: RandomForestClassifier model 'rf_model.pkl' not found.")
    print("Please run 'train_model.py' first to generate the models.")

try:
    lr_model = joblib.load("lr_model.pkl")
    print("LinearRegression model loaded successfully.")
except FileNotFoundError:
    print("Error: LinearRegression model 'lr_model.pkl' not found.")
    print("Please run 'train_model.py' first to generate the models.")

# Define a more appropriate threshold for converting Linear Regression performance_rate output
# Based on train_model.py, 'unsafe' performance_rate is typically 5-50, 'safe' is 50-95.
# So, a value like 55 (or 50) makes sense: performance_rate < 55 implies unsafe.
LR_THRESHOLD = 55 # Performance rate BELOW this value will be classified as "Unsafe"

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if rf_model is None or lr_model is None:
        return jsonify({"error": "One or both models not loaded. Please train the models first."}), 500

    try:
        data = request.get_json(force=True)
        
        # Extract features
        temperature = data.get('temperature')
        voltage = data.get('voltage')
        current = data.get('current')
        charge_cycles = data.get('charge_cycles')

        if None in [temperature, voltage, current, charge_cycles]:
            return jsonify({"error": "Missing one or more required parameters (temperature, voltage, current, charge_cycles)"}), 400

        # Create a Pandas DataFrame with feature names to match training data
        features_df = pd.DataFrame([[
            float(temperature),
            float(voltage),
            float(current),
            int(charge_cycles)
        ]], columns=['temperature', 'voltage', 'current', 'charge_cycles'])
        
        # --- RandomForestClassifier Prediction ---
        rf_prediction_binary = int(rf_model.predict(features_df)[0])
        rf_prediction_proba = float(rf_model.predict_proba(features_df)[:, 1][0])
        rf_status = "Unsafe" if rf_prediction_binary == 1 else "Safe"

        # --- LinearRegression Prediction ---
        lr_prediction_raw = float(lr_model.predict(features_df)[0])
        # Logic for LR: If predicted performance_rate is BELOW the threshold, it's Unsafe
        lr_prediction_binary = int(lr_prediction_raw < LR_THRESHOLD) # Corrected logic: less than threshold is unsafe
        lr_status = "Unsafe" if lr_prediction_binary == 1 else "Safe"

        # Return predictions from both models
        response_data = {
            'rf_burn_risk_binary': rf_prediction_binary,
            'rf_burn_risk_proba': rf_prediction_proba,
            'rf_status': rf_status,
            'lr_burn_risk_raw': lr_prediction_raw,
            'lr_burn_risk_binary': lr_prediction_binary,
            'lr_status': lr_status
        }
        return jsonify(response_data)

    except Exception as e:
        # Log the full traceback to the console for debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "message": "An error occurred during prediction. Check server logs."}), 500

# Run the Flask application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

