# # train_model_fixed.py

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# import joblib

# # ---- Step 1: Generate a balanced dataset ----

# np.random.seed(42)

# # 500 Safe (burn_risk = 0)
# safe = pd.DataFrame({
#     'temperature': np.random.normal(35, 3, 500),
#     'voltage': np.random.normal(48, 1, 500),
#     'current': np.random.normal(2.5, 0.5, 500),
#     'charge_cycles': np.random.randint(100, 800, 500),
#     'burn_risk': 0
# })

# # 500 Unsafe (burn_risk = 1)
# unsafe = pd.DataFrame({
#     'temperature': np.random.normal(60, 5, 500),
#     'voltage': np.random.normal(45, 1.5, 500),
#     'current': np.random.normal(5.5, 1.0, 500),
#     'charge_cycles': np.random.randint(900, 1500, 500),
#     'burn_risk': 1
# })

# # Combine and shuffle
# df = pd.concat([safe, unsafe]).sample(frac=1).reset_index(drop=True)

# # ---- Step 2: Train the model ----
# X = df.drop("burn_risk", axis=1)
# y = df["burn_risk"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# # ---- Step 3: Evaluate ----
# y_pred = model.predict(X_test)
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # ---- Step 4: Save the model ----
# joblib.dump(model, "battery_model.pkl")
# print("✅ Model saved as battery_model.pkl")
# train_model.py (Final Version - Further Enhanced Extreme Temperature Handling and LR Classification Metrics)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
import json
import os

np.random.seed(42)

# Define a consistent threshold for binarizing Linear Regression output
# This should match the LR_THRESHOLD in app.py for consistency
LR_THRESHOLD_FOR_CLASSIFICATION = 55 

# --- Generate a more balanced and complex dataset ---
num_safe = 1500
num_unsafe_normal = 1000 # Unsafe conditions within somewhat typical ranges
num_unsafe_extreme_temp = 1000 # Further increased explicitly unsafe due to extreme temp samples (from 700 to 1000)

# Safe (burn_risk = 0) - Data for ideal operating conditions
safe_data = pd.DataFrame({
    'temperature': np.random.normal(30, 4, num_safe),
    'voltage': np.random.normal(50, 1.5, num_safe),
    'current': np.random.normal(2.0, 1.0, num_safe),
    'charge_cycles': np.random.randint(50, 800, num_safe),
    'burn_risk': 0
})

# Unsafe (burn_risk = 1) - Data for high-risk conditions (normal range)
unsafe_normal_data = pd.DataFrame({
    'temperature': np.random.normal(50, 6, num_unsafe_normal),
    'voltage': np.random.normal(46, 2.5, num_unsafe_normal),
    'current': np.random.normal(8.0, 2.0, num_unsafe_normal),
    'charge_cycles': np.random.randint(700, 2000, num_unsafe_normal),
    'burn_risk': 1
})

# Unsafe (burn_risk = 1) - Data for EXTREME temperature conditions
# Ensure temperatures > 70-80 are consistently flagged as unsafe
unsafe_extreme_temp_data = pd.DataFrame({
    'temperature': np.random.normal(130, 25, num_unsafe_extreme_temp), # Higher mean (130 from 110), wider std dev (25 from 20) to ensure values like 120+ are very common
    'voltage': np.random.normal(35, 7, num_unsafe_extreme_temp), # Voltage might drop or be erratic
    'current': np.random.normal(15.0, 5.0, num_unsafe_extreme_temp), # Even higher current due to short or thermal runaway
    'charge_cycles': np.random.randint(1800, 3500, num_unsafe_extreme_temp), # High cycles
    'burn_risk': 1
})

# Combine all data and shuffle
df = pd.concat([safe_data, unsafe_normal_data, unsafe_extreme_temp_data]).sample(frac=1).reset_index(drop=True)

# Generate 'performance_rate' based on burn_risk, with clear separation and interaction
# This formula makes 'performance_rate' generally high for safe and low for unsafe
df['performance_rate'] = 0.0 # Initialize

# For Safe data (burn_risk == 0) - Higher performance rates
safe_indices = df['burn_risk'] == 0
df.loc[safe_indices, 'performance_rate'] = np.clip(
    80 - (df.loc[safe_indices, 'temperature'] * 0.5) + \
    (df.loc[safe_indices, 'voltage'] * 0.8) - \
    (df.loc[safe_indices, 'current'] * 0.7) - \
    (df.loc[safe_indices, 'charge_cycles'] * 0.01) + \
    np.random.normal(0, 5, safe_indices.sum()),
    55, 100 # Adjusted range for clear 'Safe' performance
)

# For Unsafe data (burn_risk == 1) - Lower performance rates
unsafe_indices = df['burn_risk'] == 1
df.loc[unsafe_indices, 'performance_rate'] = np.clip(
    20 + (df.loc[unsafe_indices, 'temperature'] * 0.3) - \
    (df.loc[unsafe_indices, 'voltage'] * 1.0) + \
    (df.loc[unsafe_indices, 'current'] * 1.5) + \
    (df.loc[unsafe_indices, 'charge_cycles'] * 0.02) + \
    np.random.normal(0, 8, unsafe_indices.sum()),
    0, 50 # Adjusted range for clear 'Unsafe' performance
)

# Explicitly ensure extreme temperatures are always associated with very low performance rates and marked unsafe
extreme_temp_threshold = 85 # Raised threshold slightly to 85, from 90, to catch more
extreme_temp_indices = df['temperature'] > extreme_temp_threshold
df.loc[extreme_temp_indices, 'performance_rate'] = np.clip(
    df.loc[extreme_temp_indices, 'performance_rate'] * 0.1 - (df.loc[extreme_temp_indices, 'temperature'] - extreme_temp_threshold) * 5, # Make it drop even more sharply
    -20, 5 # Force very, very low performance rate for very high temps, allow some negative
)
df.loc[extreme_temp_indices, 'burn_risk'] = 1 # Explicitly mark as unsafe if temp is too high


# --- Prepare data for model training ---
X = df[['temperature', 'voltage', 'current', 'charge_cycles']]
y_clf = df["burn_risk"] # Target for classification (Random Forest)
y_reg = df["performance_rate"] # Target for regression (Linear Regression)

# Split data for Classification Model
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# Split data for Regression Model (using overall data split for consistency of features)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)


# --- Train the models ---
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
rf_model.fit(X_train_clf, y_train_clf)
print("✅ RandomForestClassifier trained.")

lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)
print("✅ LinearRegression model trained.")

# --- Evaluate and save metrics ---
evaluation_data = {}

# Evaluate Random Forest (Classification)
y_pred_clf = rf_model.predict(X_test_clf)
y_proba_clf = rf_model.predict_proba(X_test_clf)[:, 1]

evaluation_data['rf_classification_report'] = classification_report(y_test_clf, y_pred_clf, output_dict=True)
evaluation_data['rf_confusion_matrix'] = confusion_matrix(y_test_clf, y_pred_clf).tolist()

fpr, tpr, thresholds = roc_curve(y_test_clf, y_proba_clf)
evaluation_data['rf_roc_curve'] = {
    'fpr': fpr.tolist(),
    'tpr': tpr.tolist(),
    'auc': auc(fpr, tpr)
}
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
evaluation_data['rf_feature_importances'] = feature_importances.to_dict()

# Evaluate Linear Regression (Regression Metrics first)
y_pred_reg_lr = lr_model.predict(X_test_reg) # These are the raw continuous predictions

evaluation_data['lr_metrics'] = {
    'mae': mean_absolute_error(y_test_reg, y_pred_reg_lr),
    'mse': mean_squared_error(y_test_reg, y_pred_reg_lr),
    'r2': r2_score(y_test_reg, y_pred_reg_lr)
}
evaluation_data['lr_actual_vs_predicted'] = {
    'actual': y_test_reg.tolist(),
    'predicted': y_pred_reg_lr.tolist() # Store the raw continuous predictions
}

# Evaluate Linear Regression as a Classifier (using the threshold)
# Binarize LR prediction for classification metrics using the defined threshold
y_pred_lr_binary = (y_pred_reg_lr < LR_THRESHOLD_FOR_CLASSIFICATION).astype(int) # Lower performance_rate means unsafe (1)

# Ensure y_test_clf is used for classification metrics comparison
evaluation_data['lr_classification_report'] = classification_report(y_test_clf, y_pred_lr_binary, output_dict=True, zero_division=0)
evaluation_data['lr_confusion_matrix'] = confusion_matrix(y_test_clf, y_pred_lr_binary).tolist()

# ROC Curve for Linear Regression (treating its continuous output as a score)
# The `y_score` for roc_curve should be higher for the positive class (unsafe = 1).
# Since lower performance_rate indicates unsafe, we use (max_possible_performance_rate - predicted_performance_rate) as a 'risk score'.
max_performance_rate = 100 # Based on our clipping range in data generation
lr_risk_scores = max_performance_rate - y_pred_reg_lr
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test_clf, lr_risk_scores) 
evaluation_data['lr_roc_curve'] = {
    'fpr': fpr_lr.tolist(),
    'tpr': tpr_lr.tolist(),
    'auc': auc(fpr, tpr)
}


# --- Save models and evaluation data ---
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(lr_model, "lr_model.pkl")
print("✅ Models saved as rf_model.pkl and lr_model.pkl")

with open("model_evaluation_data.json", "w") as f:
    json.dump(evaluation_data, f, indent=4)
print("✅ Model evaluation data saved as model_evaluation_data.json")

