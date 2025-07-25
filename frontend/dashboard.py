
# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
# import numpy as np
# import json
# import os

# st.set_page_config(layout="wide", page_title="Ola S1 Battery Safety Monitor", page_icon="⚡")

# # --- Function to load CSV data ---
# @st.cache_data
# def load_csv_data(file_path):
#     try:
#         df = pd.read_csv(file_path)
#         df.columns = df.columns.str.strip().str.lower()
        
#         # Add a dummy 'burn_risk' column for visualization if it's not present.
#         if 'burn_risk' not in df.columns:
#             st.warning("No 'burn_risk' column found in CSV. Generating dummy risk for visualization.")
#             df['burn_risk'] = ((df['temperature'] > df['temperature'].mean() + 1.5 * df['temperature'].std()) |
#                                (df['voltage'] < df['voltage'].mean() - 1.5 * df['voltage'].std()) |
#                                (df['current'] > df['current'].mean() + 1.5 * df['current'].std()) |
#                                (df['charge_cycles'] > df['charge_cycles'].mean() + 1.5 * df['charge_cycles'].std())).astype(int)
        
#         required_cols = ['temperature', 'voltage', 'current', 'charge_cycles']
#         for col in required_cols:
#             if col not in df.columns:
#                 st.error(f"Missing required column for model prediction and visualization: '{col}'. Please check your CSV.")
#                 return pd.DataFrame()
#         return df
#     except FileNotFoundError:
#         st.error(f"Error: The file '{file_path}' was not found. Please ensure 'battery_data.csv' is in the same directory as this script.")
#         return pd.DataFrame()
#     except Exception as e:
#         st.error(f"An error occurred while loading or processing the CSV: {e}")
#         return pd.DataFrame()

# # --- Function to load evaluation data ---
# @st.cache_data
# def load_evaluation_data(file_path="model_evaluation_data.json"):
#     try:
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#         return data
#     except FileNotFoundError:
#         st.error(f"Error: Evaluation data file '{file_path}' not found. Please run 'train_model.py' first.")
#         return None
#     except Exception as e:
#         st.error(f"An error occurred while loading evaluation data: {e}")
#         return None

# # --- Plotting Functions ---

# def plot_confusion_matrix(cm, title):
#     fig = go.Figure(data=go.Heatmap(
#         z=cm,
#         x=['Predicted 0 (Safe)', 'Predicted 1 (Unsafe)'],
#         y=['Actual 0 (Safe)', 'Actual 1 (Unsafe)'],
#         hoverongaps = False,
#         colorscale='Viridis'
#     ))
#     fig.update_layout(title=title, xaxis_title="Predicted Class", yaxis_title="Actual Class")
#     return fig

# def plot_roc_curve(fpr, tpr, roc_auc, title):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
#     fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guessing'))
#     fig.update_layout(title=title, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
#     return fig

# def plot_feature_importance(importances, feature_names, title):
#     df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
#     df_importance = df_importance.sort_values('Importance', ascending=False)
#     fig = px.bar(df_importance, x='Importance', y='Feature', orientation='h',
#                  title=title,
#                  labels={'Importance': 'Feature Importance', 'Feature': ''})
#     fig.update_layout(yaxis={'categoryorder':'total ascending'})
#     return fig

# # --- Main Dashboard Layout ---
# st.title("⚡ Ola S1 Battery Safety Monitoring Dashboard")
# st.markdown("""
# This dashboard provides insights into battery performance, compares machine learning models, and helps predict potential burn risks for Ola S1 electric scooters.
# """)

# # --- Sidebar Navigation ---
# st.sidebar.title("Navigation")
# page_selection = st.sidebar.radio(
#     "Go to",
#     ("Historical Data", "ML Model Insights", "Predict New Values")
# )

# # --- Load historical data once ---
# csv_file_path = "battery_data.csv"
# historical_df = load_csv_data(csv_file_path)

# # --- Page Content based on Navigation ---

# if page_selection == "Historical Data":
#     st.header("📊 Historical Battery Performance Data")
#     st.markdown("Explore trends and distributions within your uploaded historical battery data.")

#     if not historical_df.empty:
#         st.write("---")
#         st.subheader("Raw Data Sample")
#         st.dataframe(historical_df.head(10))

#         st.subheader("Burn Risk Distribution in Historical Data")
#         if 'burn_risk' in historical_df.columns:
#             burn_risk_counts = historical_df['burn_risk'].value_counts().rename(index={0: 'Safe', 1: 'Unsafe'})
#             fig_pie = px.pie(
#                 values=burn_risk_counts.values,
#                 names=burn_risk_counts.index,
#                 title='Distribution of Battery Safety Status',
#                 color_discrete_sequence=['#4CAF50', '#F44336']
#             )
#             st.plotly_chart(fig_pie, use_container_width=True)
#         else:
#             st.info("No 'burn_risk' column available in historical data for distribution plot.")

#         st.subheader("Feature Distribution by Battery Safety Status")
#         available_features = [col for col in ['temperature', 'voltage', 'current', 'charge_cycles'] if col in historical_df.columns]

#         if available_features and 'burn_risk' in historical_df.columns:
#             selected_feature = st.selectbox("Select a feature to visualize against Burn Risk:", available_features)
#             fig_hist = px.histogram(
#                 historical_df,
#                 x=selected_feature,
#                 color='burn_risk',
#                 title=f'{selected_feature.replace("_", " ").title()} Distribution by Burn Risk',
#                 marginal="box",
#                 color_discrete_map={0: 'green', 1: 'red'}
#             )
#             st.plotly_chart(fig_hist, use_container_width=True)
#         else:
#             st.info("Cannot generate feature distribution plots. Ensure 'burn_risk' and feature columns are present.")

#     else:
#         st.warning("Historical data could not be loaded or is empty. Please check `team-14(in).csv`.")
#     st.write("---")

# elif page_selection == "ML Model Insights":
#     st.header("🔬 Machine Learning Model Performance Insights")
#     st.markdown("Here, you can compare the performance of the trained Random Forest and Linear Regression models on the test data.")

#     evaluation_data = load_evaluation_data()

#     if evaluation_data:
#         st.write("---")
#         st.subheader("Random Forest Model Performance")
        
#         col_rf1, col_rf2 = st.columns(2)
#         with col_rf1:
#             st.markdown("##### Confusion Matrix")
#             try:
#                 cm_rf = np.array(evaluation_data['rf_confusion_matrix'])
#                 st.plotly_chart(plot_confusion_matrix(cm_rf, "Random Forest Confusion Matrix"), use_container_width=True)
#             except KeyError:
#                 st.error("Random Forest Confusion Matrix data not found.")
#         with col_rf2:
#             st.markdown("##### Classification Report")
#             try:
#                 report_rf = evaluation_data['rf_classification_report']
#                 st.text(pd.DataFrame(report_rf).transpose().to_string())
#             except KeyError:
#                 st.error("Random Forest Classification Report data not found.")
            
#         st.markdown("##### ROC Curve")
#         try:
#             roc_rf = evaluation_data['rf_roc_curve']
#             st.plotly_chart(plot_roc_curve(roc_rf['fpr'], roc_rf['tpr'], roc_rf['auc'], "Random Forest ROC Curve"), use_container_width=True)
#         except KeyError:
#             st.error("Random Forest ROC Curve data not found.")

#         st.markdown("##### Feature Importance (Random Forest)")
#         try:
#             feature_importances_rf = evaluation_data['rf_feature_importances']
#             features = list(feature_importances_rf.keys())
#             importances = list(feature_importances_rf.values())
#             st.plotly_chart(plot_feature_importance(importances, features, "Random Forest Feature Importance"), use_container_width=True)
#         except KeyError:
#             st.error("Random Forest Feature Importance data not found.")

#         st.write("---")
#         st.subheader("Linear Regression Model Performance (as Classifier)")
        
#         col_lr1, col_lr2 = st.columns(2)
#         with col_lr1:
#             st.markdown("##### Confusion Matrix")
#             try:
#                 cm_lr = np.array(evaluation_data['lr_confusion_matrix'])
#                 st.plotly_chart(plot_confusion_matrix(cm_lr, "Linear Regression Confusion Matrix"), use_container_width=True)
#             except KeyError:
#                 st.error("Linear Regression Confusion Matrix data not found.")
#         with col_lr2:
#             st.markdown("##### Classification Report")
#             try:
#                 report_lr = evaluation_data['lr_classification_report']
#                 st.text(pd.DataFrame(report_lr).transpose().to_string())
#             except KeyError:
#                 st.error("Linear Regression Classification Report data not found.")
            
#         st.markdown("##### ROC Curve")
#         try:
#             roc_lr = evaluation_data['lr_roc_curve']
#             st.plotly_chart(plot_roc_curve(roc_lr['fpr'], roc_lr['tpr'], roc_lr['auc'], "Linear Regression ROC Curve"), use_container_width=True)
#         except KeyError:
#             st.error("Linear Regression ROC Curve data not found.")

#         st.markdown("##### Actual vs. Predicted Performance Rate (Linear Regression)")
#         try:
#             lr_actual_vs_predicted = evaluation_data['lr_actual_vs_predicted']
#             df_lr_pred = pd.DataFrame({
#                 'Actual Performance Rate': lr_actual_vs_predicted['actual'],
#                 'Predicted Performance Rate': lr_actual_vs_predicted['predicted']
#             })
#             fig_lr_scatter = px.scatter(df_lr_pred, x='Predicted Performance Rate', y='Actual Performance Rate',
#                                          title='Linear Regression: Actual vs. Predicted Performance Rate',
#                                          labels={'Predicted Performance Rate': 'Predicted Performance Rate', 'Actual Performance Rate': 'Actual Performance Rate'},
#                                          color_discrete_sequence=['#FF4B4B'])
#             fig_lr_scatter.add_trace(go.Scatter(x=[df_lr_pred['Actual Performance Rate'].min(), df_lr_pred['Actual Performance Rate'].max()],
#                                                 y=[df_lr_pred['Actual Performance Rate'].min(), df_lr_pred['Actual Performance Rate'].max()],
#                                                 mode='lines', name='Ideal Fit', line=dict(dash='dash', color='gray')))
#             st.plotly_chart(fig_lr_scatter, use_container_width=True)

#         except KeyError as e:
#             st.error(f"Linear Regression Actual vs. Predicted data not found or incorrect key: {e}. Please run 'train_model.py' again.")

#     else:
#         st.warning("Model evaluation data could not be loaded. Please ensure `train_model.py` was run successfully.")
#     st.write("---")

# elif page_selection == "Predict New Values":
#     st.header("🔍 Predict Battery Burn Risk for Custom Values")
#     st.markdown("Enter the battery parameters below to get instant safety predictions from both models.")

#     col1, col2 = st.columns(2)
#     with col1:
#         temperature_input = st.number_input("Battery Temperature (°C)", min_value=10.0, max_value=150.0, value=35.0, step=0.1, help="Typical operating range is 30-45°C. Higher temperatures can indicate issues.")
#         voltage_input = st.number_input("Voltage (V)", min_value=30.0, max_value=60.0, value=48.0, step=0.1, help="Nominal voltage is around 48V. Significant deviations can be problematic.")
#     with col2:
#         current_input = st.number_input("Current (A)", min_value=0.0, max_value=20.0, value=2.0, step=0.1, help="Current draw during operation. Unusually high current can indicate a short or overload.")
#         charge_cycles_input = st.number_input("Charge Cycles", min_value=0, max_value=2500, value=500, step=1, help="Number of times the battery has been fully charged and discharged. Higher cycles indicate battery wear.")

#     if (temperature_input is not None and voltage_input is not None and
#         current_input is not None and charge_cycles_input is not None):
        
#         data_to_send = {
#             "temperature": temperature_input,
#             "voltage": voltage_input,
#             "current": current_input,
#             "charge_cycles": charge_cycles_input
#         }

#         st.write("---")
#         st.subheader("Prediction Results:")
#         with st.spinner('Checking battery safety with both models...'):
#             try:
#                 flask_api_url = "http://localhost:5000/predict"
#                 response = requests.post(flask_api_url, json=data_to_send)
#                 response.raise_for_status()
                
#                 result = response.json()

#                 st.markdown("#### Random Forest Prediction")
#                 if result.get('rf_burn_risk_binary') == 1:
#                     st.error(f"🔥 **Warning: High Risk of Battery Burn!** (Random Forest)")
#                 elif result.get('rf_burn_risk_binary') == 0:
#                     st.success(f"✅ **Battery is Safe.** (Random Forest)")
#                 st.write(f"**Status:** `{result.get('rf_status', 'N/A')}`")
#                 st.write(f"**Predicted Burn Risk (0=Safe, 1=Unsafe):** `{result.get('rf_burn_risk_binary', 'N/A')}`")
#                 st.write(f"**Probability of Unsafe:** `{result.get('rf_burn_risk_proba', 'N/A'):.2f}`")

#                 st.markdown("#### Linear Regression Prediction")
#                 if result.get('lr_burn_risk_binary') == 1:
#                     st.error(f"🔥 **Warning: High Risk of Battery Burn!** (Linear Regression)")
#                 elif result.get('lr_burn_risk_binary') == 0:
#                     st.success(f"✅ **Battery is Safe.** (Linear Regression)")
#                 st.write(f"**Status:** `{result.get('lr_status', 'N/A')}`")
#                 st.write(f"**Predicted Burn Risk (0=Safe, 1=Unsafe):** `{result.get('lr_burn_risk_binary', 'N/A')}`")
#                 st.write(f"**Raw Continuous Score:** `{result.get('lr_burn_risk_raw', 'N/A'):.2f}`")

#             except requests.exceptions.ConnectionError:
#                 st.error("❌ **Connection Error:** Could not connect to the Flask backend.")
#                 st.markdown("Please ensure the Flask app (`app.py`) is running on `http://localhost:5000`.")
#             except requests.exceptions.RequestException as e:
#                 st.error(f"❌ **API Request Error:** {e}")
#                 st.markdown("An error occurred while communicating with the prediction service. Check the Flask console for details.")
#             except Exception as e:
#                 st.error(f"❌ **An unexpected error occurred:** {e}")
#     else:
#         st.info("Please enter values in all fields to get a prediction.")

# st.write("---")
# st.info("💡 **Note:** This model is for demonstration purposes. A real-world safety system would require extensive, real-world data and rigorous testing.")
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np
import json
import os

st.set_page_config(layout="wide", page_title="Ola S1 Battery Safety Monitor", page_icon="⚡")

# --- Function to load CSV data ---
@st.cache_data
def load_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()
        
        # Add a dummy 'burn_risk' column for visualization if it's not present.
        if 'burn_risk' not in df.columns:
            st.warning("No 'burn_risk' column found in CSV. Generating dummy risk for visualization.")
            df['burn_risk'] = ((df['temperature'] > df['temperature'].mean() + 1.5 * df['temperature'].std()) |
                               (df['voltage'] < df['voltage'].mean() - 1.5 * df['voltage'].std()) |
                               (df['current'] > df['current'].mean() + 1.5 * df['current'].std()) |
                               (df['charge_cycles'] > df['charge_cycles'].mean() + 1.5 * df['charge_cycles'].std())).astype(int)
        
        required_cols = ['temperature', 'voltage', 'current', 'charge_cycles']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column for model prediction and visualization: '{col}'. Please check your CSV.")
                return pd.DataFrame()
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure 'team-14(in).csv' is in the same directory as this script.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading or processing the CSV: {e}")
        return pd.DataFrame()

# --- Function to load evaluation data ---
@st.cache_data
def load_evaluation_data(file_path="model_evaluation_data.json"):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"Error: Evaluation data file '{file_path}' not found. Please run 'train_model.py' first.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading evaluation data: {e}")
        return None

# --- Plotting Functions ---

def plot_confusion_matrix(cm, title):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted 0 (Safe)', 'Predicted 1 (Unsafe)'],
        y=['Actual 0 (Safe)', 'Actual 1 (Unsafe)'],
        hoverongaps = False,
        colorscale='Viridis'
    ))
    fig.update_layout(title=title, xaxis_title="Predicted Class", yaxis_title="Actual Class")
    return fig

def plot_roc_curve(fpr, tpr, roc_auc, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guessing'))
    fig.update_layout(title=title, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    return fig

def plot_feature_importance(importances, feature_names, title):
    df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_importance = df_importance.sort_values('Importance', ascending=False)
    fig = px.bar(df_importance, x='Importance', y='Feature', orientation='h',
                 title=title,
                 labels={'Importance': 'Feature Importance', 'Feature': ''})
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

# --- Main Dashboard Layout ---
st.title("⚡ Ola S1 Battery Safety Monitoring Dashboard")
st.markdown("""
This dashboard provides insights into battery performance, compares machine learning models, and helps predict potential burn risks for Ola S1 electric scooters.
""")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    ("Historical Data", "ML Model Insights", "Predict New Values")
)

# --- Load historical data once ---
csv_file_path = "battery_data.csv"
historical_df = load_csv_data(csv_file_path)

# --- Page Content based on Navigation ---

if page_selection == "Historical Data":
    st.header("📊 Historical Battery Performance Data")
    st.markdown("Explore trends and distributions within your uploaded historical battery data.")

    if not historical_df.empty:
        st.write("---")
        st.subheader("Raw Data Sample")
        st.dataframe(historical_df.head(10))

        st.subheader("Burn Risk Distribution in Historical Data")
        if 'burn_risk' in historical_df.columns:
            burn_risk_counts = historical_df['burn_risk'].value_counts().rename(index={0: 'Safe', 1: 'Unsafe'})
            fig_pie = px.pie(
                values=burn_risk_counts.values,
                names=burn_risk_counts.index,
                title='Distribution of Battery Safety Status',
                color_discrete_sequence=['#4CAF50', '#F44336']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No 'burn_risk' column available in historical data for distribution plot.")

        st.subheader("Feature Distribution by Battery Safety Status")
        available_features = [col for col in ['temperature', 'voltage', 'current', 'charge_cycles'] if col in historical_df.columns]

        if available_features and 'burn_risk' in historical_df.columns:
            selected_feature = st.selectbox("Select a feature to visualize against Burn Risk:", available_features)
            fig_hist = px.histogram(
                historical_df,
                x=selected_feature,
                color='burn_risk',
                title=f'{selected_feature.replace("_", " ").title()} Distribution by Burn Risk',
                marginal="box",
                color_discrete_map={0: 'green', 1: 'red'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Cannot generate feature distribution plots. Ensure 'burn_risk' and feature columns are present.")

    else:
        st.warning("Historical data could not be loaded or is empty. Please check `team-14(in).csv`.")
    st.write("---")

elif page_selection == "ML Model Insights":
    st.header("🔬 Machine Learning Model Performance Insights")
    st.markdown("Here, you can compare the performance of the trained Random Forest and Linear Regression models on the test data.")

    evaluation_data = load_evaluation_data()

    if evaluation_data:
        st.write("---")
        st.subheader("Random Forest Model Performance")
        
        col_rf1, col_rf2 = st.columns(2)
        with col_rf1:
            st.markdown("##### Confusion Matrix")
            try:
                cm_rf = np.array(evaluation_data['rf_confusion_matrix'])
                st.plotly_chart(plot_confusion_matrix(cm_rf, "Random Forest Confusion Matrix"), use_container_width=True)
            except KeyError:
                st.error("Random Forest Confusion Matrix data not found.")
        with col_rf2:
            st.markdown("##### Classification Report")
            try:
                report_rf = evaluation_data['rf_classification_report']
                st.text(pd.DataFrame(report_rf).transpose().to_string())
            except KeyError:
                st.error("Random Forest Classification Report data not found.")
            
        st.markdown("##### ROC Curve")
        try:
            roc_rf = evaluation_data['rf_roc_curve']
            st.plotly_chart(plot_roc_curve(roc_rf['fpr'], roc_rf['tpr'], roc_rf['auc'], "Random Forest ROC Curve"), use_container_width=True)
        except KeyError:
            st.error("Random Forest ROC Curve data not found.")

        st.markdown("##### Feature Importance (Random Forest)")
        try:
            feature_importances_rf = evaluation_data['rf_feature_importances']
            features = list(feature_importances_rf.keys())
            importances = list(feature_importances_rf.values())
            st.plotly_chart(plot_feature_importance(importances, features, "Random Forest Feature Importance"), use_container_width=True)
        except KeyError:
            st.error("Random Forest Feature Importance data not found.")

        st.write("---")
        st.subheader("Linear Regression Model Performance (as Classifier)")
        
        col_lr1, col_lr2 = st.columns(2)
        with col_lr1:
            st.markdown("##### Confusion Matrix")
            try:
                cm_lr = np.array(evaluation_data['lr_confusion_matrix'])
                st.plotly_chart(plot_confusion_matrix(cm_lr, "Linear Regression Confusion Matrix"), use_container_width=True)
            except KeyError:
                st.error("Linear Regression Confusion Matrix data not found.")
        with col_lr2:
            st.markdown("##### Classification Report")
            try:
                report_lr = evaluation_data['lr_classification_report']
                st.text(pd.DataFrame(report_lr).transpose().to_string())
            except KeyError:
                st.error("Linear Regression Classification Report data not found.")
            
        st.markdown("##### ROC Curve")
        try:
            roc_lr = evaluation_data['lr_roc_curve']
            st.plotly_chart(plot_roc_curve(roc_lr['fpr'], roc_lr['tpr'], roc_lr['auc'], "Linear Regression ROC Curve"), use_container_width=True)
        except KeyError:
            st.error("Linear Regression ROC Curve data not found.")

        st.markdown("##### Actual vs. Predicted Performance Rate (Linear Regression)")
        try:
            lr_actual_vs_predicted = evaluation_data['lr_actual_vs_predicted']
            # CORRECTED LINE: Using 'predicted' key, not 'predicted_raw'
            df_lr_pred = pd.DataFrame({
                'Actual Performance Rate': lr_actual_vs_predicted['actual'],
                'Predicted Performance Rate': lr_actual_vs_predicted['predicted']
            })
            fig_lr_scatter = px.scatter(df_lr_pred, x='Predicted Performance Rate', y='Actual Performance Rate',
                                         title='Linear Regression: Actual vs. Predicted Performance Rate',
                                         labels={'Predicted Performance Rate': 'Predicted Performance Rate', 'Actual Performance Rate': 'Actual Performance Rate'},
                                         color_discrete_sequence=['#FF4B4B'])
            fig_lr_scatter.add_trace(go.Scatter(x=[df_lr_pred['Actual Performance Rate'].min(), df_lr_pred['Actual Performance Rate'].max()],
                                                y=[df_lr_pred['Actual Performance Rate'].min(), df_lr_pred['Actual Performance Rate'].max()],
                                                mode='lines', name='Ideal Fit', line=dict(dash='dash', color='gray')))
            st.plotly_chart(fig_lr_scatter, use_container_width=True)

        except KeyError as e:
            st.error(f"Linear Regression Actual vs. Predicted data not found or incorrect key: {e}. Please run 'train_model.py' again.")

    else:
        st.warning("Model evaluation data could not be loaded. Please ensure `train_model.py` was run successfully.")
    st.write("---")

elif page_selection == "Predict New Values":
    st.header("🔍 Predict Battery Burn Risk for Custom Values")
    st.markdown("Enter the battery parameters below to get instant safety predictions from both models.")

    col1, col2 = st.columns(2)
    with col1:
        temperature_input = st.number_input("Battery Temperature (°C)", min_value=10.0, max_value=150.0, value=35.0, step=0.1, help="Typical operating range is 30-45°C. Higher temperatures can indicate issues.")
        voltage_input = st.number_input("Voltage (V)", min_value=30.0, max_value=60.0, value=48.0, step=0.1, help="Nominal voltage is around 48V. Significant deviations can be problematic.")
    with col2:
        current_input = st.number_input("Current (A)", min_value=0.0, max_value=20.0, value=2.0, step=0.1, help="Current draw during operation. Unusually high current can indicate a short or overload.")
        charge_cycles_input = st.number_input("Charge Cycles", min_value=0, max_value=2500, value=500, step=1, help="Number of times the battery has been fully charged and discharged. Higher cycles indicate battery wear.")

    if (temperature_input is not None and voltage_input is not None and
        current_input is not None and charge_cycles_input is not None):
        
        data_to_send = {
            "temperature": temperature_input,
            "voltage": voltage_input,
            "current": current_input,
            "charge_cycles": charge_cycles_input
        }

        st.write("---")
        st.subheader("Prediction Results:")
        with st.spinner('Checking battery safety with both models...'):
            try:
                # flask_api_url = "http://localhost:5000/predict"
                flask_api_url ="https://mini-project-main-1.onrender.com/predict"
                response = requests.post(flask_api_url, json=data_to_send)
                response.raise_for_status()
                
                result = response.json()

                st.markdown("#### Random Forest Prediction")
                if result.get('rf_burn_risk_binary') == 1:
                    st.error(f"🔥 **Warning: High Risk of Battery Burn!** (Random Forest)")
                elif result.get('rf_burn_risk_binary') == 0:
                    st.success(f"✅ **Battery is Safe.** (Random Forest)")
                st.write(f"**Status:** `{result.get('rf_status', 'N/A')}`")
                st.write(f"**Predicted Burn Risk (0=Safe, 1=Unsafe):** `{result.get('rf_burn_risk_binary', 'N/A')}`")
                st.write(f"**Probability of Unsafe:** `{result.get('rf_burn_risk_proba', 'N/A'):.2f}`")

                st.markdown("#### Linear Regression Prediction")
                if result.get('lr_burn_risk_binary') == 1:
                    st.error(f"🔥 **Warning: High Risk of Battery Burn!** (Linear Regression)")
                elif result.get('lr_burn_risk_binary') == 0:
                    st.success(f"✅ **Battery is Safe.** (Linear Regression)")
                st.write(f"**Status:** `{result.get('lr_status', 'N/A')}`")
                st.write(f"**Predicted Burn Risk (0=Safe, 1=Unsafe):** `{result.get('lr_burn_risk_binary', 'N/A')}`")
                st.write(f"**Raw Continuous Score:** `{result.get('lr_burn_risk_raw', 'N/A'):.2f}`")

            except requests.exceptions.ConnectionError:
                st.error("❌ **Connection Error:** Could not connect to the Flask backend.")
                st.markdown("Please ensure the Flask app (`app.py`) is running on `http://localhost:5000`.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ **API Request Error:** {e}")
                st.markdown("An error occurred while communicating with the prediction service. Check the Flask console for details.")
            except Exception as e:
                st.error(f"❌ **An unexpected error occurred:** {e}")
    else:
        st.info("Please enter values in all fields to get a prediction.")

st.write("---")
st.info("💡 **Note:** This model is for demonstration purposes. A real-world safety system would require extensive, real-world data and rigorous testing.")

