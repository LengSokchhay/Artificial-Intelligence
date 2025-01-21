import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained models and scaler
kmeans = joblib.load('kmeans_model.pkl')
dbscan = joblib.load('dbscan_model.pkl')
gmm = joblib.load('gmm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the features used in scaling
features_for_scaling = ['TotalSpend', 'Quantity', 'Recency']

# Function to perform the clustering and prediction
def predict_segment(input_data, model_type):
    # Scale the input data
    input_scaled = scaler.transform(input_data[features_for_scaling])
    
    # Predict the cluster based on the selected model
    if model_type == 'KMeans':
        return kmeans.predict(input_scaled)
    
    elif model_type == 'DBSCAN':
        return dbscan.fit_predict(input_scaled)
    
    elif model_type == 'GMM':
        return gmm.predict(input_scaled)
    else:
        return "Invalid model"

# Streamlit UI
st.title('Customer Segmentation App')

# Input fields for the user
st.header("Input Customer Data")
total_spend = st.number_input("Total Spend", step=1.0)
quantity = st.number_input("Quantity", step=1)
recency = st.number_input("Recency (Days since last purchase)", min_value=1, step=1)

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'TotalSpend': [total_spend],
    'Quantity': [quantity],
    'Recency': [recency]
})

# Dropdown for model selection
model_type = st.selectbox("Select Clustering Model", options=['KMeans', 'DBSCAN', 'GMM'])

# When the user clicks "Predict", run the model
if st.button('Predict Segment'):
    # Predict the segment for the input data
    segment = predict_segment(input_data, model_type)
    st.write(f"The predicted segment for this customer is: {segment[0]}")

    # Cache the data processing to avoid recalculating each time
    @st.cache_data
    def load_and_process_data():
        # Load only necessary columns and clean the data
        data = pd.read_excel("cleaned_data.xlsx", usecols=features_for_scaling)
        data = data.dropna(subset=features_for_scaling)

        # Remove outliers based on IQR
        Q1 = data[features_for_scaling].quantile(0.25)
        Q3 = data[features_for_scaling].quantile(0.75)
        IQR = Q3 - Q1
        data_cleaned = data[~((data[features_for_scaling] < (Q1 - 1.5 * IQR)) | (data[features_for_scaling] > (Q3 + 1.5 * IQR))).any(axis=1)]
        return data_cleaned

    # Load and preprocess the data
    data_cleaned = load_and_process_data()

    # Scale the data sample
    X_scaled_sample = scaler.transform(data_cleaned[features_for_scaling])

    # Get the predicted segments for the cleaned data using the selected model
    if model_type == 'KMeans':
        segments_sample = kmeans.predict(X_scaled_sample)
    elif model_type == 'DBSCAN':
        segments_sample = dbscan.fit_predict(X_scaled_sample)
    elif model_type == 'GMM':
        segments_sample = gmm.predict(X_scaled_sample)
    else:
        segments_sample = "Invalid model"
    
    # Add the predicted segments to the cleaned data
    data_cleaned['Segment'] = segments_sample

    # Visualize the result for the cleaned data
    st.subheader("Clustering Visualization")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_cleaned['TotalSpend'], y=data_cleaned['Quantity'], hue=data_cleaned['Segment'], palette='viridis', s=100, marker='o')
    plt.title(f'Customer Segments using {model_type} (Cleaned Data)')
    plt.xlabel('Total Spend')
    plt.ylabel('Quantity')
    st.pyplot(plt)

# Add some styling to the app
st.markdown("""<style> .css-18e3tke { text-align: center; } </style>""", unsafe_allow_html=True)
