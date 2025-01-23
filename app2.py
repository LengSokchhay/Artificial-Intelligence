import streamlit as st
import joblib
import numpy as np

# Load pre-trained models
log_reg_model = joblib.load('logistic_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
svm_model = joblib.load('svm_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')
gb_model = joblib.load('gradient_boosting_model.pkl')
knn_model = joblib.load('knn_model.pkl') 

# Load the scaler used during training
scaler = joblib.load('scaler.pkl')

# Function to predict cluster
def predict_cluster(model, recency, frequency, monetary):
    input_data = np.array([[recency, frequency, monetary]])
    # Predict the cluster
    cluster = model.predict(input_data)
    return cluster[0]

# Streamlit UI with styling
st.set_page_config(page_title="Customer Segmentation Classifier", page_icon=":bar_chart:", layout="wide")

# Custom CSS styles
st.markdown("""
    <style>
    .main {background-color: #f4f4f4;}
    .sidebar .sidebar-content {background-color: #ddebf7;}
    h1, h2, h3, h4, h5 {color: #2a5d84;}
    .stButton button {background-color: #2a5d84; color: white; border-radius: 5px;}
    .stButton button:hover {background-color: #3b73a9;}
    .prediction {background-color: #e9f7ef; padding: 10px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# Title and Header
st.title('📊 Customer Segmentation Classifier')
st.markdown("#### Classify customer segments based on Recency, Frequency, and Monetary values.")

# Sidebar Input
st.sidebar.header('🔧 Input Parameters')
recency = st.sidebar.number_input('Recency (Days since last purchase)', value=0, step=1)
frequency = st.sidebar.number_input('Frequency (Total number of purchases)', value=0, step=1)
monetary = st.sidebar.number_input('Monetary (Total spent $)', value=0, step=1)

# Prediction Button
if st.sidebar.button('🔍 Classify'):
    st.subheader("📌 Classifications for the Given Customer")
    
    # Predict clusters
    log_reg_cluster = predict_cluster(log_reg_model, recency, frequency, monetary)
    rf_cluster = predict_cluster(rf_model, recency, frequency, monetary)
    svm_cluster = predict_cluster(svm_model, recency, frequency, monetary)
    dt_cluster = predict_cluster(dt_model, recency, frequency, monetary)
    gb_cluster = predict_cluster(gb_model, recency, frequency, monetary)
    knn_cluster = predict_cluster(knn_model, recency, frequency, monetary)  # Predict with KNN model

    # Cluster labels
    cluster_mapping = {
        0: "High Spender",
        1: "Lapsed Customer",
        2: "Frequent Buyer",
        3: "Occasional Buyer"
    }

    # Display model predictions
    st.markdown("""
    <div class='prediction'>
        <h3 style="color: black; margin-bottom: 10px;">Model Classifications:</h3>
        <ul style="list-style-type: none; padding: 0; margin: 0;">
            <li style="padding: 10px; background-color: #d8eafd; border-radius: 5px; margin-bottom: 5px; color: black;">
                <strong>Logistic Regression:</strong> {log_reg_cluster}
            </li>
            <li style="padding: 10px; background-color: #e9f7ef; border-radius: 5px; margin-bottom: 5px; color: black;">
                <strong>Random Forest:</strong> {rf_cluster}
            </li>
            <li style="padding: 10px; background-color: #d8eafd; border-radius: 5px; margin-bottom: 5px; color: black;">
                <strong>SVM:</strong> {svm_cluster}
            </li>
            <li style="padding: 10px; background-color: #e9f7ef; border-radius: 5px; margin-bottom: 5px; color: black;">
                <strong>Decision Tree:</strong> {dt_cluster}
            </li>
            <li style="padding: 10px; background-color: #d8eafd; border-radius: 5px; margin-bottom: 5px; color: black;">
                <strong>Gradient Boosting:</strong> {gb_cluster}
            </li>
            <li style="padding: 10px; background-color: #e9f7ef; border-radius: 5px; margin-bottom: 5px; color: black;">
                <strong>K-Nearest Neighbors:</strong> {knn_cluster}
            </li>  
        </ul>
    </div>
    """.format(
        log_reg_cluster=cluster_mapping[log_reg_cluster],
        rf_cluster=cluster_mapping[rf_cluster],
        svm_cluster=cluster_mapping[svm_cluster],
        dt_cluster=cluster_mapping[dt_cluster],
        gb_cluster=cluster_mapping[gb_cluster],
        knn_cluster=cluster_mapping[knn_cluster]
    ), unsafe_allow_html=True)


    # Final Prediction
    clusters = [log_reg_cluster, rf_cluster, svm_cluster, dt_cluster, gb_cluster, knn_cluster]
    final_cluster = max(set(clusters), key=clusters.count)
    st.markdown(f"### 🎯 Final Classification: **{cluster_mapping[final_cluster]}**")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <footer style='text-align: center; color: #6c757d; font-size: small;'>
    </footer>
    """,
    unsafe_allow_html=True
)
