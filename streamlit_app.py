import streamlit as st
import pandas as pd
import math
import numpy as np
import datetime
import altair as alt
import time
import warnings
warnings.filterwarnings("ignore")
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Import lifetimes package with proper error handling
try:
    import lifetimes
    from lifetimes.plotting import *
    from lifetimes import BetaGeoFitter, ParetoNBDFitter
    from lifetimes.utils import calibration_and_holdout_data
except ImportError:
    st.error("The lifetimes package is not installed. Please install it with 'pip install lifetimes'.")
    st.stop()

# Seed for reproducibility
np.random.seed(42)

# App title and description
st.title("CLV Prediction and Segmentation App")
st.markdown("Upload the RFM data to get the customer lifetime value and their segmentation")

# Display header image
st.image("https://ultracommerce.co/wp-content/uploads/2022/04/maximize-customer-lifetime-value.png", use_container_width=True)

# File uploader
data = st.file_uploader("File Uploader", type=['csv'])

# Sidebar
st.sidebar.image("https://www.adlibweb.com/wp-content/uploads/2020/06/customer-lifetime-value.jpg", width=150)
st.sidebar.markdown("**MBA Project**")
st.sidebar.title("Input Features :pencil:")

# Sidebar inputs
days = st.sidebar.slider("Select The No. Of Days", min_value=1, max_value=365, step=1)
profit = st.sidebar.slider("Select the Profit Margin", min_value=0.01, max_value=0.09, step=0.01)

# Sidebar instructions
st.sidebar.markdown("""
Before uploading the file, please select the input features first.

Also, please make sure the columns are in proper format. For reference you can download the [dummy data](https://github.com/tejas-tilekar/CDAC-Project/blob/main/Deployment%20files/sample_file.csv).

**Note:** Only Use "CSV" File.
""")

# Main function to process data
if data is not None:
    def load_data(data, days, profit):
        try:
            # Load data
            input_data = pd.read_csv(data)
            
            # Handle potential first column as index issue
            if input_data.columns[0] == 'Unnamed: 0':
                input_data = input_data.iloc[:, 1:]
            
            # Check if required columns exist
            required_columns = ['frequency', 'recency', 'T', 'monetary_value']
            if 'CustomerID' not in input_data.columns:
                required_columns.append('CustomerID')
                
            missing_columns = [col for col in required_columns if col not in input_data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}. Please ensure your CSV has these columns.")
                return
            
            # BG-NBD Model - exactly as in first code
            bgf = BetaGeoFitter(penalizer_coef=0.5)
            bgf.fit(input_data['frequency'], input_data['recency'], input_data['T'])
            
            # Predicting purchases for specified time period
            t = days
            input_data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
                t, input_data['frequency'], input_data['recency'], input_data['T']
            )
            
            # Filter data for Gamma-Gamma model - exactly as in first code
            summary_ = input_data[(input_data["monetary_value"] > 0) & (input_data["frequency"] > 0)].copy()
            
            if len(summary_) == 0:
                st.error("No valid data remains after filtering out rows with frequency or monetary_value <= 0")
                return
                
            # Initialize and fit Gamma-Gamma fitter - exactly as in first code
            ggf = lifetimes.GammaGammaFitter(penalizer_coef=0.0)
            ggf.fit(summary_["frequency"], summary_["monetary_value"])
            
            # Calculate expected average sales - exactly as in first code
            summary_["Expected_Avg_Sales"] = ggf.conditional_expected_average_profit(
                summary_["frequency"], summary_["monetary_value"]
            )
            
            # Calculate predicted CLV - exactly as in first code
            summary_["predicted_clv"] = ggf.customer_lifetime_value(
                bgf,
                summary_["frequency"],
                summary_["recency"],
                summary_["T"],
                summary_["monetary_value"],
                time=days,
                freq='D',
                discount_rate=0.01
            )
            
            # Calculate profit margin
            summary_["profit_margin"] = summary_["predicted_clv"] * profit
            
            # K-Means Model
            col = ["predicted_purchases", "Expected_Avg_Sales", "predicted_clv", "profit_margin"]
            new_df = summary_[col]
            
            # Scale the data for better clustering
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(new_df)
            
            # K-Means clustering with 4 clusters as in first code
            k_model = KMeans(n_clusters=4, init="k-means++", max_iter=1000, random_state=42)
            cluster_labels = k_model.fit_predict(scaled_data)
            
            # Add labels to the dataframe
            summary_["Labels"] = cluster_labels
            
            # Map numerical labels to descriptive labels - using the same mapping as first code
            label_mapper = {0: "Low", 1: "High", 2: "Medium", 3: "V_High"}
            summary_["Labels"] = summary_["Labels"].map(label_mapper)
            
            # Display the results dataframe
            st.write(summary_)
            
            # Create count bar chart
            chart = alt.Chart(summary_).mark_bar().encode(
                y=alt.Y('Labels:N', title='Customer Segment'),
                x=alt.X('count(Labels):Q', title='Number of Customers')
            ).properties(
                title='Customer Segmentation Distribution'
            )
            
            # Add text labels to the chart
            text = chart.mark_text(
                align='left',
                baseline='middle',
                dx=3
            ).encode(
                text='count(Labels):Q'
            )
            
            # Display the chart
            st.altair_chart(chart + text, use_container_width=True)
            
            # Add download button
            csv = summary_.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="clv_prediction_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    # Call the function with the uploaded data
    st.markdown("## Customer Lifetime Prediction Result :bar_chart:")
    load_data(data, days, profit)
    
else:
    st.info("Please Upload the CSV File")
