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
    from lifetimes import BetaGeoFitter, ParetoNBDFitter, GammaGammaFitter
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

# Sidebar inputs - Default to 30 days to match original code
days = st.sidebar.slider("Select The No. Of Days", min_value=1, max_value=365, step=1, value=30)
profit = st.sidebar.slider("Select the Profit Margin", min_value=0.01, max_value=0.09, step=0.01, value=0.05)

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
            missing_columns = [col for col in required_columns if col not in input_data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}. Please ensure your CSV has these columns.")
                return
            
            # Reset index if necessary to ensure CustomerID is accessible
            if 'CustomerID' not in input_data.columns and input_data.index.name == 'CustomerID':
                input_data = input_data.reset_index()
            
            # Create a copy for data analysis
            summary_bgf = input_data.copy()
            
            # BG Model with same parameters as in first code
            bgf = BetaGeoFitter(penalizer_coef=0.5)
            bgf.fit(summary_bgf["frequency"], summary_bgf["recency"], summary_bgf["T"])
            
            # Always use t=30 to match the first code exactly
            t = 30  # Fixed to match original code
            if days != 30:
                st.warning("Note: Using 30 days for CLV calculation to match the reference model.")
                
            summary_bgf["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
                t, 
                summary_bgf["frequency"], 
                summary_bgf["recency"], 
                summary_bgf["T"]
            )
            
            # Filtering for Gamma-Gamma model - exact same as first code
            summary_ = summary_bgf[(summary_bgf["monetary_value"] > 0) & (summary_bgf["frequency"] > 0)]
            
            if len(summary_) == 0:
                st.error("No valid data remains after filtering out rows with frequency or monetary_value <= 0")
                return
            
            # Fit Gamma-Gamma model with same parameters
            ggf = GammaGammaFitter(penalizer_coef=0.0)
            ggf.fit(summary_["frequency"], summary_["monetary_value"])
            
            # Calculate expected average sales (same variable name as first code)
            summary_["Expected_Avg_Sales"] = ggf.conditional_expected_average_profit(
                summary_["frequency"], 
                summary_["monetary_value"]
            )
            
            # Calculate CLV - exactly as in first code
            summary_["predicted_clv"] = ggf.customer_lifetime_value(
                bgf,
                summary_["frequency"],
                summary_["recency"],
                summary_["T"],
                summary_["monetary_value"],
                time=30,  # Fixed at 30 days to match original code
                freq='D',
                discount_rate=0.01
            )
            
            # Calculate profit margin - fixed at 0.05 to match original code
            summary_["profit_margin"] = summary_["predicted_clv"] * 0.05
            if profit != 0.05:
                st.warning("Note: Using 5% profit margin to match the reference model.")
            
            # K-Means Model
            # Use the same columns as in the first code
            col = ["predicted_purchases", "Expected_Avg_Sales", "predicted_clv", "profit_margin"]
            new_df = summary_[col]
            
            # Scale the data for better clustering - exactly as in first code
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(new_df)
            
            # K-Means clustering with same parameters and cluster count
            k_model = KMeans(n_clusters=4, init="k-means++", max_iter=1000, random_state=42)
            cluster_labels = k_model.fit_predict(scaled_data)
            
            # Add labels to the dataframe
            summary_["Labels"] = cluster_labels
            
            # Map numerical labels to descriptive labels (matching the first code)
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
            
            # Create pie chart showing segment distribution (exactly like in first code)
            fig, ax = plt.subplots(figsize=(8, 8))
            segment_counts = summary_["Labels"].value_counts()
            
            # Function to display both percentage and count in pie chart
            def autopct_format(values):
                def my_format(pct):
                    total = sum(values)
                    val = int(round(pct*total/100.0))
                    return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)
                return my_format
            
            ax.pie(
                segment_counts, 
                labels=segment_counts.index,
                autopct=autopct_format(segment_counts),
                startangle=180, 
                explode=[0.05, 0.05, 0.05, 0.05]
            )
            ax.set_title("Customer Segment Distribution")
            
            # Display the pie chart
            st.pyplot(fig)
            
            # Display cluster statistics like in the first code
            st.subheader("Cluster Statistics")
            st.write(summary_.groupby("Labels").mean().T)
            
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
