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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
st.markdown("Upload the transaction data to get the customer lifetime value and their segmentation")

# Display header image
st.image("https://ultracommerce.co/wp-content/uploads/2022/04/maximize-customer-lifetime-value.png", use_container_width=True)

# Sidebar
st.sidebar.image("https://www.adlibweb.com/wp-content/uploads/2020/06/customer-lifetime-value.jpg", width=150)
st.sidebar.markdown("**MBA Project**")
st.sidebar.title("Input Features :pencil:")

# Sidebar inputs
days = st.sidebar.slider("Select The No. Of Days", min_value=1, max_value=365, step=1, value=30)  # Default to 30 to match original code
profit_margin = st.sidebar.slider("Select the Profit Margin", min_value=0.01, max_value=0.09, step=0.01, value=0.05)  # Default to 0.05 as in original code

# Add calibration date selector
cal_end_date = st.sidebar.date_input(
    "Calibration Period End Date", 
    value=datetime.date(2011, 6, 8),  # Default from original code
    help="The date ending the calibration period"
)

obs_end_date = st.sidebar.date_input(
    "Observation Period End Date", 
    value=datetime.date(2011, 12, 9),  # Default from original code
    help="The date ending the observation period"
)

# Sidebar instructions
st.sidebar.markdown("""
Before uploading the file, please select the input features first.

**Required File Format**:
- A CSV file with transaction data containing columns:
  - CustomerID
  - InvoiceDate
  - Quantity
  - UnitPrice
  - Amount (optional, will be calculated if not present)

For best results, use the same dataset as the original analysis.

**Note:** Only Use "CSV" File.
""")

# File uploader now accepts transaction data, not RFM data
uploaded_file = st.file_uploader("Upload Transaction Data CSV", type=['csv'])

# Main function to process transaction data
if uploaded_file is not None:
    def process_data(transaction_data, days, profit_margin, cal_end_date, obs_end_date):
        try:
            # Progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Load the transaction data
            status_text.text('Loading data...')
            df = pd.read_csv(transaction_data)
            progress_bar.progress(10)
            
            # Check if required columns exist
            required_columns = ['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return
            
            # Step 2: Data preprocessing as in original code
            status_text.text('Preprocessing data...')
            
            # Rename column if needed
            if 'Customer ID' in df.columns and 'CustomerID' not in df.columns:
                df.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)
            
            # Drop duplicates
            df = df.drop_duplicates()
            
            # Drop rows with null values in Description and CustomerID
            if 'Description' in df.columns:
                df.dropna(axis=0, subset=["Description"], inplace=True)
            df.dropna(axis=0, subset=["CustomerID"], inplace=True)
            
            # Keep only positive quantities
            df = df[(df.Quantity > 0)]
            
            # Convert dates
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date
            
            # Calculate Amount if not present
            if 'Amount' not in df.columns:
                df['Amount'] = df['Quantity'] * df['UnitPrice']
            
            progress_bar.progress(30)
            
            # Step 3: Generate RFM data
            status_text.text('Generating RFM data...')
            rfmt_data = lifetimes.utils.summary_data_from_transaction_data(
                df, 'CustomerID', 'InvoiceDate', monetary_value_col='Amount'
            )
            
            progress_bar.progress(40)
            
            # Step 4: Beta Geo Fitter model
            status_text.text('Building BG/NBD model...')
            bgf = BetaGeoFitter(penalizer_coef=0.5)
            bgf.fit(rfmt_data['frequency'], rfmt_data['recency'], rfmt_data['T'])
            
            # Step 5: Calibration and holdout data
            status_text.text('Splitting into calibration and holdout data...')
            summary_cal_holdout = calibration_and_holdout_data(
                df, 'CustomerID', 'InvoiceDate',
                calibration_period_end=cal_end_date.strftime('%Y-%m-%d'),
                observation_period_end=obs_end_date.strftime('%Y-%m-%d')
            )
            
            progress_bar.progress(50)
            
            # Step 6: Fit model on calibration data
            status_text.text('Fitting model on calibration data...')
            bgf.fit(
                summary_cal_holdout['frequency_cal'], 
                summary_cal_holdout['recency_cal'], 
                summary_cal_holdout['T_cal'],
                penalizer_coef=0.5
            )
            
            progress_bar.progress(60)
            
            # Step 7: Create summary for predictions
            status_text.text('Calculating predictions...')
            summary_bgf = rfmt_data.copy().reset_index()
            
            # Calculate predicted purchases for the specified time period
            summary_bgf['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
                days, 
                rfmt_data['frequency'], 
                rfmt_data['recency'], 
                rfmt_data['T']
            )
            
            # Calculate actual purchases using same method as original code
            summary_bgf["actual_purchases"] = summary_cal_holdout['frequency_holdout']/10
            summary_bgf = summary_bgf.fillna(value=0)  # Fill NA values with 0 like in original
            
            progress_bar.progress(70)
            
            # Step 8: Filter data where monetary_value and frequency > 0
            status_text.text('Building Gamma-Gamma model...')
            summary_ = summary_bgf[(summary_bgf["monetary_value"] > 0) & (summary_bgf["frequency"] > 0)]
            
            # Fit Gamma-Gamma model
            ggf = GammaGammaFitter(penalizer_coef=0.0)
            ggf.fit(summary_["frequency"], summary_["monetary_value"])
            
            # Calculate expected average sales
            summary_["Expected_Avg_Sales"] = ggf.conditional_expected_average_profit(
                summary_["frequency"], summary_["monetary_value"]
            )
            
            progress_bar.progress(80)
            
            # Step 9: Calculate CLV using same parameters as original
            status_text.text('Calculating customer lifetime value...')
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
            summary_["profit_margin"] = summary_["predicted_clv"] * profit_margin
            
            progress_bar.progress(90)
            
            # Step 10: K-means clustering
            status_text.text('Segmenting customers...')
            
            # Select columns for clustering exactly as in original
            col = ["predicted_purchases", "Expected_Avg_Sales", "predicted_clv", "profit_margin"]
            new_df = summary_[col]
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(new_df)
            
            # K-means clustering with 4 clusters
            k_model = KMeans(n_clusters=4, init="k-means++", max_iter=1000, random_state=42)
            k_model_fit = k_model.fit(scaled_data)
            
            # Get labels and map to segments
            labels = pd.Series(k_model_fit.labels_, name="Labels")
            summary_ = pd.concat([summary_, labels], axis=1)
            
            # Map numerical labels to descriptive labels - using same mapping as original
            label_mapper = {0: "Low", 1: "High", 2: "Medium", 3: "V_High"}
            summary_["Labels"] = summary_["Labels"].map(label_mapper)
            
            progress_bar.progress(100)
            status_text.text('Done!')
            
            # Display results
            st.markdown("## Customer Lifetime Value and Segmentation Results")
            st.dataframe(summary_)
            
            # Display model performance metrics
            st.markdown("## Model Performance Metrics")
            
            # Calculate prediction metrics using holdout data
            summary_cal_holdout['Predicted_purchases_holdout'] = bgf.conditional_expected_number_of_purchases_up_to_time(
                184,  # 184 days - same as original code
                summary_cal_holdout['frequency_cal'], 
                summary_cal_holdout['recency_cal'], 
                summary_cal_holdout['T_cal']
            )
            
            # Filter out NaN values
            mask = ~(np.isnan(summary_cal_holdout['frequency_holdout']) | np.isnan(summary_cal_holdout['Predicted_purchases_holdout']))
            valid_actual = summary_cal_holdout['frequency_holdout'][mask]
            valid_predicted = summary_cal_holdout['Predicted_purchases_holdout'][mask]
            
            # Calculate metrics
            bgf_mae_purchase = mean_absolute_error(valid_actual, valid_predicted)
            bgf_mse_purchase = mean_squared_error(valid_actual, valid_predicted)
            bgf_rmse_purchase = sqrt(bgf_mse_purchase)
            r2 = r2_score(valid_actual, valid_predicted)
            
            # Create metrics DataFrame
            metrics_df = pd.DataFrame({
                'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
                'Value': [bgf_mae_purchase, bgf_mse_purchase, bgf_rmse_purchase, r2]
            })
            
            st.dataframe(metrics_df)
            
            # Create visualizations
            
            # Cluster pie chart - similar to original analysis
            st.markdown("## Customer Segmentation")
            segment_counts = summary_['Labels'].value_counts()
            
            # Create Altair pie chart
            pie_data = pd.DataFrame({
                'Segment': segment_counts.index,
                'Count': segment_counts.values,
                'Percentage': segment_counts.values / segment_counts.sum() * 100
            })
            
            # Create pie chart using matplotlib as in original code
            fig, ax = plt.subplots(figsize=(8, 8))
            explode = [0.05] * len(segment_counts)
            
            # Function to show percentage and count
            def autopct_format(values):
                def my_format(pct):
                    total = sum(values)
                    val = int(round(pct*total/100.0))
                    return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)
                return my_format
            
            # Create the pie chart
            ax.pie(segment_counts, 
                labels=segment_counts.index, 
                explode=explode,
                autopct=autopct_format(segment_counts),
                startangle=180,
                shadow=False)
            
            ax.set_title("Customer Segment Distribution", fontsize=16)
            st.pyplot(fig)
            
            # Scatterplot of predicted purchases vs predicted CLV
            st.markdown("## Purchase vs CLV by Segment")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            for label in summary_['Labels'].unique():
                subset = summary_[summary_['Labels'] == label]
                ax2.scatter(subset['predicted_purchases'], subset['predicted_clv'], 
                           label=label, alpha=0.5)
            
            ax2.set_xlabel('Predicted Purchases')
            ax2.set_ylabel('Predicted CLV')
            ax2.set_title('Customer Segments by Purchase Prediction and CLV')
            ax2.legend()
            st.pyplot(fig2)
            
            # Add download button for results
            csv = summary_.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="customer_clv_segments.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    # Call the function with the uploaded data
    process_data(uploaded_file, days, profit_margin, cal_end_date, obs_end_date)
    
else:
    st.info("Please upload a transaction data CSV file")
