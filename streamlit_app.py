import streamlit as st
import pandas as pd
import math
import numpy as np
import datetime
import altair as alt
import time
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import lifetimes package with proper error handling
try:
    import lifetimes
    from lifetimes.plotting import *
    from lifetimes import BetaGeoFitter
    from lifetimes.utils import calibration_and_holdout_data
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    st.error("The lifetimes package is not installed. Please install it with 'pip install lifetimes'.")
    st.stop()

# Seed for reproducibility
np.random.seed(42)

# App title and description
st.title("CLV Prediction and Segmentation App")
st.markdown("Upload transaction data to get the customer lifetime value and their segmentation")

# Display header image
st.image("https://www.adlibweb.com/wp-content/uploads/2020/06/customer-lifetime-value.jpg", use_container_width=True)

# File uploader
data = st.file_uploader("File Uploader", type=['csv'])

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/f8/SIMS_official_logo.jpg", width=150)
st.sidebar.markdown("**MBA Project by Vrushali Ovhal**")
st.sidebar.title("Input Features :pencil:")

# Sidebar inputs
days = st.sidebar.slider("Select The No. Of Days", min_value=1, max_value=365, step=1)
profit = st.sidebar.slider("Select the Profit Margin", min_value=0.01, max_value=0.09, step=0.01)

# Date inputs for calibration period
st.sidebar.subheader("Calibration Period Settings")
cal_period_end = st.sidebar.date_input("Calibration Period End Date", value=datetime.date(2011, 6, 8))
obs_period_end = st.sidebar.date_input("Observation Period End Date", value=datetime.date(2011, 12, 9))

# Sidebar instructions
st.sidebar.markdown("""
Before uploading the file, please select the input features first.

Also, please make sure the columns are in proper format with transaction data containing:
- CustomerID
- InvoiceDate 
- Quantity
- UnitPrice

**Note:** Only Use "CSV" File.
""")

# Main function to process data
if data is not None:
    def load_data(data, days, profit, cal_period_end, obs_period_end):
        try:
            # Load transaction data
            df = pd.read_csv(data)
            
            # Process transaction data - similar to first code
            with st.status("Processing transaction data...", expanded=True) as status:
                st.write("Cleaning data...")
                
                # Handle column names
                if 'Customer ID' in df.columns:
                    df.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)
                
                # Check required columns
                required_cols = ['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    return
                
                # Clean data
                st.write("Converting dates...")
                df=df.drop_duplicates()
                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date
                
                st.write("Removing negative quantities...")
                df = df[(df.Quantity > 0)]
                
                
                st.write("Removing missing CustomerIDs...")
                df.dropna(axis=0, subset=["CustomerID"], inplace=True)
                df.dropna(axis = 0, subset = ["Description"], inplace = True)
                
                st.write("Calculating amount...")
                df['Amount'] = df['Quantity'] * df['UnitPrice']
                # st.dataframe(df.describe())
                
                # Create RFM data
                st.write("Creating RFM summaries...")
                rfmt_data = lifetimes.utils.summary_data_from_transaction_data(
                    df, 'CustomerID', 'InvoiceDate', monetary_value_col='Amount'
                )
                # st.dataframe(rfmt_data.describe())
                
                # Create summary_bgf (to match first code)
                summary_bgf = rfmt_data.copy()
                
                # Fit BG/NBD model
                st.write("Fitting BG/NBD model...")
                bgf = BetaGeoFitter(penalizer_coef=0.5)
                bgf.fit(summary_bgf['frequency'], summary_bgf['recency'], summary_bgf['T'])

                # Predict purchases
                t = days
                summary_bgf['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
                    t, summary_bgf['frequency'], summary_bgf['recency'], summary_bgf['T']
                )
                
                # Create calibration and holdout data (exactly as in first code)
                st.write("Creating calibration and holdout data...")
                cal_end_date = pd.to_datetime(cal_period_end).date()
                obs_end_date = pd.to_datetime(obs_period_end).date()
                
                summary_cal_holdout = calibration_and_holdout_data(
                    df, 'CustomerID', 'InvoiceDate',
                    calibration_period_end=cal_end_date.strftime('%Y-%m-%d'),
                    observation_period_end=obs_end_date.strftime('%Y-%m-%d')
                )
                
                # Refit the model using calibration data
                st.write("Refitting model with calibration data...")
                bgf.fit(
                    summary_cal_holdout['frequency_cal'],
                    summary_cal_holdout['recency_cal'],
                    summary_cal_holdout['T_cal'],
                    penalizer_coef=0.5
                )
                

                
                # Set actual purchases from holdout (divide by 10 as in first code)
                st.write("Calculating actual purchases...")
                summary_bgf['actual_purchases'] = summary_cal_holdout['frequency_holdout'] / 10
                
                # Fill NA values with 0
                summary_bgf = summary_bgf.fillna(value=0)
                
                # Filter for monetary value > 0 and frequency > 0
                summary_ = summary_bgf[(summary_bgf['monetary_value'] > 0) & (summary_bgf['frequency'] > 0)]
                
                # Fit Gamma Gamma model
                st.write("Fitting Gamma-Gamma model...")
                ggf = lifetimes.GammaGammaFitter(penalizer_coef=0.0)
                ggf.fit(summary_['frequency'], summary_['monetary_value'])
                
                # Calculate expected average sales
                summary_['Expected_Avg_Sales'] = ggf.conditional_expected_average_profit(
                    summary_['frequency'], summary_['monetary_value']
                )

                # st.dataframe(summary_.head())
                
                # Calculate CLV - exactly as in first code
                st.write("Calculating Customer Lifetime Value...")
                summary_['predicted_clv'] = ggf.customer_lifetime_value(
                    bgf,
                    summary_['frequency'],
                    summary_['recency'],
                    summary_['T'],
                    summary_['monetary_value'],
                    time=days,  # Fixed at 30 days as in first code
                    freq='D',
                    discount_rate=0.01
                )
                
                # Calculate profit margin
                summary_['profit_margin'] = summary_['predicted_clv'] * profit
                
                # Reset index to include CustomerID in the final results
                summary_ = summary_.reset_index()
                
                # Prepare for clustering
                st.write("Performing customer segmentation...")
                col = ['predicted_purchases', 'Expected_Avg_Sales', 'predicted_clv', 'profit_margin']
                new_df = summary_[col]
                
                # Scale data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(new_df)
                
                # K-means clustering with 4 clusters
                k_model = KMeans(n_clusters=4, init='k-means++', max_iter=1000, random_state=42)
                k_model_fit = k_model.fit(scaled_data)
                
                # Get labels
                labels = k_model_fit.labels_
                labels = pd.Series(labels, name='Labels')
                
                # Combine with original dataframe
                summary_ = pd.concat([summary_, labels], axis=1)
                
                # Map labels
                label_mapper = {0: 'Low', 1: 'High', 2: 'Medium', 3: 'V_High'}
                summary_['Labels'] = summary_['Labels'].map(label_mapper)
                
                status.update(label="Processing complete!", state="complete")
            
            # Display results
            st.subheader("Customer Lifetime Value and Segmentation Results")
            st.dataframe(summary_)
            
            
            # Visualization of segments
            st.subheader("Customer Segmentation Visualization")
            
            # Create count bar chart with Altair
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
            
            # Scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = sns.scatterplot(
                data=summary_, 
                x='predicted_purchases', 
                y='predicted_clv', 
                hue='Labels',
                palette='Set1',
                alpha=0.6,
                ax=ax
            )
            plt.title('Customer Segments by Predicted Purchases and CLV')
            plt.xlabel('Predicted Purchases')
            plt.ylabel('Predicted CLV')
            st.pyplot(fig)
            
            # Pie chart
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            segment_counts = summary_['Labels'].value_counts()
            segment_labels = segment_counts.index
            
            # Function to display both percentage and count
            def autopct_format(values):
                def my_format(pct):
                    total = sum(values)
                    val = int(round(pct*total/100.0))
                    return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)
                return my_format
                
            ax2.pie(
                segment_counts, 
                labels=segment_labels,
                startangle=180, 
                explode=[0.05, 0.05, 0.05, 0.05],
                autopct=autopct_format(segment_counts),
                textprops={'fontsize': 12}
            )
            ax2.set_title('Customer Segment Distribution', fontsize=16)
            st.pyplot(fig2)

           # Top 10 Customers by Predicted CLV
            st.subheader("Top 10 Customers by Predicted CLV")
            
            top_customers = summary_[['CustomerID', 'predicted_clv']].sort_values(by='predicted_clv', ascending=False).head(10)
            
            bar_fig, bar_ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=top_customers,
                x='CustomerID',
                y='predicted_clv',
                palette='viridis',
                ax=bar_ax
            )
            bar_ax.set_title("Top 10 Customers by Predicted CLV")
            bar_ax.set_xlabel("Customer ID")
            bar_ax.set_ylabel("Predicted CLV")
            bar_ax.set_xticklabels(bar_ax.get_xticklabels(), rotation=45)  # optional: rotate x labels
            st.pyplot(bar_fig)


            
            # Download button
            csv = summary_.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="customer_segmentation_result_streamlit.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    # Call the function with the uploaded data
    st.markdown("## Customer Lifetime Prediction Result :bar_chart:")
    load_data(data, days, profit, cal_period_end, obs_period_end)
    
else:
    st.info("Please upload a CSV file with transaction data")
