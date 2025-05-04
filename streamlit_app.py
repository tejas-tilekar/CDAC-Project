import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# App title and description
st.title("Customer Segmentation from Pre-Processed RFM Data")
st.markdown("Upload your pre-processed RFM data to visualize customer segments")

# Display header image
st.image("https://ultracommerce.co/wp-content/uploads/2022/04/maximize-customer-lifetime-value.png", use_container_width=True)

# Sidebar
st.sidebar.image("https://www.adlibweb.com/wp-content/uploads/2020/06/customer-lifetime-value.jpg", width=150)
st.sidebar.markdown("**MBA Project**")
st.sidebar.title("Input Features :pencil:")

# Sidebar input for profit margin (in case you need to recalculate)
profit_margin = st.sidebar.slider("Select the Profit Margin", min_value=0.01, max_value=0.09, step=0.01, value=0.05)

# Sidebar instructions
st.sidebar.markdown("""
## Instructions
Upload your pre-processed RFM data CSV file.

**Expected columns:**
- CustomerID
- frequency
- recency
- T
- monetary_value
- predicted_purchases
- actual_purchases (optional)
- Expected_Avg_Sales (optional)
- predicted_clv 
- profit_margin (optional, will recalculate if needed)

The app will segment customers based on your data and provide visualizations.
""")

# File uploader
uploaded_file = st.file_uploader("Upload Pre-processed RFM Data", type=['csv'])

# Main processing function
if uploaded_file is not None:
    def process_rfm_data(rfm_data, profit_margin):
        try:
            # Load the data
            df = pd.read_csv(rfm_data)
            
            # Check for required columns
            required_columns = ["CustomerID", "frequency", "recency", "T", "monetary_value", "predicted_purchases", "predicted_clv"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return
            
            # Show data sample
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Basic data stats
            st.subheader("Dataset Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", df["CustomerID"].nunique())
            with col2:
                st.metric("Avg CLV", f"${df['predicted_clv'].mean():.2f}")
            with col3:
                st.metric("Avg Predicted Purchases", f"{df['predicted_purchases'].mean():.2f}")
            
            # Recalculate profit margin if needed or missing
            if "profit_margin" not in df.columns or st.checkbox("Recalculate profit margin with new value"):
                df["profit_margin"] = df["predicted_clv"] * profit_margin
                st.success(f"Profit margin recalculated using {profit_margin:.2%}")
            
            # Prepare data for clustering
            if "Expected_Avg_Sales" not in df.columns:
                clustering_columns = ["predicted_purchases", "predicted_clv", "profit_margin"]
                st.warning("Expected_Avg_Sales column not found. Using predicted_purchases, predicted_clv, and profit_margin for clustering.")
            else:
                clustering_columns = ["predicted_purchases", "Expected_Avg_Sales", "predicted_clv", "profit_margin"]
            
            # Extract data for clustering
            cluster_data = df[clustering_columns].copy()
            
            # Handle any NaN values
            cluster_data = cluster_data.fillna(0)
            
            # Scale the data for clustering
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Perform K-means clustering
            st.subheader("Customer Segmentation")
            
            # Allow user to choose number of clusters or use default
            if st.checkbox("Customize number of clusters"):
                num_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=4)
            else:
                num_clusters = 4  # Default to 4 clusters as in original code
            
            k_model = KMeans(n_clusters=num_clusters, init="k-means++", max_iter=1000, random_state=42)
            cluster_labels = k_model.fit_predict(scaled_data)
            
            # Map cluster labels to meaningful segments
            # If 4 clusters, use the original mapping
            if num_clusters == 4:
                label_mapper = {0: "Low", 1: "High", 2: "Medium", 3: "V_High"}
            else:
                # Create generic labels for other cluster counts
                label_mapper = {i: f"Segment {i+1}" for i in range(num_clusters)}
            
            # Add segment labels to dataframe
            df["Segment"] = [label_mapper[label] for label in cluster_labels]
            
            # Show cluster centroids
            st.subheader("Cluster Characteristics")
            
            # Transform centers back to original scale
            centers_original = scaler.inverse_transform(k_model.cluster_centers_)
            centers_df = pd.DataFrame(centers_original, columns=clustering_columns)
            centers_df["Segment"] = [label_mapper[i] for i in range(num_clusters)]
            centers_df = centers_df.set_index("Segment")
            
            # Format the centroid values for better readability
            st.dataframe(centers_df.style.format({
                "predicted_purchases": "{:.2f}",
                "predicted_clv": "${:.2f}",
                "profit_margin": "${:.2f}",
                "Expected_Avg_Sales": "${:.2f}" if "Expected_Avg_Sales" in centers_df.columns else None
            }))
            
            # Segment statistics
            st.subheader("Segment Statistics")
            segment_stats = df.groupby("Segment").agg({
                "CustomerID": "count",
                "predicted_purchases": "mean",
                "predicted_clv": "mean",
                "profit_margin": "mean"
            }).rename(columns={"CustomerID": "Count"})
            
            # Format and display segment statistics
            st.dataframe(segment_stats.style.format({
                "Count": "{:.0f}",
                "predicted_purchases": "{:.2f}",
                "predicted_clv": "${:.2f}",
                "profit_margin": "${:.2f}"
            }))
            
            # Visualizations
            st.subheader("Segment Visualizations")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Segment Distribution", "Scatter Plot", "CLV Distribution"])
            
            with tab1:
                # Pie chart of segment distribution
                fig, ax = plt.subplots(figsize=(8, 8))
                segment_counts = df["Segment"].value_counts()
                
                # Function to show percentage and count
                def autopct_format(values):
                    def my_format(pct):
                        total = sum(values)
                        val = int(round(pct*total/100.0))
                        return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)
                    return my_format
                
                # Create the pie chart with the same style as original
                explode = [0.05] * len(segment_counts)
                ax.pie(segment_counts, 
                    labels=segment_counts.index, 
                    explode=explode,
                    autopct=autopct_format(segment_counts.values),
                    startangle=180,
                    shadow=False)
                
                ax.set_title("Customer Segment Distribution", fontsize=16)
                st.pyplot(fig)
            
            with tab2:
                # Scatter plot of predicted purchases vs CLV by segment
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                
                # Create a scatter plot with segments
                sns.scatterplot(
                    data=df, 
                    x="predicted_purchases", 
                    y="predicted_clv", 
                    hue="Segment", 
                    palette="Set1",
                    alpha=0.6,
                    ax=ax2
                )
                
                # Add segment centroids
                for i, segment in enumerate(centers_df.index):
                    ax2.scatter(
                        centers_df.loc[segment, "predicted_purchases"],
                        centers_df.loc[segment, "predicted_clv"], 
                        s=200, 
                        c='black', 
                        marker='X', 
                        label=f"{segment} centroid" if i == 0 else None
                    )
                
                ax2.set_xlabel('Predicted Purchases', fontsize=12)
                ax2.set_ylabel('Predicted CLV ($)', fontsize=12)
                ax2.set_title('Customer Segments: Predicted Purchases vs CLV', fontsize=14)
                
                # Move the legend outside the plot for better visibility
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                st.pyplot(fig2)
            
            with tab3:
                # Box plot of CLV by segment
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                
                # Sort segments by median CLV for better visualization
                segment_order = df.groupby("Segment")["predicted_clv"].median().sort_values().index
                
                sns.boxplot(
                    data=df,
                    x="Segment",
                    y="predicted_clv",
                    order=segment_order,
                    palette="Set1",
                    ax=ax3
                )
                
                ax3.set_xlabel('Segment', fontsize=12)
                ax3.set_ylabel('Predicted CLV ($)', fontsize=12)
                ax3.set_title('CLV Distribution by Segment', fontsize=14)
                
                st.pyplot(fig3)
            
            # Option to download segmented data
            st.subheader("Download Results")
            
            # Prepare download data
            output_columns = ["CustomerID", "frequency", "recency", "T", 
                             "monetary_value", "predicted_purchases", 
                             "predicted_clv", "profit_margin", "Segment"]
            
            download_df = df[[col for col in output_columns if col in df.columns]]
            
            # Add download button
            csv = download_df.to_csv(index=False)
            st.download_button(
                label="Download Segmented Customer Data",
                data=csv,
                file_name="customer_segments.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    # Execute the function with the uploaded file
    process_rfm_data(uploaded_file, profit_margin)
    
else:
    st.info("Please upload your pre-processed RFM data CSV file")
    
    # Show example of expected data format
    st.subheader("Expected Data Format Example:")
    example_data = {
        "CustomerID": [12346, 12347, 12348],
        "frequency": [4.0, 1.0, 0.0],
        "recency": [226.0, 1.0, 0.0],
        "T": [378.0, 376.0, 373.0],
        "monetary_value": [77.18, 637.78, 0.0],
        "predicted_purchases": [1.28, 0.45, 0.23],
        "actual_purchases": [0.0, 0.0, 0.0],
        "Expected_Avg_Sales": [58.83, 637.78, 0.0],
        "predicted_clv": [75.22, 286.42, 0.0],
        "profit_margin": [3.76, 14.32, 0.0]
    }
    st.dataframe(pd.DataFrame(example_data))
