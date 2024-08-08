
#adding necessary libraries
import streamlit as st
import pandas as pd
import lifetimes
import math
import numpy as np
import xlrd
import datetime
np.random.seed(42)
import altair as alt
import time
import warnings
warnings.filterwarnings("ignore")
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from lifetimes.plotting import *
from lifetimes import ParetoNBDFitter


np.float = float 

st.markdown(""" # CLV Prediction and Segmentation App 


Upload the RFM data to get the customer lifetime value and their segmentation

	""")


st.image("https://ultracommerce.co/wp-content/uploads/2022/04/maximize-customer-lifetime-value.png", use_column_width = True)


data = st.file_uploader("File Uploader")

st.sidebar.image("https://www.adlibweb.com/wp-content/uploads/2020/06/customer-lifetime-value.jpg", width = 150)
st.sidebar.markdown(""" **CDAC Project ** """)


st.sidebar.title("Input Features :pencil:")


days = st.sidebar.slider("Select The No. Of Days", min_value = 1, max_value = 365, step = 1)

profit = st.sidebar.slider("Select the Profit Margin", min_value = 0.01, max_value = 0.09, step = 0.01)


# t_days = days
# profit_m = profit

# slider_data = {
# 	"Days": days,
# 	"Profit": profit
# }

# st.sidebar.markdown("""

# ### Selected Input Features :page_with_curl:

# 	""")

# features = pd.DataFrame(slider_data, index = [0])

#st.sidebar.write(features)

st.sidebar.markdown("""

Before uploading the file, please select the input features first.

Also, please make sure the columns are in proper format. For reference you can download the [dummy data](https://github.com/tejas-tilekar/CDAC-Project/blob/main/sample_file.csv).

**Note:** Only Use "CSV" File.

	""")


if data is not None:

	def load_data(data, days , profit):

		input_data = pd.read_csv(data)

		input_data = pd.DataFrame(input_data.iloc[:, 1:])

        #Pareto Model

		pareto_model = lifetimes.ParetoNBDFitter(penalizer_coef = 0.0)
		pareto_model.fit(input_data["frequency"],input_data["recency"],input_data["T"])
		input_data["p_not_alive"] = 1-pareto_model.conditional_probability_alive(input_data["frequency"], input_data["recency"], input_data["T"])
		input_data["p_alive"] = pareto_model.conditional_probability_alive(input_data["frequency"], input_data["recency"], input_data["T"])
		t = days
		input_data["predicted_purchases"] = pareto_model.conditional_expected_number_of_purchases_up_to_time(t, input_data["frequency"], input_data["recency"], input_data["T"])
        

        #Gamma Gamma Model

		# idx = input_data[(input_data["frequency"] <= 0.0)]
		# idx = idx.index
		# input_data = input_data.drop(idx, axis = 0)
		# m_idx = input_data[(input_data["monetary_value"] <= 0.0)].index
		# input_data = input_data.drop(m_idx, axis = 0)

		input_data = input_data[(input_data["frequency"] > 0) & (input_data["monetary_value"] > 0)]


		input_data.reset_index().drop("index", axis = 1, inplace = True)

		ggf_model =  lifetimes.GammaGammaFitter(penalizer_coef=0.0)

		ggf_model.fit(input_data["frequency"], input_data["monetary_value"])

		input_data["expected_avg_sales"] = ggf_model.conditional_expected_average_profit(input_data["frequency"], input_data["monetary_value"])

		input_data["predicted_clv"] = ggf_model.customer_lifetime_value(pareto_model, input_data["frequency"], input_data["recency"], input_data["T"], input_data["monetary_value"], time = days, freq = 'D', discount_rate = 0.01)

		input_data["profit_margin"] = input_data["predicted_clv"]*profit

		input_data = input_data.reset_index().drop("index", axis = 1)

		#K-Means Model

		col = ["predicted_purchases", "expected_avg_sales", "predicted_clv", "profit_margin"]

		new_df = input_data[col]

		k_model = KMeans(n_clusters = 5, init = "k-means++", max_iter = 1000)
		k_model_fit = k_model.fit(new_df)

		labels = k_model_fit.labels_
		

		labels = pd.Series(labels, name = "Labels")


		input_data = pd.concat([input_data, labels], axis = 1)

		label_mapper = dict({0 : "Medium", 3: "Low", 1: "V_High", 2: "V_Low", 4:"High"})

		input_data["Labels"] = input_data["Labels"].map(label_mapper)

		#saving the input data in the separate variable 

		# download = input_data

		st.write(input_data)

		#adding a count bar chart

		fig = alt.Chart(input_data).mark_bar().encode(

			y = "Labels:N",
			x = "count(Labels):Q"

			)

		#adding a annotation to the chart

		text = fig.mark_text(

			align = "left",
			baseline = "middle",
			dx = 3

			).encode(

			text = "count(Labels):Q"

			)


		chart = (fig+text)

		#showing the chart

		st.altair_chart(chart, use_container_width = True)

		#creating a button to download the result

		# if st.button("Download"):
		# 	st.write("Successfully Downloaded!!! Please Check Your Default Download Location...:smile:" )
		# 	return input_data.to_csv("yeah.csv")


	#calling the function		

	st.markdown("""

		## Customer Lifetime Prediction Result :bar_chart:

		""")

	load_data(data,days,profit)

else:
	st.text("Please Upload the CSV File")
