import numpy as np
import streamlit as st

# Define price bins and labels
price_bins = [0, 30000, 60000, 100000, np.inf]
price_labels = ['Budget', 'Mid-Range', 'High-End', 'Premium']

# Take price as input using a slider
price = st.slider("Select your price (in INR):", min_value=0, max_value=200000, step=1000)

# Categorize price into budget ranges
if price > 0:
    budget = price_labels[np.digitize(price, price_bins) - 1]  # Get the corresponding budget label
    st.write(f"Based on the entered price, your budget category is: **{budget}**")
else:
    st.write("Please select a valid price to determine your budget category.")
