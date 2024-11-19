import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
from scraper_linked_ui import scrape_laptop_details_parallel  # Import the scraping function
from prompt import generate_user_input

# Load the trained model and dataset
model_path = 'laptop_recommendation_model_tuned_3.pkl'
laptop_data_path = 'laptop_cleaned copy.csv'

# Load model and data
try:
    model = joblib.load(model_path)
    laptop_data = pd.read_csv(laptop_data_path)
except Exception as e:
    st.error(f"Failed to load the model or dataset: {e}")
    st.stop()

# Define brand and operating system mapping based on the provided mappings
brand_mapping = {
    0: 'AGB', 1: 'Acer', 2: 'Apple', 3: 'Asus', 4: 'Avita', 5: 'Chuwi', 6: 'Dell',
    7: 'Fujitsu', 8: 'Gigabyte', 9: 'HP', 10: 'Honor', 11: 'Huawei', 12: 'Infinix',
    13: 'LG', 14: 'Lenovo', 15: 'MSI', 16: 'Microsoft', 17: 'Nokia', 18: 'Realme',
    19: 'Samsung', 20: 'Ultimus', 21: 'Xiaomi'
}

operating_system_mapping = {
    0: 'DOS', 1: 'Chrome', 2: 'Mac', 3: 'Ubuntu', 4: 'Windows'
}

# Replace encoded values with original names in the dataset
if 'brand' in laptop_data.columns:
    laptop_data['brand'] = laptop_data['brand'].map(brand_mapping)
if 'operating_system' in laptop_data.columns:
    laptop_data['operating_system'] = laptop_data['operating_system'].map(operating_system_mapping)

# Remove duplicate entries based on the 'model_name' column
laptop_data = laptop_data.drop_duplicates(subset='model_name', keep='first').reset_index(drop=True)

# Define Streamlit app
st.set_page_config(layout="wide")

# Center the heading
st.markdown("<h1 style='text-align: center;'>AI-Powered Laptop Recommendation System</h1>", unsafe_allow_html=True)

# User inputs in the sidebar
with st.sidebar:
    st.subheader("User Inputs")
    manual_input_enabled = st.radio("Choose Input Method:", ["Set Inputs Manually", "Describe Your Case"])
    user_input = {}

    if manual_input_enabled == "Set Inputs Manually":
        budget = st.slider("Budget (INR):", 0, 350000, 1000)
        intended_use = st.selectbox("Intended Use:", ['General Use', 'Gaming', 'Business', 'Programming'])
        brand = st.selectbox("Preferred Brand:", ['Any'] + list(laptop_data['brand'].dropna().unique()))
        operating_system = st.selectbox("Operating System:", ['Any'] + list(laptop_data['operating_system'].dropna().unique()))
        min_ram = st.select_slider("Minimum RAM (GB):", [4, 8, 16, 32, 64])
        storage = st.selectbox("Storage:", ['256 GB', '512 GB', '1 TB', '2 TB'])
        graphics_card = st.selectbox("Graphics Card:", ['No Preference', 'Dedicated'])

        user_input = {
            'budget': budget,
            'intended_use': intended_use,
            'brand': brand,
            'operating_system': operating_system,
            'min_ram': min_ram,
            'storage': storage,
            'graphics_card': 1 if graphics_card == 'Dedicated' else 0
        }
    else:
        user_message = st.text_area("Describe your requirements:")
        if st.button("Generate Specifications"):
         if user_message.strip():
            gemini_output = generate_user_input(user_message)
            st.write("Raw Gemini Output:", gemini_output)  # Debugging output
            st.write("Type of Gemini Output:", type(gemini_output))  # Debugging type

        try:
            # Parse Gemini output based on its type
            if isinstance(gemini_output, dict):
                user_input = gemini_output  # Directly use the dictionary
            else:
                user_input = json.loads(gemini_output)  # Parse JSON string
            st.write("Parsed User Input:", user_input)
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse the Gemini API output: {e}")
            user_input = {}
        else:
          st.warning("Please enter a valid description.")


# Handle empty user input
if not user_input:
    st.warning("No valid input provided. Please fill the form or describe your case.")
    st.stop()

# Filtering logic
show_results = False
filtered_data = laptop_data.copy()

st.write("Initial Filtered Data:", filtered_data)

if user_input.get('budget'):
    filtered_data = filtered_data[filtered_data['price_category'] == user_input['budget']]
    st.write("After Budget Filter:", filtered_data)

if user_input.get('intended_use') == 'Gaming':
    filtered_data = filtered_data[filtered_data['spec_score'] >= 70]
    st.write("After Intended Use Filter:", filtered_data)

if user_input.get('brand') != 'Any':
    filtered_data = filtered_data[filtered_data['brand'] == user_input['brand']]
    st.write("After Brand Filter:", filtered_data)

# Additional filtering for RAM, storage, etc.
# Display recommendations or fallback
if not filtered_data.empty:
    st.write("Top Recommendations:", filtered_data.head(6))
else:
    st.warning("No laptops match your criteria.")

