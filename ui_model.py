import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the trained model and dataset
model_path = 'laptop_recommendation_model_tuned_3.pkl'
laptop_data_path = 'laptop_cleaned.csv'

# Load model and data
model = joblib.load(model_path)
laptop_data = pd.read_csv(laptop_data_path)

# Define Streamlit app
st.title("AI-Powered Laptop Recommendation System")
st.write("Get personalized laptop recommendations based on your preferences!")

# User inputs
budget = st.selectbox("Select your budget:", ['Budget', 'Mid-Range', 'High-End', 'Premium'])
intended_use = st.selectbox("Select intended use:", ['General Use', 'Gaming', 'Business', 'Programming'])
brand = st.selectbox("Preferred Brand:", laptop_data['brand'].unique())
operating_system = st.selectbox("Preferred Operating System:", laptop_data['operating_system'].unique())
min_ram = st.slider("Minimum RAM (GB):", min_value=4, max_value=64, step=4)
storage = st.slider("Minimum Storage (SSD in GB):", min_value=128, max_value=2000, step=128)
graphics_card = st.selectbox("Graphics Card Requirement:", ['No Preference', 'Dedicated'])

# Process user inputs to filter laptops
user_input = {
    'budget': budget,
    'intended_use': intended_use,
    'brand': brand,
    'operating_system': operating_system,
    'min_ram': min_ram,
    'storage': storage,
    'graphics_card': 1 if graphics_card == 'Dedicated' else 0
}

# Filter function
filtered_data = laptop_data.copy()

# Debugging: Initial dataset size
st.write(f"Initial dataset size: {filtered_data.shape}")

# Filter by budget
if user_input['budget']:
    filtered_data = filtered_data[filtered_data['price_category'] == user_input['budget']]
    st.write(f"After budget filter: {filtered_data.shape}")

# Filter by intended use (e.g., gaming)
if user_input['intended_use'] == 'Gaming':
    filtered_data = filtered_data[filtered_data['performance_score'] >= 70]
    st.write(f"After intended use filter (Gaming): {filtered_data.shape}")

# Filter by brand preference
if user_input['brand']:
    filtered_data = filtered_data[filtered_data['brand'] == user_input['brand']]
    st.write(f"After brand filter: {filtered_data.shape}")

# Filter by operating system
if user_input['operating_system']:
    filtered_data = filtered_data[filtered_data['operating_system'] == user_input['operating_system']]
    st.write(f"After operating system filter: {filtered_data.shape}")

# Filter by minimum RAM
filtered_data = filtered_data[filtered_data['ram_gb'] >= user_input['min_ram']]
st.write(f"After RAM filter: {filtered_data.shape}")

# Filter by storage preference
filtered_data = filtered_data[filtered_data['ssd_gb'] >= user_input['storage']]
st.write(f"After storage filter: {filtered_data.shape}")

# Filter by graphics card requirement
if user_input['graphics_card']:
    filtered_data = filtered_data[filtered_data['graphics'] >= user_input['graphics_card']]
    st.write(f"After graphics card filter: {filtered_data.shape}")

# Display recommended laptops
st.subheader("Recommended Laptops")
if filtered_data.empty:
    st.write("No laptops match your criteria. Please adjust your filters and try again.")
else:
    st.write(filtered_data[['model_name', 'spec_score', 'ram_gb', 'ssd_gb', 'price_category', 'model_link']])

    # Explainability using SHAP
    st.subheader("Model Explainability")
    st.write("Why were these laptops recommended?")
    explainer = shap.TreeExplainer(model)
    columns_to_drop = [col for col in ['model_name', 'price_category', 'model_link'] if col in filtered_data.columns]
    shap_values = explainer.shap_values(filtered_data.drop(columns=columns_to_drop))
    shap.summary_plot(shap_values, filtered_data.drop(columns=columns_to_drop), feature_names=filtered_data.drop(columns=columns_to_drop).columns)
    st.pyplot(plt.gcf(), bbox_inches='tight')
