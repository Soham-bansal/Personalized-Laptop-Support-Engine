import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import time

# Load the trained model and dataset
model_path = 'laptop_recommendation_model_tuned_3.pkl'
laptop_data_path = 'laptop_cleaned.csv'

# Load model and data
model = joblib.load(model_path)
laptop_data = pd.read_csv(laptop_data_path)

# Define brand and operating system mapping based on the provided mappings
brand_mapping = {
    0: 'AGB',
    1: 'Acer',
    2: 'Apple',
    3: 'Asus',
    4: 'Avita',
    5: 'Chuwi',
    6: 'Dell',
    7: 'Fujitsu',
    8: 'Gigabyte',
    9: 'HP',
    10: 'Honor',
    11: 'Huawei',
    12: 'Infinix',
    13: 'LG',
    14: 'Lenovo',
    15: 'MSI',
    16: 'Microsoft',
    17: 'Nokia',
    18: 'Realme',
    19: 'Samsung',
    20: 'Ultimus',
    21: 'Xiaomi'
}

operating_system_mapping = {
    0: 'DOS',
    1: 'Chrome',
    2: 'Mac',
    3: 'Ubuntu',
    4: 'Windows'
}

# Replace encoded values with original names in the dataset
laptop_data['brand'] = laptop_data['brand'].map(brand_mapping)
laptop_data['operating_system'] = laptop_data['operating_system'].map(operating_system_mapping)

# Define Streamlit app
st.set_page_config(layout="wide")

# Center the heading
st.markdown("<h1 style='text-align: center; color: white; font-size: 36px;'>AI-Powered Laptop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>Get personalized laptop recommendations based on your preferences!</h3>", unsafe_allow_html=True)

# User inputs in the sidebar
with st.sidebar:
    st.markdown("<div style='background-color: #f0f0f5; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
    st.subheader("User Inputs")
    # User inputs
    budget = st.selectbox("Select your budget:", ['Budget', 'Mid-Range', 'High-End', 'Premium'])
    intended_use = st.selectbox("Select intended use:", ['General Use', 'Gaming', 'Business', 'Programming'])
    brand = st.selectbox("Preferred Brand:", laptop_data['brand'].dropna().unique())
    operating_system = st.selectbox("Preferred Operating System:", laptop_data['operating_system'].dropna().unique())
    min_ram = st.slider("Minimum RAM (GB):", min_value=4, max_value=64, step=4)
    storage = st.selectbox("Minimum Storage (SSD):", ['256 GB', '512 GB', '1 TB', '2 TB'])
    graphics_card = st.selectbox("Graphics Card Requirement:", ['No Preference', 'Dedicated'])
    st.markdown("</div>", unsafe_allow_html=True)

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

# Only show outputs if inputs are provided and results exist
show_results = False

# Initial message when starting the app
if not any(user_input.values()):
    st.markdown("<div style='text-align: center; padding: 20px;'><h3>Welcome to the AI-Powered Laptop Recommendation System!</h3><p>Use the sidebar to enter your preferences and find the perfect laptop for your needs.</p></div>", unsafe_allow_html=True)

# Filter by budget
if user_input['budget']:
    filtered_data = filtered_data[filtered_data['price_category'] == user_input['budget']]
    show_results = True

# Filter by intended use (e.g., gaming)
if user_input['intended_use'] == 'Gaming':
    filtered_data = filtered_data[filtered_data['spec_score'] >= 70]
    show_results = True

# Filter by brand preference
if user_input['brand']:
    filtered_data = filtered_data[filtered_data['brand'] == user_input['brand']]
    show_results = True

# Filter by operating system
if user_input['operating_system']:
    filtered_data = filtered_data[filtered_data['operating_system'] == user_input['operating_system']]
    show_results = True

# Filter by minimum RAM
filtered_data = filtered_data[filtered_data['ram_gb'] >= user_input['min_ram']]
show_results = True

# Filter by storage preference
storage_gb = {'256 GB': 256, '512 GB': 512, '1 TB': 1024, '2 TB': 2048}[user_input['storage']]
filtered_data = filtered_data[filtered_data['ssd_gb'] >= storage_gb]
show_results = True

# Filter by graphics card requirement
if user_input['graphics_card']:
    filtered_data = filtered_data[filtered_data['graphics'] >= user_input['graphics_card']]
    show_results = True

# Sort results by spec score
top_recommendations = filtered_data.sort_values(by='spec_score', ascending=False).reset_index(drop=True).head(6)

# Display recommended laptops with loading animation
if show_results:
    with st.spinner('Finding the best laptops for you...'):
        time.sleep(2)  # Simulate loading time
        st.markdown("<div style='padding: 20px;'>", unsafe_allow_html=True)
        st.subheader("Recommended Laptops")
        if top_recommendations.empty:
            st.write("<h4 style='color: red;'>No laptops match your criteria. Please adjust your filters and try again.</h4>", unsafe_allow_html=True)
        else:
            top_recommendations.index = top_recommendations.index + 1  # Start index from 1
            # Format model links to be clickable
            top_recommendations['model_link'] = top_recommendations['model_link'].apply(lambda x: f"<a href='{x}' target='_blank'>Link</a>" if pd.notna(x) else "None")
            st.write(top_recommendations[['model_name', 'spec_score', 'ram_gb', 'ssd_gb', 'price_category', 'model_link']]
                     .style.set_table_styles([
                         {'selector': 'th', 'props': [('background-color', '#4F81BD'), ('color', 'white'), ('text-align', 'center')]},
                         {'selector': 'td', 'props': [('text-align', 'center')]}]).to_html(escape=False), unsafe_allow_html=True)

            # Explainability using SHAP
            st.subheader("Model Explainability")
            st.write("Why were these laptops recommended?")

            # Revert brand and operating system to encoded values for SHAP
            inverse_brand_mapping = {v: k for k, v in brand_mapping.items()}
            inverse_os_mapping = {v: k for k, v in operating_system_mapping.items()}
            top_recommendations['brand'] = top_recommendations['brand'].map(inverse_brand_mapping)
            top_recommendations['operating_system'] = top_recommendations['operating_system'].map(inverse_os_mapping)

            explainer = shap.TreeExplainer(model)
            columns_to_drop = [col for col in ['model_name', 'price_category', 'model_link'] if col in top_recommendations.columns]
            shap_values = explainer.shap_values(top_recommendations.drop(columns=columns_to_drop))
            shap.summary_plot(shap_values, top_recommendations.drop(columns=columns_to_drop), feature_names=top_recommendations.drop(columns=columns_to_drop).columns)
            st.pyplot(plt.gcf(), bbox_inches='tight')
        st.markdown("</div>", unsafe_allow_html=True)
