import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import time
from scraper_linked_ui import scrape_laptop_details_parallel  # Import the scraping function

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

# Remove duplicate entries based on the 'model_name' column
laptop_data = laptop_data.drop_duplicates(subset='model_name', keep='first').reset_index(drop=True)

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

# Apply filters based on user input
if user_input['budget']:
    filtered_data = filtered_data[filtered_data['price_category'] == user_input['budget']]
    show_results = True

if user_input['intended_use'] == 'Gaming':
    filtered_data = filtered_data[filtered_data['spec_score'] >= 70]
    show_results = True

if user_input['brand']:
    filtered_data = filtered_data[filtered_data['brand'] == user_input['brand']]
    show_results = True

if user_input['operating_system']:
    filtered_data = filtered_data[filtered_data['operating_system'] == user_input['operating_system']]
    show_results = True

filtered_data = filtered_data[filtered_data['ram_gb'] >= user_input['min_ram']]
show_results = True

storage_gb = {'256 GB': 256, '512 GB': 512, '1 TB': 1024, '2 TB': 2048}[user_input['storage']]
filtered_data = filtered_data[filtered_data['ssd_gb'] >= storage_gb]
show_results = True

if user_input['graphics_card']:
    filtered_data = filtered_data[filtered_data['graphics'] >= user_input['graphics_card']]
    show_results = True

# Sort and limit results
# Limit results to a maximum of 6 laptops
top_recommendations = filtered_data.sort_values(by='spec_score', ascending=False).reset_index(drop=True).head(6)

# Display recommended laptops with loading animation
if show_results:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            """
            <div style='text-align: center;'>
                <div class='loading-spinner'></div>
                <h4>Finding the best laptops for you...</h4>
            </div>
            <style>
                .loading-spinner {
                    border: 16px solid #f3f3f3;
                    border-top: 16px solid #3498db;
                    border-radius: 50%;
                    width: 120px;
                    height: 120px;
                    animation: spin 2s linear infinite;
                    margin: auto;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        time.sleep(2)  # Simulate loading time
    placeholder.empty()  # Remove the loading spinner after loading

    # Get detailed product information
    laptop_links = top_recommendations['model_link'].dropna().unique()[:6]  # Limit links to 6 unique valid links
    scraped_details = scrape_laptop_details_parallel(laptop_links)

    # Merge scraped details into top_recommendations
    scraped_details = scraped_details.drop_duplicates(subset='link', keep='first')
    top_recommendations = pd.merge(top_recommendations, scraped_details, how='left', left_on='model_link', right_on='link')

    # Display each laptop with its details
    seen_models = set()  # To keep track of displayed models and prevent duplicates

    for index, row in top_recommendations.iterrows():
        # Skip if this model has already been displayed
        if row['model_name'] in seen_models:
            continue

        # If link is unavailable, use only the dataset information
        if pd.isna(row['model_link']) or row['model_link'] == "None":
            st.markdown(f"""
                <div style='border: 2px solid #e6e6e6; padding: 20px; margin-bottom: 20px; border-radius: 10px;'>
                    <h4>{row['model_name']}</h4>
                    <strong>Spec Score:</strong> {row['spec_score']}<br>
                    <strong>RAM:</strong> {row['ram_gb']} GB<br>
                    <strong>Storage:</strong> {row['ssd_gb']} GB<br>
                    <strong>Price Category:</strong> {row['price_category']}<br>
                    <a href='#' style='color: gray; text-decoration: none;'>Link Not Available</a>
                </div>
            """, unsafe_allow_html=True)
        else:
            # For laptops with valid links, use scraped details and fall back if necessary
            product_name = row['product_name'] if row['product_name'] != "N/A" else row['model_name']
            image_url = row['image_url'] if row['image_url'] != "N/A" else "https://via.placeholder.com/200"
            price = row['price'] if row['price'] != "N/A" else "Currently Unavailable"
            rating = row['rating'] if row['rating'] != "N/A" else "No rating available"

            st.markdown(f"""
                <div style='border: 2px solid #e6e6e6; padding: 20px; margin-bottom: 20px; border-radius: 10px;'>
                    <h4>{product_name}</h4>
                    <img src='{image_url}' width='200' /><br>
                    <strong>Price:</strong> {price}<br>
                    <strong>Rating:</strong> {rating}<br>
                    <strong>Spec Score:</strong> {row['spec_score']}<br>
                    <a href='{row['model_link']}' target='_blank'>View on Website</a>
                </div>
            """, unsafe_allow_html=True)

        # Add model to seen_models to prevent duplicates
        seen_models.add(row['model_name'])

    # Explainability using SHAP
    st.subheader("Model Explainability")
    st.write("Why were these laptops recommended?")

    # Revert brand and operating system to encoded values for SHAP
    inverse_brand_mapping = {v: k for k, v in brand_mapping.items()}
    inverse_os_mapping = {v: k for k, v in operating_system_mapping.items()}
    top_recommendations['brand'] = top_recommendations['brand'].map(inverse_brand_mapping)
    top_recommendations['operating_system'] = top_recommendations['operating_system'].map(inverse_os_mapping)

    explainer = shap.TreeExplainer(model)
    columns_to_drop = [col for col in ['model_name', 'price_category', 'model_link', 'link', 'product_name', 'price', 'image_url', 'rating'] if col in top_recommendations.columns]
    shap_values = explainer.shap_values(top_recommendations.drop(columns=columns_to_drop))
    shap.summary_plot(shap_values, top_recommendations.drop(columns=columns_to_drop), feature_names=top_recommendations.drop(columns=columns_to_drop).columns)
    st.pyplot(plt.gcf(), bbox_inches='tight')
