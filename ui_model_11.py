import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
import time
from scraper_linked_ui import scrape_laptop_details_parallel  # Import the scraping function
from prompt import generate_user_input  # Import the LLM function
from predict_price import predict_price


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
laptop_data['brand'] = laptop_data['brand'].map(brand_mapping)
laptop_data['operating_system'] = laptop_data['operating_system'].map(operating_system_mapping)

# Remove duplicate entries based on the 'model_name' column
laptop_data = laptop_data.drop_duplicates(subset='model_name', keep='first').reset_index(drop=True)

# Define Streamlit app
st.set_page_config(layout="wide")

# Center the heading
st.markdown(
    "<h1 style='text-align: center; color: white; font-size: 36px;'>Personalized Laptop Support Engine</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center; color: grey;'>Get personalized laptop based on your preferences!</h3>",
    unsafe_allow_html=True
)

# User inputs in the sidebar
with st.sidebar:
    st.markdown(
        "<div style='background-color: #f0f0f5; padding: 20px; border-radius: 10px;'>",
        unsafe_allow_html=True
    )
    st.subheader("User Inputs")

    # Option to choose input method
    manual_input_enabled = st.radio("Choose Input Method:", ["Set Inputs Manually", "Describe Your Case"])
    user_input = {}

    if manual_input_enabled == "Set Inputs Manually":
        # User inputs
        price_bins = [0, 30000, 60000, 100000, np.inf]
        price_labels = ['Budget', 'Mid-Range', 'High-End', 'Premium']
        budget = None
        price = st.slider("Select your price (in INR):", min_value=0, max_value=350000, step=1000)
        if price > 0:
            budget = price_labels[np.digitize(price, price_bins) - 1]  # Get the corresponding budget label

        intended_use = st.selectbox("Select intended use:", ['General Use', 'Gaming', 'Business', 'Programming'])
        brand = st.selectbox("Preferred Brand:", ['Any'] + list(laptop_data['brand'].dropna().unique()))
        operating_system = st.selectbox(
            "Preferred Operating System:",
            ['Any'] + list(laptop_data['operating_system'].dropna().unique())
        )
        min_ram = st.select_slider("Minimum RAM (GB):", options=[4, 8, 16, 32, 64])
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
        if st.button("Predict Price"):
            storage_map={

                256: '256 GB',
                512: '512 GB',
                1000: '1 TB',
                2000: '2 TB'
            }
            features=[
                brand_mapping.get(user_input.get('brand','Any'),-1),
                10,
                user_input.get('min_ram',4),
                storage_map.get(user_input.get('storage','256 GB'),256),
                0,
                operating_system_mapping.get(user_input.get('operating_system','Any'),-1),
                user_input.get('graphics_card',0),
                4,
                8,
                70 if user_input.get('intended_use')=='Gaming' else 60,
                1920,
                1080,
                130.0
            ]
            predicted_price=predict_price(features)
            st.subheader(f"Predicted Price: INR {predicted_price:,.2f}")

    else:
        # LLM-based input
        user_message = st.text_area(
            "Describe your requirements (e.g., 'I need a laptop for gaming under 80000 INR')"
        )
        if st.button("Generate Specifications"):
            if user_message.strip():
                with st.spinner("Analyzing your requirements..."):
                    gemini_output = generate_user_input(user_message)
                    st.write("Raw Gemini Output:", gemini_output)  # Debugging output
                    try:
                        if isinstance(gemini_output, dict):
                            user_input = gemini_output  # Directly use if output is a dict
                        else:
                            user_input = json.loads(gemini_output)  # Parse JSON string

                        # numerical
                        # budget=user_input.get('budget',25000)
                        # price_bins = [0, 30000, 60000, 100000, np.inf]
                        # price_labels = ['Budget', 'Mid-Range', 'High-End', 'Premium']
                        # budget = price_labels[np.digitize(budget, price_bins) - 1]

                        raw_budget = user_input.get('budget', 25000)
                        price_bins = [0, 30000, 60000, 100000, np.inf]
                        price_labels = ['Budget', 'Mid-Range', 'High-End', 'Premium']
                        mapped_budget = price_labels[np.digitize(raw_budget, price_bins) - 1]
                        user_input['budget'] = mapped_budget  # Update budget in user_input


                        intended_use = user_input.get('intended_use', 'General Use')
                        brand = user_input.get('brand', 'Any')
                        operating_system = user_input.get('operating_system', 'Any')
                        min_ram = user_input.get('min_ram', 4)
                        # storage = user_input.get('storage', '256 GB')
                        graphics_card = user_input.get('graphics_card', 0)
                        storage_mapping = {
                                      256: '256 GB',
                                       512: '512 GB',
                                       1024: '1 TB',
                                       2048: '2 TB'
                         }
                        user_input['storage'] = storage_mapping.get(user_input['storage'], '256 GB')  # Default to '256 GB' if not found






                        st.success("Specifications generated successfully!")
                        st.write("Parsed User Input:", user_input)
                    except json.JSONDecodeError as e:
                        st.error(f"Failed to parse the Gemini API output: {e}")
                        user_input = {}
            else:
                st.warning("Please enter a valid description.")
        st.markdown("</div>", unsafe_allow_html=True)

# Handle empty user input
if not user_input:
    st.warning("No valid input provided. Please fill the form or describe your case.")
    st.stop()

filtered_data = laptop_data.copy()

# Only show outputs if inputs are provided and results exist
show_results = False

# Apply filters based on user input
if user_input.get('budget'):
    filtered_data = filtered_data[filtered_data['price_category'] == user_input['budget']]
    show_results = True

if user_input.get('intended_use') == 'Gaming':
    filtered_data = filtered_data[filtered_data['spec_score'] >= 70]
    show_results = True

if user_input.get('brand') and user_input['brand'] != 'Any':
    filtered_data = filtered_data[filtered_data['brand'] == user_input['brand']]
    show_results = True

if user_input.get('operating_system') and user_input['operating_system'] != 'Any':
    filtered_data = filtered_data[filtered_data['operating_system'] == user_input['operating_system']]
    show_results = True

if user_input.get('min_ram'):
    filtered_data = filtered_data[filtered_data['ram_gb'] >= user_input['min_ram']]
    show_results = True

if user_input.get('storage'):
    storage_gb_mapping = {'256 GB': 256, '512 GB': 512, '1 TB': 1024, '2 TB': 2048}
    storage_gb = storage_gb_mapping.get(user_input['storage'], 0)
    filtered_data = filtered_data[filtered_data['ssd_gb'] >= storage_gb]
    show_results = True

if user_input.get('graphics_card'):
    filtered_data = filtered_data[filtered_data['graphics'] >= user_input['graphics_card']]
    show_results = True


# # extra part 
# st.write("User Input for Storage:", user_input['storage'])
# st.write("Filtered Dataset After Storage:", filtered_data[['model_name', 'ssd_gb']].head())



# Sort and limit results
# Limit results to a maximum of 6 laptops
top_recommendations = filtered_data.sort_values(by='spec_score', ascending=False).reset_index(drop=True).head(6)

# Display recommended laptops with loading animation
if show_results and not top_recommendations.empty:
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
    top_recommendations = pd.merge(
        top_recommendations, scraped_details, how='left', left_on='model_link', right_on='link'
    )

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
            product_name = row['product_name'] if row.get('product_name') and row['product_name'] != "N/A" else row['model_name']
            image_url = row['image_url'] if row.get('image_url') and row['image_url'] != "N/A" else "https://via.placeholder.com/200"
            price = row['price'] if row.get('price') and row['price'] != "N/A" else "Currently Unavailable"
            rating = row['rating'] if row.get('rating') and row['rating'] != "N/A" else "No rating available"

            st.markdown(f"""
                <div style='border: 2px solid #e6e6e6; padding: 20px; margin-bottom: 20px; border-radius: 10px;'>
                    <h4>{product_name}</h4>
                    <img src='{image_url}' alt='Laptop Image' style='width: 200px; height: 200px; float: left; margin-right: 20px;'>
                    <strong>Spec Score:</strong> {row['spec_score']}<br>
                    <strong>RAM:</strong> {row['ram_gb']} GB<br>
                    <strong>Storage:</strong> {row['ssd_gb']} GB<br>
                    <strong>Price Category:</strong> {row['price_category']}<br>
                    <strong>Price:</strong> {price}<br>
                    <strong>Rating:</strong> {rating}<br>
                    <a href='{row['model_link']}' target='_blank' style='color: #3498db;'>View Details</a>
                </div>
            """, unsafe_allow_html=True)

        seen_models.add(row['model_name'])

    # Optionally, include SHAP explainability code if desired
    # st.subheader("Model Explainability")
    # st.write("Why were these laptops recommended?")
    # Revert brand and operating system to encoded values for SHAP
    # inverse_brand_mapping = {v: k for k, v in brand_mapping.items()}
    # inverse_os_mapping = {v: k for k, v in operating_system_mapping.items()}
    # top_recommendations['brand'] = top_recommendations['brand'].map(inverse_brand_mapping)
    # top_recommendations['operating_system'] = top_recommendations['operating_system'].map(inverse_os_mapping)

    # explainer = shap.TreeExplainer(model)
    # columns_to_drop = [col for col in ['model_name', 'price_category', 'model_link', 'link',
    #                                    'product_name', 'price', 'image_url', 'rating']
    #                    if col in top_recommendations.columns]
    # shap_values = explainer.shap_values(top_recommendations.drop(columns=columns_to_drop))
    # shap.summary_plot(
    #     shap_values, top_recommendations.drop(columns=columns_to_drop),
    #     feature_names=top_recommendations.drop(columns=columns_to_drop).columns
    # )
    # st.pyplot(plt.gcf(), bbox_inches='tight')

else:
    if show_results:
        st.warning("No laptops match your criteria.")
    else:
        # Initial message when starting the app
        st.markdown(
            "<div style='text-align: center; padding: 20px;'>"
            "<h3>Welcome to the Personalized Laptop Support Engine!</h3>"
            "<p>Use the sidebar to enter your preferences and find the perfect laptop for your needs.</p>"
            "</div>",
            unsafe_allow_html=True
        )
