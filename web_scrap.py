import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
import os

# Set up logging for debugging
logging.basicConfig(filename='scraper_debug.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the cleaned dataset
file_path = r'C:\Users\bansa\Downloads\two_preprocessing\laptop.csv'
laptop_data = pd.read_csv(file_path)

# ScraperAPI setup
SCRAPER_API_KEY = '8afc1018407c0f89d27ef887c6f8e999'

# Function to scrape product links from Flipkart using ScraperAPI
def get_flipkart_link_scraperapi(query):
    base_url = f"https://www.flipkart.com/search?q={query.replace(' ', '+')}"
    url = f"https://api.scraperapi.com?api_key={SCRAPER_API_KEY}&url={base_url}"
    
    headers = {
        'User-Agent': random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]),
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the product link
            link_tag = soup.find("a", class_="_1fQZEK") or soup.find("a", class_="_2rpwqI")
            
            if link_tag:
                product_link = "https://www.flipkart.com" + link_tag['href']
                return product_link
            else:
                logging.warning(f"No link found for query: {query}")
                save_failed_html(response.text, query)
                return None
        else:
            logging.warning(f"Failed to retrieve data for query: {query} with status code: {response.status_code}")
            save_failed_html(response.text, query)
            return None

    except requests.RequestException as e:
        logging.error(f"Error occurred for query: {query} - {e}")
        return None

# Function to save HTML response to a file for debugging purposes
def save_failed_html(html_content, query):
    file_name = f"failed_response_{query.replace(' ', '_')}.html"
    file_path = os.path.join("failed_html", file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(html_content)

# Add a new column for links
laptop_data['product_link'] = None

# Iterate through the dataset and get links for each laptop model
for index, row in laptop_data.iterrows():
    model_name = row['model_name']
    link = get_flipkart_link_scraperapi(model_name)
    
    if link:
        laptop_data.at[index, 'product_link'] = link
        print(f"Link for model '{model_name}': {link}")
    else:
        logging.info(f"No link found for model: {model_name}")
    
    # Sleep to prevent being blocked by the website (random sleep interval)
    time.sleep(random.uniform(5, 15))

# Save the updated dataset with product links
laptop_data.to_csv('laptop_with_links.csv', index=False)

print("Scraping completed and dataset updated with product links.")
