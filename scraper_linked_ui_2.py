import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
import concurrent.futures

def scrape_laptop_detail(link, laptop_data):
    # Set up the list of user agents
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
    ]

    # Placeholder for scraped details
    if pd.notna(link) and link != "None":
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/'
        }
        try:
            response = requests.get(link, headers=headers)
            if response.status_code != 200:
                return (link, "N/A", "N/A", "N/A", "N/A")

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract product name
            product_name = "N/A"
            if "flipkart.com" in link:
                name_tag = soup.find("span", class_="VU-ZeZ")
            elif "amazon.in" in link:
                name_tag = soup.find("span", id="productTitle")
            if name_tag:
                product_name = name_tag.get_text(strip=True)

            # Extract product price
            price = "N/A"
            if "flipkart.com" in link:
                price_tag = soup.find("div", class_="Nx9bqj CxhGGd yKS41a")
            elif "amazon.in" in link:
                price_tag = soup.find("span", class_="a-price-whole")
            if price_tag:
                price = price_tag.get_text(strip=True)

            # Extract product image URL
            image_url = "N/A"
            if "flipkart.com" in link:
                image_tag = soup.find("img", class_="DBuyf4 IZeXxJ jLEJ7H")
            elif "amazon.in" in link:
                image_tag = soup.find("img", id="landingImage")
            if image_tag:
                image_url = image_tag.get("src")

            # Extract product rating
            rating = "N/A"
            if "flipkart.com" in link:
                rating_tag = soup.find("div", class_="_3LWZlK")
            elif "amazon.in" in link:
                rating_tag = soup.find("span", class_="a-icon-alt")
            if rating_tag:
                rating = rating_tag.get_text(strip=True)

            return (link, product_name, price, image_url, rating)

        except Exception as e:
            logging.error(f"Error occurred for URL: {link} - {e}")
            return (link, "N/A", "N/A", "N/A", "N/A")
    else:
        logging.info(f"Invalid link found: {link}")
        # If no link is available, retrieve price from the dataset
        model_name = link.split('/')[-1]  # Assuming model_name is in the link, you may change this logic based on actual structure
        laptop_info = laptop_data[laptop_data['model_name'] == model_name]
        if not laptop_info.empty:
            price = laptop_info.iloc[0]['price']  # Get the price from the dataset
        else:
            price = "N/A"
        return (link, "N/A", price, "N/A", "N/A")

def scrape_laptop_details_parallel(laptop_links, laptop_data):
    # Set up logging for debugging
    logging.basicConfig(filename='linked_scraper.log', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    laptop_details = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_link = {executor.submit(scrape_laptop_detail, link, laptop_data): link for link in laptop_links}
        for future in concurrent.futures.as_completed(future_to_link):
            result = future.result()
            laptop_details.append(result)
            print(f"Scraped details for link '{result[0]}': Name - {result[1]}, Price - {result[2]}, Image URL - {result[3]}, Rating - {result[4]}")

    # Create a DataFrame to hold the scraped details
    details_df = pd.DataFrame(laptop_details, columns=['link', 'product_name', 'price', 'image_url', 'rating']).head(6)
    return details_df

# Example of how to call this function from ui_model.py
if __name__ == "__main__":
    sample_links = ["https://www.flipkart.com/some-laptop-product", "https://www.amazon.in/some-other-laptop"]
    laptop_data = pd.read_csv('laptop_cleaned.csv')  # Load the dataset
    scraped_details = scrape_laptop_details_parallel(sample_links, laptop_data)
    scraped_details.to_csv('scraped_laptop_details.csv', index=False)
    print("Scraping completed and details saved.")
