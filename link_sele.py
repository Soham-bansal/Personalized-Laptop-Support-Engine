import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup

# Load dataset
df = pd.read_csv('laptop_cleaned copy.csv')  # Replace with your actual dataset path

# ScraperAPI Credentials
SCRAPERAPI_KEY = '8afc1018407c0f89d27ef887c6f8e999'
SCRAPERAPI_PROXY = f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001"

# Edge WebDriver Setup
service = Service(r'C:\Users\bansa\Downloads\edgedriver_win64\msedgedriver.exe')  # Replace with the correct path

# Iterate over each row in the dataset
links = []

for index, row in df.iterrows():
    laptop_name = row['model_name']
    print(f"Searching for: {laptop_name}")

    # Set up Edge options with ScraperAPI proxy
    options = webdriver.EdgeOptions()
    options.add_argument(f'--proxy-server={SCRAPERAPI_PROXY}')

    # Initialize the Selenium WebDriver with Edge
    driver = webdriver.Edge(service=service, options=options)

    try:
        # Open Smartprix website
        base_url = "https://www.smartprix.com/laptops/"
        driver.get(base_url)
        time.sleep(random.uniform(3, 6))  # Random delay for human-like behavior

        # Wait until the search bar is present (using the correct name 'q')
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, 'q'))
        )
        # Move the mouse over the search box to simulate human interaction
        actions = ActionChains(driver)
        actions.move_to_element(search_box).perform()
        time.sleep(random.uniform(1, 3))  # Pause to simulate reading/searching

        # Enter the laptop name in the search bar
        search_box.send_keys(laptop_name)
        search_box.send_keys(Keys.RETURN)

        # Wait for a random time, mimicking browsing behavior
        time.sleep(random.uniform(4, 7))

        # Click on the first result
        first_result = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a.hover-link'))
        )
        actions.move_to_element(first_result).perform()  # Move the mouse over the first result
        time.sleep(random.uniform(1, 2))  # Simulate hover
        first_result.click()
        time.sleep(random.uniform(4, 7))  # Wait for the product page to load

        # Find either Amazon or Flipkart link by looking for the button with class "buy-btn"
        buy_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'buy-btn'))
        )
        actions.move_to_element(buy_button).perform()  # Move mouse to buy button
        time.sleep(random.uniform(1, 2))  # Pause before clicking
        link = buy_button.get_attribute('href')

    except Exception as e:
        print(f"Error finding link for {laptop_name}: {e}")
        link = None

    # Append the found link or 'No link found' if not present
    if link:
        links.append(link)
        print(f"Link found for {laptop_name}: {link}")  # Print the link in the terminal
    else:
        links.append('No link found')
        print(f"No link found for {laptop_name}")

    # Random delay before next iteration
    time.sleep(random.uniform(5, 10))

    # Quit driver before initializing the next one with a new proxy
    driver.quit()

# Add the new links to the dataframe
df['link'] = links

# Save the updated dataset
df.to_csv('updated_laptop_dataset_links.csv', index=False)

# Final driver quit to ensure cleanup
driver.quit()
