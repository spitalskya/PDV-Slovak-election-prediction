import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests

# Path to Chromedriver
driver_path = "C:\\Windows\\system32\\chromedriver\\win64-131.0.6778.0\\chromedriver-win64\\chromedriver.exe"  
chrome_options = Options()
#chrome_options.add_argument("--start-maximized")
service = Service(driver_path)

# Initialize WebDriver
driver = webdriver.Chrome(service=service, options=chrome_options)

def search_and_download_pdf(month: str, year: str):
    try:
        print(f"Starting search for {month} {year}...")
        driver.get("https://www.google.com")
        print("Google opened successfully.")
        
        # Accept cookies if prompted
        try:
            print("Checking for cookies prompt...")
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//button[text()="I agree" or contains(text(), "Accept")]'))
            ).click()
            print("Cookies prompt handled.")
        except Exception as e:
            print(f"No cookies prompt or error: {e}")

        # Search for the query
        print("Finding search box...")
        search_box = driver.find_element(By.NAME, "q")
        print("Search box found. Sending query...")
        search_box.send_keys(f"focus prieskum volby {month} {year}" + " filetype:pdf")
        search_box.send_keys(Keys.RETURN)

        # Wait for search results
        print("Waiting for search results...")
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, '//a[contains(@href, ".pdf")]')))
        links = driver.find_elements(By.XPATH, '//a[contains(@href, ".pdf")]')
        
        if links:
            pdf_url = links[0].get_attribute("href")
            print(f"Found PDF URL: {pdf_url}")

            # Download the PDF
            print(f"Downloading PDF for {month} {year}...")
            response = requests.get(pdf_url)
            if response.status_code == 200:
                pdf_name = f"src/focus_scraping/FOCUS_pdf/Prieskum_{year}_{month}.pdf"
                with open(pdf_name, "wb") as file:
                    file.write(response.content)
                print(f"PDF downloaded successfully as {pdf_name}")
            else:
                print(f"Failed to download the PDF for {month} {year}.")
        else:
            print(f"No PDF links found for {month} {year}.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print(f"Finished search for {month} {year}.\n")

# Iterate over months and years
months = ["januar", "februar", "marec", "april", "maj", "jun", "jul", "august", "september", "oktober", "november", "december"]
years = range(2010, 2013)

for year in years:
    for month in months:
        print(month, year)
        search_and_download_pdf(month=month, year=year)

# Close the driver at the end
driver.quit()
