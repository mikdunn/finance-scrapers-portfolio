import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

def fetch_static(url, headers=None):
    headers = headers or {'User-Agent': 'Mozilla/5.0...'}  # Rotate UAs
    resp = requests.get(url, headers=headers)
    return BeautifulSoup(resp.text, 'lxml')

def fetch_dynamic(url):
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=webdriver.ChromeOptions().add_argument('--headless'))
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    driver.quit()
    return soup

def parse_ticker_data(soup, ticker):
    data = {'ticker': ticker}
    # e.g., price = soup.select_one('[data-testid="qsp-price"]').text
    return data  # Dict for easy pd.DataFrame
