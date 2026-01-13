import requests
from bs4 import BeautifulSoup, FeatureNotFound
import sys
import tempfile
from pathlib import Path
from selenium import webdriver
try:
    # Optional fallback. Selenium 4+ includes Selenium Manager, which is preferred.
    from webdriver_manager.chrome import ChromeDriverManager  # type: ignore
except Exception:  # pragma: no cover
    ChromeDriverManager = None  # type: ignore

try:
    # Optional fallback for Edge.
    from webdriver_manager.microsoft import EdgeChromiumDriverManager  # type: ignore
except Exception:  # pragma: no cover
    EdgeChromiumDriverManager = None  # type: ignore

from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options  # add this

from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService


def _make_soup(html: str) -> BeautifulSoup:
    """Parse HTML into a BeautifulSoup object.

    Prefer lxml (faster, more lenient) when installed, but fall back to the
    built-in html.parser so the project runs even without optional parsers.
    """
    try:
        return BeautifulSoup(html, 'lxml')
    except FeatureNotFound:
        return BeautifulSoup(html, 'html.parser')

def fetch_static(url, headers=None):
    headers = headers or {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return _make_soup(resp.text)


def fetch_dynamic_edge(url, *, headless: bool = True):
    """Fetch a page rendered by Microsoft Edge (Chromium).

    This is useful when ChromeDriver is blocked or unstable on a given machine.
    """
    edge_options = EdgeOptions()
    if headless:
        edge_options.add_argument('--headless=new')
    edge_options.add_argument('--disable-gpu')
    edge_options.add_argument('--window-size=1920,1080')
    edge_options.add_argument('--disable-extensions')
    edge_options.add_argument('--no-first-run')
    edge_options.add_argument('--no-default-browser-check')
    edge_options.add_argument('--remote-debugging-port=0')

    tmp_base = None
    if sys.platform.startswith('win'):
        tmp_base = Path.cwd() / '.selenium'
        tmp_base.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='selenium-profile-', dir=str(tmp_base) if tmp_base else None) as profile_dir:
        edge_options.add_argument(f'--user-data-dir={profile_dir}')

        # Prefer Selenium Manager first.
        try:
            driver = webdriver.Edge(options=edge_options)
        except Exception as e:
            if EdgeChromiumDriverManager is None:
                raise
            raw_path = Path(EdgeChromiumDriverManager().install())
            driver_path = raw_path
            if driver_path.suffix.lower() != '.exe':
                # webdriver-manager sometimes returns a non-exe marker file.
                # Avoid an expensive recursive scan; check a small set of likely locations.
                candidates = [
                    raw_path.parent / 'msedgedriver.exe',
                    raw_path.parent / 'msedgedriver' / 'msedgedriver.exe',
                    raw_path.parent.parent / 'msedgedriver.exe',
                ]
                driver_path = next((p for p in candidates if p.exists()), driver_path)

            service = EdgeService(str(driver_path))
            driver = webdriver.Edge(service=service, options=edge_options)

        try:
            driver.get(url)
            soup = _make_soup(driver.page_source)
        finally:
            driver.quit()

        return soup

def fetch_dynamic(url, *, headless: bool = True, browser: str = 'edge'):
    """Fetch a page rendered by a browser.

    browser: 'edge' (default) or 'chrome'
    """
    browser = (browser or 'edge').strip().lower()
    if browser in {'edge', 'msedge', 'microsoft-edge'}:
        return fetch_dynamic_edge(url, headless=headless)

    if browser not in {'chrome', 'google-chrome'}:
        raise ValueError("browser must be 'edge' or 'chrome'")

    chrome_options = Options()
    if headless:
        # Prefer Chrome's "new" headless mode (more stable on recent Chrome).
        chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--no-first-run')
    chrome_options.add_argument('--no-default-browser-check')
    chrome_options.add_argument('--remote-debugging-port=0')

    # Linux/container flags (avoid setting these on Windows/macOS; they can cause weirdness).
    if sys.platform.startswith('linux'):
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

    # A temporary user profile fixes many "DevToolsActivePort file doesn't exist" crashes.
    # On some Windows setups, using a shorter, user-writable directory is more reliable
    # than the default %TEMP% path.
    tmp_base = None
    if sys.platform.startswith('win'):
        tmp_base = Path.cwd() / '.selenium'
        tmp_base.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='selenium-profile-', dir=str(tmp_base) if tmp_base else None) as profile_dir:
        chrome_options.add_argument(f'--user-data-dir={profile_dir}')

        # Prefer Selenium Manager (built into Selenium 4.6+). It downloads/chooses the
        # correct driver automatically and avoids common webdriver-manager issues.
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            if ChromeDriverManager is None:
                raise
            # Fallback to webdriver-manager.
            raw_path = Path(ChromeDriverManager().install())
            driver_path = raw_path
            # webdriver-manager may return a notice file (not an exe) with newer driver zips.
            if driver_path.suffix.lower() != '.exe':
                # Avoid a recursive scan; check a few likely locations.
                candidates = [
                    raw_path.parent / 'chromedriver.exe',
                    raw_path.parent / 'chromedriver' / 'chromedriver.exe',
                    raw_path.parent.parent / 'chromedriver.exe',
                ]
                driver_path = next((p for p in candidates if p.exists()), driver_path)

            service = Service(str(driver_path))
            try:
                driver = webdriver.Chrome(service=service, options=chrome_options)
            except OSError as oe:
                # Common on Windows when the driver architecture doesn't match.
                raise RuntimeError(
                    'Failed to start ChromeDriver. If you are on Windows, ensure you have a 64-bit Chrome installed '
                    'and that the downloaded driver matches your architecture. '
                    f'Original error: {oe}'
                ) from oe
            except Exception:
                raise e

        try:
            driver.get(url)
            soup = _make_soup(driver.page_source)
        finally:
            driver.quit()

        return soup

def parse_ticker_data(soup, ticker):
    data = {'ticker': ticker}
    # TODO: extract real fields (price, etc.) via soup selectors
    return data
