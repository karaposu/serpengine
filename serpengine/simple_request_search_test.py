# simple.py

import logging
import requests
from bs4 import BeautifulSoup
from typing import List

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
    headers = {"User-Agent": user_agent}

    query = "best food in USA"
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"

    # Make the request
    logger.debug(f"Sending GET request to: {search_url}")
    response = requests.get(search_url, headers=headers)

    logger.debug(f"HTTP status code: {response.status_code}")
    logger.debug(f"Final response URL: {response.url}")

    # Parse the HTML response
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find search result containers
    search_divs = soup.find_all('div', class_='tF2Cxc')
    logger.debug(f"Found {len(search_divs)} 'tF2Cxc' divs (search results).")

    if not search_divs:
        logger.debug("No standard search results found. Possibly blocked or captcha.")
        return  # Return early or handle accordingly

    # Extract links
    links: List[str] = []
    for g in search_divs:
        anchor = g.find('a')
        if anchor and anchor.get('href'):
            raw_link = anchor['href']
            links.append(raw_link)

    logger.debug(f"Raw extracted links: {links}")
    logger.info(f"Returning {len(links)} link(s) from HTML search.")

    # Print or return results
    print(f"\nExtracted {len(links)} link(s):")
    for link in links:
        print(link)

if __name__ == "__main__":
    main()
