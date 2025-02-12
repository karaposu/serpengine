# here is google_searcher.py

import os
import logging
import requests
from bs4 import BeautifulSoup
from typing import List

from .utils import parse_tld  # Adjust if needed or remove if you're not using parse_tld
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

google_search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
cse_id = os.getenv("GOOGLE_CSE_ID")


class GoogleSearcher:
    def __init__(self, user_agent: str = None):
        """
        If no user agent is provided, use a default.
        """
        if user_agent is None:
            user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        self.headers = {"User-Agent": user_agent}

    def is_link_format_valid(self, link: str) -> bool:
        """
        Basic check for valid link formatting.
        For example, confirm it starts with http:// or https://, or is not empty.
        """
        if not link:
            return False
        return link.startswith("http")

    def is_link_leads_to_a_website(self, link: str) -> bool:
        """
        Check if the link is likely leading to a standard web page (HTML),
        not a file or something else.
        """
        excluded_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.zip']
        lower_link = link.lower()
        return not any(lower_link.endswith(ext) for ext in excluded_extensions)

    def is_link_leads_to_a_file(self, link: str) -> bool:
        """
        Check if the link points to a direct file download (e.g., PDF, ZIP, etc.)
        """
        file_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.zip']
        lower_link = link.lower()
        return any(lower_link.endswith(ext) for ext in file_extensions)
    

    def is_blocked_page(self, response_text: str) -> bool:
        """
        Heuristically checks if this is likely a Google block/captcha page.
        """
        text_lower = response_text.lower()

        block_signals = [
            "unusual traffic",
            "/sorry/",                   # often in an English captcha URL
            "detected suspicious activity",
            "captcha",
            "enablejs?sei=",            # seen in your snippet
            "birkaç saniye içinde",     # Turkish text
            "httpservice/retry/enablejs"
        ]
        return any(signal in text_lower for signal in block_signals)


    def search(self, query: str) -> List[str]:
        """
        Scrape Google HTML results.
        Returns a list of valid links (by default).
        """
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        logger.debug(f"[Scraper] Sending GET request to: {search_url}")
        response = requests.get(search_url, headers=self.headers, allow_redirects=True)

        logger.debug(f"[Scraper] Response status code: {response.status_code}")
        logger.debug(f"[Scraper] Final response URL: {response.url}")

        # Log snippet of the response to see if it's a captcha page
        response_snippet = response.text[:500].replace('\n', ' ')

        blocked_page=self.is_blocked_page(response_snippet)
        logger.debug(f" response_snippet: {response_snippet}")
        logger.debug(f"is blocked_page: {blocked_page}")
        if blocked_page:
           logger.debug(f"[Scraper] blocked_page")
        

        if response.status_code != 200:
            logger.info(f'[Scraper] Google HTML search failed. status code: {response.status_code}')
            return []

        # If we get 200, we parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        search_divs = soup.find_all('div', class_='tF2Cxc')
        logger.debug(f"[Scraper] Found {len(search_divs)} 'tF2Cxc' divs (search results).")

        if not search_divs:
            # Possibly a captcha or "unusual traffic" page even though code = 200
            logger.debug("[Scraper] No standard search results found. Possibly blocked or captcha.")
            return []

        links: List[str] = []
        for g in search_divs:
            anchor = g.find('a')
            if not anchor or not anchor.get('href'):
                continue
            raw_link = anchor['href']
            if self.is_link_format_valid(raw_link) and self.is_link_leads_to_a_website(raw_link):
                links.append(raw_link)

        logger.debug(f"[Scraper] Raw extracted links: {links}")
        logger.info(f"[Scraper] Returning {len(links)} link(s) from HTML search.")
        return links

    def search_with_api(self,
                        query: str,
                        num_results: int,
                        google_search_api_key=google_search_api_key,
                        cse_id=cse_id) -> List[str]:
        """
        Use Google Custom Search API to get search results.
        Returns a list of valid links that appear to be websites.
        """
        logger.debug(f"[API] Searching with query='{query}', num_results={num_results}")
        search_url = "https://www.googleapis.com/customsearch/v1"
        results = []
        start_index = 1

        while len(results) < num_results:
            params = {
                'q': query,
                'key': google_search_api_key,
                'cx': cse_id,
                'num': min(10, num_results - len(results)),
                'start': start_index
            }
            logger.debug(f"[API] GET {search_url} with params={params}")
            response = requests.get(search_url, params=params)
            response_data = response.json()
            logger.debug(f"[API] Response data keys: {list(response_data.keys())}")

            if 'items' in response_data:
                results.extend(response_data['items'])
                logger.debug(f"[API] Fetched {len(response_data['items'])} items, total so far: {len(results)}")
            else:
                logger.debug(f"[API] No more results found at start={start_index}")
                break

            start_index += 10

        # Extract the link field
        raw_links = [item.get('link', '') for item in results]
        valid_links = []
        for link in raw_links:
            if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                valid_links.append(link)

        logger.info(f"[API] Found {len(valid_links)} valid link(s). (raw total={len(raw_links)})")
        return valid_links
