# serpengine/google_searcher.py

# to run python -m serpengine.google_searcher

import os, logging, time, asyncio
import requests
import httpx  
from bs4 import BeautifulSoup
from typing import List

from .utils import parse_tld  # adjust or remove if unused
from dotenv import load_dotenv
from .schemes import SearchHit, UsageInfo, SERPMethodOp

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
        if not link:
            return False
        return link.startswith("http")

    def is_link_leads_to_a_website(self, link: str) -> bool:
        excluded_extensions = ['.pdf', '.doc', '.docx', '.ppt', 
                               '.pptx', '.xls', '.xlsx', '.zip']
        lower_link = link.lower()
        return not any(lower_link.endswith(ext) for ext in excluded_extensions)

    def is_blocked_page(self, response_text: str) -> bool:
        text_lower = response_text.lower()
        block_signals = [
            "unusual traffic", "/sorry/", "detected suspicious activity",
            "captcha", "enablejs?sei=", "birkaç saniye içinde",
            "httpservice/retry/enablejs"
        ]
        return any(signal in text_lower for signal in block_signals)

    def search(self, query: str) -> SERPMethodOp:
        """
        Scrape Google HTML results.
        Returns a SERPMethodOp with:
          - name="scrape"
          - results: List[SearchHit]
          - usage.cost=0.0
          - elapsed_time
        """
        start = time.time()
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        logger.debug(f"[Scraper] Sending GET request to: {search_url}")
        response = requests.get(search_url, headers=self.headers, allow_redirects=True)

        snippet = response.text[:500].replace('\n', ' ')
        if self.is_blocked_page(snippet):
            logger.debug("[Scraper] blocked_page detected")

        if response.status_code != 200:
            logger.info(f"[Scraper] HTML search failed (status {response.status_code})")
            elapsed = time.time() - start
            return SERPMethodOp(
                name="scrape",
                results=[],
                usage=UsageInfo(cost=0.0),
                elapsed_time=elapsed
            )

        soup = BeautifulSoup(response.text, 'html.parser')
        divs = soup.find_all('div', class_='tF2Cxc')
        if not divs:
            logger.debug("[Scraper] no standard results found")
            elapsed = time.time() - start
            return SERPMethodOp(
                name="scrape",
                results=[],
                usage=UsageInfo(cost=0.0),
                elapsed_time=elapsed
            )

        valid_links: List[str] = []
        for g in divs:
            a = g.find('a')
            if a and a.get('href'):
                href = a['href']
                if self.is_link_format_valid(href) and self.is_link_leads_to_a_website(href):
                    valid_links.append(href)

        hits = [SearchHit(link=lnk, metadata="", title="") for lnk in valid_links]
        elapsed = time.time() - start
        logger.info(f"[Scraper] Returning {len(hits)} hits in {elapsed:.2f}s")

        return SERPMethodOp(
            name="scrape",
            results=hits,
            usage=UsageInfo(cost=0.0),
            elapsed_time=elapsed
        )

    def search_with_api(
        self,
        query: str,
        num_results: int,
        google_search_api_key=google_search_api_key,
        cse_id=cse_id
    ) -> SERPMethodOp:
        """
        Use Google Custom Search API.
        Returns a SERPMethodOp with:
          - name="api"
          - results: List[SearchHit]
          - usage.cost=0.0  (update later if you compute actual cost)
          - elapsed_time
        """
        start = time.time()
        logger.debug(f"[API] query='{query}', num_results={num_results}")
        url = "https://www.googleapis.com/customsearch/v1"
        items = []
        idx = 1

        while len(items) < num_results:
            params = {
                'q': query,
                'key': google_search_api_key,
                'cx': cse_id,
                'num': min(10, num_results - len(items)),
                'start': idx
            }
            logger.debug(f"[API] GET {url} params={params}")
            resp = requests.get(url, params=params)
            data = resp.json()

            if 'items' in data:
                items.extend(data['items'])
            else:
                logger.debug(f"[API] no more items at start={idx}")
                break

            idx += 10

        raw_links = [item.get('link', '') for item in items]
        valid_links = [
            link for link in raw_links
            if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link)
        ]

        hits = [SearchHit(link=lnk, metadata="", title="") for lnk in valid_links]
        elapsed = time.time() - start
        logger.info(f"[API] Returning {len(hits)} hits in {elapsed:.2f}s")

        return SERPMethodOp(
            name="api",
            results=hits,
            usage=UsageInfo(cost=0.0),
            elapsed_time=elapsed
        )
    
    
    async def async_search_with_api(
        self,
        query: str,
        num_results: int,
        google_search_api_key=google_search_api_key,
        cse_id=cse_id
    ) -> SERPMethodOp:
        start, url = time.time(), "https://www.googleapis.com/customsearch/v1"
        items, idx = [], 1
        async with httpx.AsyncClient() as client:
            while len(items) < num_results:
                params = {
                    "q": query, "key": google_search_api_key, "cx": cse_id,
                    "num": min(10, num_results - len(items)), "start": idx
                }
                data = (await client.get(url, params=params)).json()
                if "items" in data: items.extend(data["items"])
                else: break
                idx += 10

        links = [it.get("link", "") for it in items]
        valid = [l for l in links
                 if self.is_link_format_valid(l)
                 and self.is_link_leads_to_a_website(l)]
        hits  = [SearchHit(link=l, metadata="", title="") for l in valid]
        return SERPMethodOp(
            name="api_async",
            results=hits,
            usage=UsageInfo(cost=0.0),
            elapsed_time=time.time() - start
        )
    



async def _async_demo():
    gs = GoogleSearcher()
    print("\n--- ASYNC API ---")
    print(await gs.async_search_with_api("Enes Kuzucu", 5))


def main():
    # sample="Enes Kuzucu"
    # gs = GoogleSearcher()
    
    # print("\n--- API METHOD ---")
    # api_op = gs.search_with_api(sample, 5)
    # print(api_op)

    
    asyncio.run(_async_demo())


if __name__ == "__main__":
    main()