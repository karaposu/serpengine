# serpengine/google_search_api.py

# to run python -m serpengine.google_search_api

import os
import logging
import time
import asyncio
import requests
import httpx
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from .schemes import SearchHit, UsageInfo, SERPMethodOp

load_dotenv()
logger = logging.getLogger(__name__)

# Get API credentials from environment
google_search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")


class GoogleSearchAPI:
    """
    Google Custom Search API wrapper.
    Uses the official Google Custom Search JSON API.
    
    Pricing:
    - First 100 queries per day: Free
    - Additional queries: $5 per 1000 queries ($0.005 per query)
    """
    
    # Pricing tiers
    FREE_QUERIES_PER_DAY = 100
    COST_PER_QUERY_AFTER_FREE = 0.005  # $5 per 1000 queries
    
    def __init__(self, api_key: str = None, cse_id: str = None):
        """
        Initialize with Google Custom Search API credentials.
        
        Args:
            api_key: Google API key (falls back to env var)
            cse_id: Custom Search Engine ID (falls back to env var)
        """
        self.api_key = api_key or google_search_api_key
        self.cse_id = cse_id or google_cse_id
        
        if not self.api_key:
            raise ValueError(
                "Google API key missing. Set GOOGLE_SEARCH_API_KEY env var "
                "or pass api_key to constructor."
            )
        
        if not self.cse_id:
            raise ValueError(
                "Google CSE ID missing. Set GOOGLE_CSE_ID env var "
                "or pass cse_id to constructor."
            )
        
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.daily_query_count = 0  # Track for cost estimation
        
        logger.info("[Google API] Initialized with CSE ID: %s", self.cse_id[:10] + "...")
    
    def is_link_format_valid(self, link: str) -> bool:
        """Check if link format is valid."""
        if not link:
            return False
        return link.startswith(("http://", "https://"))
    
    def is_link_leads_to_a_website(self, link: str) -> bool:
        """Check if link leads to a website (not a file)."""
        excluded_extensions = [
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', 
            '.xls', '.xlsx', '.zip', '.rar', '.tar', '.gz'
        ]
        lower_link = link.lower()
        return not any(lower_link.endswith(ext) for ext in excluded_extensions)
    
    def _calculate_cost(self, num_queries: int) -> float:
        """
        Calculate the cost based on number of queries.
        First 100 queries per day are free.
        """
        if self.daily_query_count + num_queries <= self.FREE_QUERIES_PER_DAY:
            return 0.0
        
        # Calculate paid queries
        free_remaining = max(0, self.FREE_QUERIES_PER_DAY - self.daily_query_count)
        paid_queries = num_queries - free_remaining
        
        return paid_queries * self.COST_PER_QUERY_AFTER_FREE
    
    def _extract_search_hit(self, item: Dict[str, Any]) -> SearchHit:
        """
        Extract a SearchHit from a Google Custom Search API item.
        """
        link = item.get('link', '')
        title = item.get('title', '')
        
        # Build metadata from various fields
        metadata_parts = []
        
        # Snippet
        if item.get('snippet'):
            metadata_parts.append(item['snippet'])
        
        # Page map data (if available)
        pagemap = item.get('pagemap', {})
        
        # Try to get additional metadata from metatags
        metatags = pagemap.get('metatags', [{}])[0]
        if metatags:
            # Author
            author = metatags.get('author') or metatags.get('article:author')
            if author:
                metadata_parts.append(f"Author: {author}")
            
            # Published date
            pub_date = (metatags.get('article:published_time') or 
                       metatags.get('publishdate') or 
                       metatags.get('og:updated_time'))
            if pub_date:
                metadata_parts.append(f"Date: {pub_date[:10]}")  # Just date part
        
        # CSE image (if available)
        if pagemap.get('cse_image'):
            metadata_parts.append("Has image")
        
        # Display link
        if item.get('displayLink'):
            metadata_parts.append(f"Source: {item['displayLink']}")
        
        # File format (if specified)
        if item.get('fileFormat'):
            metadata_parts.append(f"Format: {item['fileFormat']}")
        
        # Combine metadata
        metadata = " | ".join(metadata_parts)
        
        return SearchHit(
            link=link,
            title=title,
            metadata=metadata
        )
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs
    ) -> SERPMethodOp:
        """
        Perform a Google Custom Search.
        
        Args:
            query: Search query
            num_results: Number of results to fetch (max 100 per search)
            **kwargs: Additional parameters for the API
                - lr: Language restriction (e.g., "lang_en")
                - cr: Country restriction (e.g., "countryUS")
                - dateRestrict: Date restriction (e.g., "d7" for past week)
                - siteSearch: Restrict to specific site
                - fileType: File type filter
                - searchType: "image" for image search
                - sort: Sort order
        
        Returns:
            SERPMethodOp with results and usage info
        """
        start_time = time.time()
        all_items = []
        total_api_calls = 0
        
        # Google CSE API returns max 10 results per call
        results_per_page = 10
        start_index = 1
        
        logger.debug(f"[Google API] Searching for: '{query}', requesting {num_results} results")
        
        while len(all_items) < num_results:
            # Prepare parameters
            params = {
                'key': self.api_key,
                'cx': self.cse_id,
                'q': query,
                'num': min(results_per_page, num_results - len(all_items)),
                'start': start_index
            }
            
            # Add any additional parameters
            params.update(kwargs)
            
            try:
                # Make API request
                logger.debug(f"[Google API] Request {total_api_calls + 1}: start={start_index}")
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                total_api_calls += 1
                
                # Check for errors
                if 'error' in data:
                    logger.error(f"[Google API] Error: {data['error']}")
                    break
                
                # Extract items
                items = data.get('items', [])
                if not items:
                    logger.debug(f"[Google API] No more results at index {start_index}")
                    break
                
                all_items.extend(items)
                
                # Log search information
                search_info = data.get('searchInformation', {})
                if search_info:
                    total_results = search_info.get('totalResults', '0')
                    search_time = search_info.get('searchTime', 0)
                    logger.debug(
                        f"[Google API] Found {total_results} total results "
                        f"in {search_time}s (fetched {len(items)} items)"
                    )
                
                # Check if we have all results
                if len(all_items) >= num_results:
                    break
                
                # Check if there are more pages
                next_page = data.get('queries', {}).get('nextPage', [])
                if not next_page:
                    break
                
                start_index += results_per_page
                
            except requests.exceptions.RequestException as e:
                logger.error(f"[Google API] Request error: {e}")
                break
            except Exception as e:
                logger.exception(f"[Google API] Unexpected error: {e}")
                break
        
        # Convert items to SearchHits
        hits = []
        for item in all_items[:num_results]:  # Limit to requested number
            try:
                link = item.get('link', '')
                if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                    hit = self._extract_search_hit(item)
                    hits.append(hit)
                    logger.debug(f"[Google API] Added: {hit.title[:50]}... | {hit.link}")
            except Exception as e:
                logger.error(f"[Google API] Error processing item: {e}")
                continue
        
        # Calculate cost
        self.daily_query_count += total_api_calls
        cost = self._calculate_cost(total_api_calls)
        
        elapsed = time.time() - start_time
        
        logger.info(
            f"[Google API] Completed: {len(hits)} results from {total_api_calls} API calls "
            f"in {elapsed:.2f}s (cost: ${cost:.4f})"
        )
        
        return SERPMethodOp(
            name="google_api",
            results=hits,
            usage=UsageInfo(cost=cost),
            elapsed_time=elapsed
        )
    
    async def async_search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs
    ) -> SERPMethodOp:
        """
        Async version of search using httpx.
        """
        start_time = time.time()
        all_items = []
        total_api_calls = 0
        
        results_per_page = 10
        start_index = 1
        
        logger.debug(f"[Google API Async] Searching for: '{query}'")
        
        async with httpx.AsyncClient() as client:
            while len(all_items) < num_results:
                params = {
                    'key': self.api_key,
                    'cx': self.cse_id,
                    'q': query,
                    'num': min(results_per_page, num_results - len(all_items)),
                    'start': start_index
                }
                params.update(kwargs)
                
                try:
                    response = await client.get(self.base_url, params=params)
                    response.raise_for_status()
                    
                    data = response.json()
                    total_api_calls += 1
                    
                    if 'error' in data:
                        logger.error(f"[Google API Async] Error: {data['error']}")
                        break
                    
                    items = data.get('items', [])
                    if not items:
                        break
                    
                    all_items.extend(items)
                    
                    if len(all_items) >= num_results:
                        break
                    
                    next_page = data.get('queries', {}).get('nextPage', [])
                    if not next_page:
                        break
                    
                    start_index += results_per_page
                    
                except Exception as e:
                    logger.error(f"[Google API Async] Error: {e}")
                    break
        
        # Process results
        hits = []
        for item in all_items[:num_results]:
            try:
                link = item.get('link', '')
                if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                    hits.append(self._extract_search_hit(item))
            except Exception:
                continue
        
        self.daily_query_count += total_api_calls
        cost = self._calculate_cost(total_api_calls)
        elapsed = time.time() - start_time
        
        logger.info(f"[Google API Async] Completed: {len(hits)} results in {elapsed:.2f}s")
        
        return SERPMethodOp(
            name="google_api_async",
            results=hits,
            usage=UsageInfo(cost=cost),
            elapsed_time=elapsed
        )
    
    def search_images(
        self,
        query: str,
        num_results: int = 10,
        **kwargs
    ) -> SERPMethodOp:
        """
        Search for images using Google Custom Search API.
        """
        kwargs['searchType'] = 'image'
        return self.search(query, num_results, **kwargs)
    
    def search_site(
        self,
        query: str,
        site: str,
        num_results: int = 10,
        **kwargs
    ) -> SERPMethodOp:
        """
        Search within a specific site.
        
        Args:
            query: Search query
            site: Site to search (e.g., "wikipedia.org")
            num_results: Number of results
            **kwargs: Additional parameters
        """
        kwargs['siteSearch'] = site
        return self.search(query, num_results, **kwargs)
    
    def reset_daily_counter(self):
        """Reset the daily query counter (call this at start of new day)."""
        self.daily_query_count = 0
        logger.info("[Google API] Daily query counter reset")


async def _async_demo():
    """Demo async functionality"""
    api = GoogleSearchAPI()
    print("\n--- ASYNC Google API Search ---")
    
    result = await api.async_search("Python web scraping tutorial", num_results=5)
    
    print(f"Found {len(result.results)} results")
    print(f"Cost: ${result.usage.cost:.4f}")
    print(f"Time: {result.elapsed_time:.2f}s")
    print("\nResults:")
    
    for i, hit in enumerate(result.results):
        print(f"\n{i+1}. {hit.title}")
        print(f"   URL: {hit.link}")
        print(f"   Metadata: {hit.metadata}")


def main():
    """Demo the Google Custom Search API"""
    
    print("=== Google Custom Search API Demo ===")
    
    # Debug: Print environment variables
    print(f"Debug - API Key: {'***' + google_search_api_key[-10:] if google_search_api_key else 'None'}")
    print(f"Debug - CSE ID: {google_cse_id if google_cse_id else 'None'}")
    print()
    
    # Initialize API
    api = GoogleSearchAPI()
    
    # Basic search
    print("\n1. Basic Search:")
    result = api.search("Python machine learning", num_results=5)
    
    print(f"Found {len(result.results)} results")
    print(f"{result.results}")
    print(f"Cost: ${result.usage.cost:.4f}")
    print(f"Time: {result.elapsed_time:.2f}s")
    
    # for i, hit in enumerate(result.results):
    #     print(f"\n{i+1}. {hit.title}")
    #     print(f"   URL: {hit.link}")
    #     if hit.metadata:
    #         print(f"   Metadata: {hit.metadata[:100]}...")
    
    # # Site-specific search
    # print("\n\n2. Site-Specific Search (Wikipedia):")
    # wiki_result = api.search_site("artificial intelligence", "wikipedia.org", num_results=3)
    
    # for i, hit in enumerate(wiki_result.results):
    #     print(f"\n{i+1}. {hit.title}")
    #     print(f"   URL: {hit.link}")
    
    # # Image search
    # print("\n\n3. Image Search:")
    # image_result = api.search_images("cute puppies", num_results=3)
    # print(f"Found {len(image_result.results)} image results")
    
    # # Run async demo
    # print("\n\n4. Running async demo...")
    # asyncio.run(_async_demo())


if __name__ == "__main__":
    main()