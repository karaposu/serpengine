# serpengine/google_search_api.py

# to run python -m serpengine.google_search_api


import time
import logging
import requests
import httpx
from typing import List, Dict, Any

from .base_channel import BaseSearchChannel
from .schemes import SearchHit, SerpChannelOp

logger = logging.getLogger(__name__)


class GoogleSearchAPI(BaseSearchChannel):
    """
    Google Custom Search API wrapper.
    Uses the official Google Custom Search JSON API.
    """
    
    # Channel metadata
    CHANNEL_NAME = "google_api"
    REQUIRED_ENV_VARS = ["GOOGLE_SEARCH_API_KEY", "GOOGLE_CSE_ID"]
    DESCRIPTION = "Google Custom Search API (paid after 100 queries/day)"
    
    # Pricing
    FREE_SEARCHES_PER_DAY = 100
    COST_PER_SEARCH = 0.005  # $5 per 1000 queries after free tier
    
    def initialize(self):
        """Initialize Google API specific resources."""
        self.api_key = self.get_credential("GOOGLE_SEARCH_API_KEY")
        self.cse_id = self.get_credential("GOOGLE_CSE_ID")
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.daily_query_count = 0
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs
    ) -> SerpChannelOp:
        """
        Perform a Google Custom Search.
        
        Additional kwargs:
            - lr: Language restriction (e.g., "lang_en")
            - cr: Country restriction (e.g., "countryUS")
            - dateRestrict: Date restriction (e.g., "d7" for past week)
            - siteSearch: Restrict to specific site
            - fileType: File type filter
            - searchType: "image" for image search
            - sort: Sort order
        """
        start_time = time.time()
        all_items = []
        total_api_calls = 0
        
        # Google CSE API returns max 10 results per call
        results_per_page = 10
        start_index = 1
        
        #logger.debug(f"[{self.name}] Searching for: '{query}'")
        
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
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                total_api_calls += 1
                
                # Check for errors
                if 'error' in data:
                    logger.error(f"[{self.name}] API error: {data['error']}")
                    break
                
                # Extract items
                items = data.get('items', [])
                if not items:
                    break
                
                all_items.extend(items)
                
                # Check if there are more pages
                next_page = data.get('queries', {}).get('nextPage', [])
                if not next_page or len(all_items) >= num_results:
                    break
                
                start_index += results_per_page
                
            except Exception as e:
                logger.error(f"[{self.name}] Request error: {e}")
                break
        
        # Convert items to SearchHits
        hits = []
        for item in all_items[:num_results]:
            link = item.get('link', '')
            
            if self.is_link_valid(link) and self.is_link_to_website(link):
                hit = self.create_search_hit(
                    link=link,
                    title=item.get('title', ''),
                    metadata=self._extract_metadata(item)
                )
                hits.append(hit)
        
        # Update daily counter and calculate cost
        self.daily_query_count += total_api_calls
        cost = self.calculate_cost(self.daily_query_count) - self.calculate_cost(self.daily_query_count - total_api_calls)
        
        elapsed = time.time() - start_time
       # logger.info(f"[{self.name}] Found {len(hits)} results in {elapsed:.2f}s (cost: ${cost:.4f})")
        
        return self.create_channel_op(hits, elapsed, cost)
    
    async def async_search(
        self,
        query: str,
        num_results: int = 10,
        **kwargs
    ) -> SerpChannelOp:
        """Async version of search using httpx."""
        start_time = time.time()
        all_items = []
        total_api_calls = 0
        
        results_per_page = 10
        start_index = 1
        
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
                        logger.error(f"[{self.name}] API error: {data['error']}")
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
                    logger.error(f"[{self.name}] Async error: {e}")
                    break
        
        # Process results
        hits = []
        for item in all_items[:num_results]:
            link = item.get('link', '')
            if self.is_link_valid(link) and self.is_link_to_website(link):
                hits.append(self.create_search_hit(
                    link=link,
                    title=item.get('title', ''),
                    metadata=self._extract_metadata(item)
                ))
        
        self.daily_query_count += total_api_calls
        cost = self.calculate_cost(self.daily_query_count) - self.calculate_cost(self.daily_query_count - total_api_calls)
        elapsed = time.time() - start_time
        
        return self.create_channel_op(hits, elapsed, cost)
    
    def _extract_metadata(self, item: Dict[str, Any]) -> str:
        """Extract metadata from a Google Custom Search API item."""
        metadata_parts = []
        
        # Snippet
        if item.get('snippet'):
            metadata_parts.append(item['snippet'])
        
        # Page map data
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
                metadata_parts.append(f"Date: {pub_date[:10]}")
        
        # Display link
        if item.get('displayLink'):
            metadata_parts.append(f"Source: {item['displayLink']}")
        
        return " | ".join(metadata_parts)
    
    def search_images(self, query: str, num_results: int = 10, **kwargs) -> SerpChannelOp:
        """Search for images."""
        kwargs['searchType'] = 'image'
        return self.search(query, num_results, **kwargs)
    
    def search_site(self, query: str, site: str, num_results: int = 10, **kwargs) -> SerpChannelOp:
        """Search within a specific site."""
        kwargs['siteSearch'] = site
        return self.search(query, num_results, **kwargs)
    
    def reset_daily_counter(self):
        """Reset the daily query counter (call at start of new day)."""
        self.daily_query_count = 0
        logger.info(f"[{self.name}] Daily query counter reset")


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