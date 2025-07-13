# dataforseo_search.py

import os
import logging
import time
import asyncio
import requests
import httpx
from typing import List, Optional, Dict, Any
from base64 import b64encode
from dotenv import load_dotenv

from .schemes import SearchHit, UsageInfo, SerpChannelOp

load_dotenv()
logger = logging.getLogger(__name__)

# Get credentials from environment
dataforseo_username = os.getenv("DATAFORSEO_USERNAME")
dataforseo_password = os.getenv("DATAFORSEO_PASSWORD")


class DataForSEOSearcher:
    """
    Wrapper for DataForSEO Google SERP API.
    Implements both Live and Standard methods.
    """
    
    BASE_URL = "https://api.dataforseo.com/v3"
    LIVE_COST_PER_REQUEST = 0.002  # $0.002 per SERP (100 results)
    
    def __init__(self, username: str = None, password: str = None):
        """
        Initialize with DataForSEO credentials.
        Falls back to environment variables if not provided.
        """
        self.username = username or dataforseo_username
        self.password = password or dataforseo_password
        
        if not self.username or not self.password:
            raise ValueError(
                "DataForSEO credentials missing. Set DATAFORSEO_USERNAME and "
                "DATAFORSEO_PASSWORD env vars or pass them to constructor."
            )
        
        # Prepare auth header
        credentials = b64encode(f"{self.username}:{self.password}".encode()).decode()
        self.headers = {
            'Authorization': f'Basic {credentials}',
            'Content-Type': 'application/json'
        }
    
    def is_link_format_valid(self, link: str) -> bool:
        """Check if link format is valid."""
        if not link:
            return False
        return link.startswith(("http://", "https://"))
    
    def is_link_leads_to_a_website(self, link: str) -> bool:
        """Check if link leads to a website (not a file)."""
        excluded_extensions = ['.pdf', '.doc', '.docx', '.ppt', 
                              '.pptx', '.xls', '.xlsx', '.zip']
        lower_link = link.lower()
        return not any(lower_link.endswith(ext) for ext in excluded_extensions)
    
    def _extract_search_hits(self, items: List[Dict[str, Any]]) -> List[SearchHit]:
        """
        Extract SearchHit objects from DataForSEO response items.
        Handles various SERP element types.
        """
        hits = []
        
        for item in items:
            item_type = item.get('type', '')
            
            # Process organic results
            if item_type == 'organic':
                link = item.get('url', '')
                title = item.get('title', '')
                description = item.get('description', '')
                
                if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                    hits.append(SearchHit(
                        link=link,
                        title=title,
                        metadata=description
                    ))
            
            # Process paid results (ads)
            elif item_type == 'paid':
                link = item.get('url', '')
                title = item.get('title', '')
                description = item.get('description', '')
                
                if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                    hits.append(SearchHit(
                        link=link,
                        title=f"[Ad] {title}",
                        metadata=description
                    ))
            
            # Process featured snippets
            elif item_type == 'featured_snippet':
                link = item.get('url', '')
                title = item.get('title', '')
                description = item.get('description', '')
                
                if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                    hits.append(SearchHit(
                        link=link,
                        title=f"[Featured] {title}",
                        metadata=description
                    ))
            
            # Process People Also Ask
            elif item_type == 'people_also_ask':
                paa_items = item.get('items', [])
                for paa in paa_items:
                    expanded = paa.get('expanded_element', [])
                    for exp in expanded:
                        link = exp.get('url', '')
                        title = paa.get('title', '')
                        description = exp.get('description', '')
                        
                        if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                            hits.append(SearchHit(
                                link=link,
                                title=f"[PAA] {title}",
                                metadata=description
                            ))
            
            # Process related results within organic results
            if item_type == 'organic' and 'related_result' in item:
                related = item.get('related_result', [])
                for rel in related:
                    link = rel.get('url', '')
                    title = rel.get('title', '')
                    description = rel.get('description', '')
                    
                    if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                        hits.append(SearchHit(
                            link=link,
                            title=title,
                            metadata=description
                        ))
        
        return hits
    
    def search_live(
        self,
        query: str,
        location_name: str = "United States",
        language_code: str = "en",
        num_results: int = 100,
        device: str = "desktop"
    ) -> SerpChannelOp:
        """
        Use DataForSEO Live API for instant results.
        
        Args:
            query: Search query
            location_name: Location for search (e.g., "United States")
            language_code: Language code (e.g., "en")
            num_results: Number of results to fetch (max 700)
            device: "desktop" or "mobile"
        
        Returns:
            SerpChannelOp with results and usage info
        """
        start = time.time()
        url = f"{self.BASE_URL}/serp/google/organic/live/advanced"
        
        # Prepare request data
        data = [{
            "keyword": query,
            "location_name": location_name,
            "language_code": language_code,
            "depth": min(num_results, 700),  # Max 700
            "device": device
        }]
        
        logger.debug(f"[DataForSEO Live] query='{query}', num_results={num_results}")
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Check for API errors
            if result.get('status_code') != 20000:
                logger.error(f"[DataForSEO] API error: {result.get('status_message')}")
                return SerpChannelOp(
                    name="dataforseo_live",
                    results=[],
                    usage=UsageInfo(cost=0.0),
                    elapsed_time=time.time() - start
                )
            
            # Extract results from the first task
            hits = []
            if result.get('tasks') and len(result['tasks']) > 0:
                task = result['tasks'][0]
                if task.get('result') and len(task['result']) > 0:
                    items = task['result'][0].get('items', [])
                    hits = self._extract_search_hits(items)
            
            # Calculate actual cost
            cost = self.LIVE_COST_PER_REQUEST
            if num_results > 100:
                # Additional cost for results beyond 100
                cost *= (num_results / 100)
            
            elapsed = time.time() - start
            logger.info(f"[DataForSEO Live] Returning {len(hits)} hits in {elapsed:.2f}s")
            
            return SerpChannelOp(
                name="dataforseo_live",
                results=hits,
                usage=UsageInfo(cost=cost),
                elapsed_time=elapsed
            )
            
        except Exception as e:
            logger.exception(f"[DataForSEO] Error in search_live: {e}")
            return SerpChannelOp(
                name="dataforseo_live",
                results=[],
                usage=UsageInfo(cost=0.0),
                elapsed_time=time.time() - start
            )
    
    async def async_search_live(
        self,
        query: str,
        location_name: str = "United States",
        language_code: str = "en",
        num_results: int = 100,
        device: str = "desktop"
    ) -> SerpChannelOp:
        """
        Async version of search_live using httpx.
        """
        start = time.time()
        url = f"{self.BASE_URL}/serp/google/organic/live/advanced"
        
        data = [{
            "keyword": query,
            "location_name": location_name,
            "language_code": language_code,
            "depth": min(num_results, 700),
            "device": device
        }]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=self.headers, json=data)
                response.raise_for_status()
                
                result = response.json()
                
                if result.get('status_code') != 20000:
                    logger.error(f"[DataForSEO Async] API error: {result.get('status_message')}")
                    return SerpChannelOp(
                        name="dataforseo_live_async",
                        results=[],
                        usage=UsageInfo(cost=0.0),
                        elapsed_time=time.time() - start
                    )
                
                hits = []
                if result.get('tasks') and len(result['tasks']) > 0:
                    task = result['tasks'][0]
                    if task.get('result') and len(task['result']) > 0:
                        items = task['result'][0].get('items', [])
                        hits = self._extract_search_hits(items)
                
                cost = self.LIVE_COST_PER_REQUEST
                if num_results > 100:
                    cost *= (num_results / 100)
                
                elapsed = time.time() - start
                logger.info(f"[DataForSEO Async] Returning {len(hits)} hits in {elapsed:.2f}s")
                
                return SerpChannelOp(
                    name="dataforseo_live_async",
                    results=hits,
                    usage=UsageInfo(cost=cost),
                    elapsed_time=elapsed
                )
                
        except Exception as e:
            logger.exception(f"[DataForSEO Async] Error: {e}")
            return SerpChannelOp(
                name="dataforseo_live_async",
                results=[],
                usage=UsageInfo(cost=0.0),
                elapsed_time=time.time() - start
            )
    
    def search_with_filters(
        self,
        query: str,
        location_name: str = "United States",
        language_code: str = "en",
        num_results: int = 100,
        target_domain: Optional[str] = None,
        search_param: Optional[str] = None
    ) -> SerpChannelOp:
        """
        Search with additional filters like target domain or search parameters.
        
        Args:
            query: Search query
            location_name: Location for search
            language_code: Language code
            num_results: Number of results
            target_domain: Filter results by domain (e.g., "example.com")
            search_param: Additional search parameters (e.g., "&tbs=qdr:d" for past day)
        
        Returns:
            SerpChannelOp with filtered results
        """
        start = time.time()
        url = f"{self.BASE_URL}/serp/google/organic/live/advanced"
        
        data = [{
            "keyword": query,
            "location_name": location_name,
            "language_code": language_code,
            "depth": min(num_results, 700),
            "device": "desktop"
        }]
        
        # Add optional filters
        if target_domain:
            data[0]["target"] = target_domain
        if search_param:
            data[0]["search_param"] = search_param
        
        logger.debug(f"[DataForSEO Filtered] query='{query}', target='{target_domain}'")
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('status_code') != 20000:
                logger.error(f"[DataForSEO] API error: {result.get('status_message')}")
                return SerpChannelOp(
                    name="dataforseo_filtered",
                    results=[],
                    usage=UsageInfo(cost=0.0),
                    elapsed_time=time.time() - start
                )
            
            hits = []
            if result.get('tasks') and len(result['tasks']) > 0:
                task = result['tasks'][0]
                if task.get('result') and len(task['result']) > 0:
                    items = task['result'][0].get('items', [])
                    hits = self._extract_search_hits(items)
            
            cost = self.LIVE_COST_PER_REQUEST
            if num_results > 100:
                cost *= (num_results / 100)
            
            elapsed = time.time() - start
            logger.info(f"[DataForSEO Filtered] Returning {len(hits)} hits in {elapsed:.2f}s")
            
            return SerpChannelOp(
                name="dataforseo_filtered",
                results=hits,
                usage=UsageInfo(cost=cost),
                elapsed_time=elapsed
            )
            
        except Exception as e:
            logger.exception(f"[DataForSEO] Error in search_with_filters: {e}")
            return SerpChannelOp(
                name="dataforseo_filtered",
                results=[],
                usage=UsageInfo(cost=0.0),
                elapsed_time=time.time() - start
            )


async def _async_demo():
    """Demo async functionality"""
    searcher = DataForSEOSearcher()
    print("\n--- ASYNC DataForSEO Live API ---")
    result = await searcher.async_search_live("machine learning", num_results=10)
    print(f"Found {len(result.results)} results")
    print(f"Cost: ${result.usage.cost:.4f}")
    print(f"Time: {result.elapsed_time:.2f}s")
    
    for i, hit in enumerate(result.results[:3]):
        print(f"\n{i+1}. {hit.title}")
        print(f"   {hit.link}")
        print(f"   {hit.metadata[:100]}...")


def main():
    """Demo the DataForSEO wrapper"""
    searcher = DataForSEOSearcher()
    
    # Example 1: Basic search
    print("\n--- DataForSEO Live Search ---")
    result = searcher.search_live("artificial intelligence", num_results=10)
    print(f"Found {len(result.results)} results")
    print(f"Cost: ${result.usage.cost:.4f}")
    print(f"Time: {result.elapsed_time:.2f}s")
    
    # Example 2: Filtered search
    print("\n--- DataForSEO Filtered Search ---")
    filtered = searcher.search_with_filters(
        "python programming",
        target_domain="github.com",
        num_results=10
    )
    print(f"Found {len(filtered.results)} results from github.com")
    
    # Example 3: Async search
    print("\n--- Running async demo ---")
    asyncio.run(_async_demo())


if __name__ == "__main__":
    main()