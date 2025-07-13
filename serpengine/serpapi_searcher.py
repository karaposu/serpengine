# serpapi_searcher.py

# python -m serpengine.serpapi_searcher

import os
import logging
import time
import asyncio
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

try:
    from serpapi import GoogleSearch
except ImportError:
    raise ImportError("Please install serpapi package: pip install google-search-results")

from .schemes import SearchHit, UsageInfo, SerpChannelOp

load_dotenv()
logger = logging.getLogger(__name__)

# Get API key from environment
serpapi_api_key = os.getenv("SERPAPI_API_KEY")


class SerpApiSearcher:
    """
    Wrapper for SerpApi Google Search.
    Uses the official serpapi package.
    """
    
    # SerpApi pricing plans
    PLAN_COSTS = {
        "free": 0.0,  # Free plan
        "developer": 75 / 5000,  # $75 for 5,000 searches = $0.015 per search
        "production": 150 / 15000,  # $75 for 5,000 searches = $0.010 per search
        "big_data": 275 / 30000,  # $275 for 5,000 searches = $0.00916 per search
        "searcher": 725 / 100000   # $725 for 5,000 searches = $0.00725 per search
    }
    
    def __init__(self, api_key: str = None, plan: str = "developer"):
        """
        Initialize with SerpApi API key and plan.
        
        Args:
            api_key: SerpApi API key (falls back to env var)
            plan: Pricing plan - "free", "developer", "big_data", or "searcher"
                  Defaults to "developer" plan
        """
        self.api_key = api_key or serpapi_api_key
        
        if not self.api_key:
            raise ValueError(
                "SerpApi API key missing. Set SERPAPI_API_KEY env var "
                "or pass api_key to constructor."
            )
        
        # Validate and set plan
        if plan not in self.PLAN_COSTS:
            raise ValueError(
                f"Invalid plan '{plan}'. Must be one of: {list(self.PLAN_COSTS.keys())}"
            )
        
        self.plan = plan
        self.cost_per_request = self.PLAN_COSTS[plan]
        
        logger.info(f"[SerpApi] Initialized with '{plan}' plan (${self.cost_per_request:.4f}/search)")
    
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
    
    def _build_metadata(self, result: Dict[str, Any]) -> str:
        """
        Build comprehensive metadata string from various result fields.
        """
        metadata_parts = []
        
        # Primary snippet
        if result.get('snippet'):
            metadata_parts.append(result['snippet'])
        
        # Date if available
        if result.get('date'):
            metadata_parts.append(f"Date: {result['date']}")
        
        # Rating and reviews if available
        rich_snippet = result.get('rich_snippet', {})
        if rich_snippet:
            top_data = rich_snippet.get('top', {})
            detected_ext = top_data.get('detected_extensions', {})
            if detected_ext:
                rating = detected_ext.get('rating')
                reviews = detected_ext.get('reviews')
                if rating and reviews:
                    metadata_parts.append(f"Rating: {rating}/5 ({reviews} reviews)")
        
        # Source domain
        if result.get('source'):
            metadata_parts.append(f"Source: {result['source']}")
        
        # Missing terms (if any)
        if result.get('missing'):
            metadata_parts.append(f"Missing terms: {', '.join(result['missing'])}")
        
        # Displayed link for context
        if result.get('displayed_link'):
            metadata_parts.append(f"URL: {result['displayed_link']}")
        
        return " | ".join(metadata_parts)
    
    def _extract_search_hits(self, results: Dict[str, Any]) -> List[SearchHit]:
        """
        Extract SearchHit objects from SerpApi response with enhanced metadata.
        """
        hits = []
        
        # Extract organic results
        organic_results = results.get('organic_results', [])
        for result in organic_results:
            link = result.get('link', '')
            title = result.get('title', '')
            
            if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                # Build comprehensive metadata
                metadata = self._build_metadata(result)
                
                hits.append(SearchHit(
                    link=link,
                    title=title,
                    metadata=metadata
                ))
                
                logger.debug(f"[SerpApi] Extracted: {title[:50]}... | {link}")
        
        # Extract ads (if included)
        ads = results.get('ads', [])
        for ad in ads:
            link = ad.get('link', '')
            title = ad.get('title', '')
            
            if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                # Build ad metadata
                ad['snippet'] = ad.get('description', '')  # Normalize field name
                metadata = self._build_metadata(ad)
                
                hits.append(SearchHit(
                    link=link,
                    title=f"[Ad] {title}",
                    metadata=metadata
                ))
        
        # Extract featured snippet/answer box
        answer_box = results.get('answer_box', {})
        if answer_box:
            # Handle different answer box types
            if answer_box.get('type') == 'organic_result':
                link = answer_box.get('link', '')
                title = answer_box.get('title', '')
                
                if link and self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                    metadata = self._build_metadata(answer_box)
                    
                    hits.append(SearchHit(
                        link=link,
                        title=f"[Featured] {title}",
                        metadata=metadata
                    ))
            
            # Handle direct answer types
            elif answer_box.get('answer'):
                # Some answer boxes don't have links but contain useful info
                answer_text = answer_box.get('answer', '')
                title = answer_box.get('title', 'Direct Answer')
                source = answer_box.get('source', {})
                link = source.get('link', '')
                
                if link and self.is_link_format_valid(link):
                    hits.append(SearchHit(
                        link=link,
                        title=f"[Answer] {title}",
                        metadata=answer_text
                    ))
        
        # Extract People Also Ask
        related_questions = results.get('related_questions', [])
        for question in related_questions:
            link = question.get('link', '')
            title = question.get('question', '')
            snippet = question.get('snippet', '')
            
            if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                # Build PAA metadata
                metadata_parts = []
                if snippet:
                    metadata_parts.append(snippet)
                if question.get('title'):
                    metadata_parts.append(f"Page: {question['title']}")
                
                hits.append(SearchHit(
                    link=link,
                    title=f"[PAA] {title}",
                    metadata=" | ".join(metadata_parts) if metadata_parts else ""
                ))
        
        # Extract knowledge graph results
        knowledge_graph = results.get('knowledge_graph', {})
        if knowledge_graph:
            # Main knowledge graph entity
            kg_link = knowledge_graph.get('website', '') or knowledge_graph.get('source', {}).get('link', '')
            kg_title = knowledge_graph.get('title', '')
            kg_description = knowledge_graph.get('description', '')
            
            if kg_link and self.is_link_format_valid(kg_link) and self.is_link_leads_to_a_website(kg_link):
                # Build rich KG metadata
                metadata_parts = []
                if kg_description:
                    metadata_parts.append(kg_description)
                
                # Add additional KG info
                if knowledge_graph.get('type'):
                    metadata_parts.append(f"Type: {knowledge_graph['type']}")
                
                # Add key facts if available
                facts = []
                for key in ['founded', 'headquarters', 'ceo', 'employees', 'revenue']:
                    if knowledge_graph.get(key):
                        facts.append(f"{key.title()}: {knowledge_graph[key]}")
                
                if facts:
                    metadata_parts.append(" | ".join(facts))
                
                hits.append(SearchHit(
                    link=kg_link,
                    title=f"[Knowledge Graph] {kg_title}",
                    metadata=" | ".join(metadata_parts)
                ))
            
            # Knowledge graph articles
            kg_articles = knowledge_graph.get('articles', [])
            for article in kg_articles:
                link = article.get('link', '')
                title = article.get('title', '')
                snippet = article.get('snippet', '')
                
                if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                    metadata = snippet or ""
                    if article.get('date'):
                        metadata += f" | Date: {article['date']}"
                    
                    hits.append(SearchHit(
                        link=link,
                        title=title,
                        metadata=metadata
                    ))
        
        # Extract shopping results if present
        shopping_results = results.get('shopping_results', [])
        for item in shopping_results:
            link = item.get('link', '')
            title = item.get('title', '')
            
            if self.is_link_format_valid(link) and self.is_link_leads_to_a_website(link):
                # Build shopping metadata
                metadata_parts = []
                if item.get('price'):
                    metadata_parts.append(f"Price: {item['price']}")
                if item.get('source'):
                    metadata_parts.append(f"Seller: {item['source']}")
                if item.get('rating'):
                    metadata_parts.append(f"Rating: {item['rating']}")
                if item.get('reviews'):
                    metadata_parts.append(f"Reviews: {item['reviews']}")
                
                hits.append(SearchHit(
                    link=link,
                    title=f"[Shopping] {title}",
                    metadata=" | ".join(metadata_parts)
                ))
        
        # Log extraction summary
        logger.info(f"[SerpApi] Extracted {len(hits)} total results from response")
        
        return hits
    
    def search(
        self,
        query: str,
        location: str = "United States",
        num_results: int = 10,
        language: str = "en",
        country: str = "us"
    ) -> SerpChannelOp:
        """
        Perform a Google search using SerpApi.
        
        Args:
            query: Search query
            location: Location for search (e.g., "Austin, Texas, United States")
            num_results: Number of results to fetch (SerpApi returns up to 100)
            language: Language code (e.g., "en")
            country: Country code for Google domain (e.g., "us")
        
        Returns:
            SerpChannelOp with results and usage info
        """
        start = time.time()
        
        # Prepare parameters
        params = {
            "api_key": self.api_key,
            "engine": "google",
            "q": query,
            "location": location,
            "google_domain": f"google.{country}" if country != "us" else "google.com",
            "gl": country,
            "hl": language,
            "num": min(num_results, 100)  # SerpApi max is 100 per request
        }
        
        logger.debug(f"[SerpApi] query='{query}', location='{location}', num={num_results}")
        
        try:
            # Perform search
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Check for errors
            if "error" in results:
                logger.error(f"[SerpApi] Error: {results['error']}")
                return SerpChannelOp(
                    name="serpapi",
                    results=[],
                    usage=UsageInfo(cost=0.0),
                    elapsed_time=time.time() - start
                )
            
            # Log search metadata
            search_info = results.get('search_information', {})
            if search_info:
                total_results = search_info.get('total_results', 0)
                time_taken = search_info.get('time_taken_displayed', 0)
                logger.info(f"[SerpApi] Found {total_results} total results in {time_taken}s")
            
            # Extract search hits with enhanced metadata
            hits = self._extract_search_hits(results)
            
            # Calculate cost based on plan
            cost = self.cost_per_request
            
            elapsed = time.time() - start
            logger.info(f"[SerpApi] Returning {len(hits)} hits in {elapsed:.2f}s (cost: ${cost:.4f})")
            
            return SerpChannelOp(
                name="serpapi",
                results=hits,
                usage=UsageInfo(cost=cost),
                elapsed_time=elapsed
            )
            
        except Exception as e:
            logger.exception(f"[SerpApi] Error in search: {e}")
            return SerpChannelOp(
                name="serpapi",
                results=[],
                usage=UsageInfo(cost=0.0),
                elapsed_time=time.time() - start
            )
    
    def search_with_params(
        self,
        query: str,
        custom_params: Dict[str, Any]
    ) -> SerpChannelOp:
        """
        Search with custom parameters for advanced use cases.
        
        Args:
            query: Search query
            custom_params: Custom parameters to pass to SerpApi
                          (will override defaults except api_key and query)
        
        Returns:
            SerpChannelOp with results
        """
        start = time.time()
        
        # Base parameters
        params = {
            "api_key": self.api_key,
            "engine": "google",
            "q": query
        }
        
        # Merge custom parameters (but don't override api_key or query)
        for key, value in custom_params.items():
            if key not in ["api_key", "q"]:
                params[key] = value
        
        logger.debug(f"[SerpApi Custom] query='{query}', params={custom_params}")
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "error" in results:
                logger.error(f"[SerpApi] Error: {results['error']}")
                return SerpChannelOp(
                    name="serpapi_custom",
                    results=[],
                    usage=UsageInfo(cost=0.0),
                    elapsed_time=time.time() - start
                )
            
            hits = self._extract_search_hits(results)
            cost = self.cost_per_request
            
            elapsed = time.time() - start
            logger.info(f"[SerpApi Custom] Returning {len(hits)} hits in {elapsed:.2f}s")
            
            return SerpChannelOp(
                name="serpapi_custom",
                results=hits,
                usage=UsageInfo(cost=cost),
                elapsed_time=elapsed
            )
            
        except Exception as e:
            logger.exception(f"[SerpApi] Error in search_with_params: {e}")
            return SerpChannelOp(
                name="serpapi_custom",
                results=[],
                usage=UsageInfo(cost=0.0),
                elapsed_time=time.time() - start
            )
    
    async def async_search(
        self,
        query: str,
        location: str = "United States",
        num_results: int = 10,
        language: str = "en",
        country: str = "us"
    ) -> SerpChannelOp:
        """
        Async wrapper for search using ThreadPoolExecutor.
        (SerpApi doesn't provide native async support)
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                self.search,
                query,
                location,
                num_results,
                language,
                country
            )
        # Update the name to indicate it was async
        result.name = "serpapi_async"
        return result
    
    def search_with_pagination(
        self,
        query: str,
        total_results: int = 100,
        location: str = "United States",
        language: str = "en",
        country: str = "us"
    ) -> SerpChannelOp:
        """
        Search with pagination to get more than 100 results.
        SerpApi limits to 100 results per query, so we paginate.
        
        Args:
            query: Search query
            total_results: Total number of results desired
            location: Location for search
            language: Language code
            country: Country code
        
        Returns:
            SerpChannelOp with all paginated results
        """
        start = time.time()
        all_hits = []
        total_cost = 0.0
        
        # Calculate number of pages needed
        results_per_page = 100
        num_pages = (total_results + results_per_page - 1) // results_per_page
        
        for page in range(num_pages):
            start_index = page * results_per_page
            
            params = {
                "api_key": self.api_key,
                "engine": "google",
                "q": query,
                "location": location,
                "google_domain": f"google.{country}" if country != "us" else "google.com",
                "gl": country,
                "hl": language,
                "num": min(results_per_page, total_results - start_index),
                "start": start_index
            }
            
            logger.debug(f"[SerpApi Pagination] Page {page + 1}/{num_pages}, start={start_index}")
            
            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                
                if "error" in results:
                    logger.error(f"[SerpApi] Error on page {page + 1}: {results['error']}")
                    break
                
                page_hits = self._extract_search_hits(results)
                all_hits.extend(page_hits)
                total_cost += self.cost_per_request
                
                # Check if we have enough results
                if len(all_hits) >= total_results:
                    break
                
                # Check if there are no more results
                if not results.get('organic_results'):
                    break
                    
            except Exception as e:
                logger.exception(f"[SerpApi] Error on page {page + 1}: {e}")
                break
        
        elapsed = time.time() - start
        logger.info(f"[SerpApi Pagination] Returning {len(all_hits)} hits in {elapsed:.2f}s")
        
        return SerpChannelOp(
            name="serpapi_paginated",
            results=all_hits[:total_results],  # Trim to requested number
            usage=UsageInfo(cost=total_cost),
            elapsed_time=elapsed
        )


async def _async_demo():
    """Demo async functionality"""
    searcher = SerpApiSearcher()
    print("\n--- ASYNC SerpApi Search ---")
    result = await searcher.async_search("artificial intelligence", num_results=10)
    print(f"Found {len(result.results)} results")
    print(f"Cost: ${result.usage.cost:.4f}")
    print(f"Time: {result.elapsed_time:.2f}s")
    
    for i, hit in enumerate(result.results[:3]):
        print(f"\n{i+1}. {hit.title}")
        print(f"   URL: {hit.link}")
        print(f"   Metadata: {hit.metadata[:150]}...")


def main():
    """Demo the SerpApi wrapper with enhanced metadata extraction"""
    
    print("\n=== SerpAPI Enhanced Demo ===")
    
    # Initialize with free plan for testing
    serpapi_searcher = SerpApiSearcher(plan="free")
    
    # Test search
    # query = "MOTORCU HAYRİ MOTORSİKLET VE BİSİKLET SANAYİ VE TİCARET LİMİTED ŞİRKETİ"
    query = "Enes Kuzucu"
    print(f"\nSearching for: {query}")
    print("-" * 80)
    
    request_result = serpapi_searcher.search(
        query,
        location="Istanbul, Turkey",
        num_results=10
    )
    
    print(f"\nFound {len(request_result.results)} results")
    print(f"Cost: ${request_result.usage.cost:.4f}")
    print(f"Time: {request_result.elapsed_time:.2f}s")
    print("\nDetailed Results:")
    print("=" * 80)
    
    for i, hit in enumerate(request_result.results):
        print(f"\n{i+1}. TITLE: {hit.title}")
        print(f"   URL: {hit.link}")
        print(f"   METADATA: {hit.metadata}")
        print("-" * 80)
    
    # Run async demo
    # print("\n\nRunning async demo...")
    # asyncio.run(_async_demo())


if __name__ == "__main__":
    main()