# serpengine/serpapi_searcher.py

# python -m serpengine.serpapi_searcher

import time
import logging
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

try:
    from serpapi import GoogleSearch
except ImportError:
    raise ImportError("Please install serpapi package: pip install google-search-results")

from .base_channel import BaseSearchChannel
from .schemes import SearchHit, SerpChannelOp

logger = logging.getLogger(__name__)


class SerpApiSearcher(BaseSearchChannel):
    """
    Wrapper for SerpApi Google Search.
    Uses the official serpapi package.
    """
    
    # ─── Channel Metadata ─────────────────────────────────────────────────
    
    CHANNEL_NAME = "serpapi"
    REQUIRED_ENV_VARS = ["SERPAPI_API_KEY"]
    DESCRIPTION = "SerpAPI service - Google search results API"
    
    # ─── Pricing Information ──────────────────────────────────────────────
    
    # SerpApi pricing plans - can be overridden in __init__
    PLAN_COSTS = {
        "free": 0.0,  # Free plan
        "developer": 75 / 5000,  # $75 for 5,000 searches = $0.015 per search
        "production": 150 / 15000,  # $150 for 15,000 searches = $0.010 per search
        "big_data": 275 / 30000,  # $275 for 30,000 searches = $0.00916 per search
        "searcher": 725 / 100000   # $725 for 100,000 searches = $0.00725 per search
    }
    
    # Default to developer plan
    COST_PER_SEARCH = PLAN_COSTS["developer"]
    
    def __init__(self, credentials: Dict[str, str] = None, plan: str = "developer"):
        """
        Initialize with SerpApi credentials and plan.
        
        Args:
            credentials: Optional dict to override environment variables
            plan: Pricing plan - "free", "developer", "production", "big_data", or "searcher"
                  Defaults to "developer" plan
        """
        # Validate and set plan before calling parent init
        if plan not in self.PLAN_COSTS:
            raise ValueError(
                f"Invalid plan '{plan}'. Must be one of: {list(self.PLAN_COSTS.keys())}"
            )
        
        self.plan = plan
        self.COST_PER_SEARCH = self.PLAN_COSTS[plan]
        
        # Call parent init which will call initialize()
        super().__init__(credentials)
    
    def initialize(self):
        """Initialize SerpApi specific resources."""
        self.api_key = self.get_credential("SERPAPI_API_KEY")
        
       # logger.info(f"[{self.name}] Using '{self.plan}' plan (${self.COST_PER_SEARCH:.4f}/search)")
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        location: str = "United States",
        language: str = "en",
        country: str = "us",
        **kwargs
    ) -> SerpChannelOp:
        """
        Perform a Google search using SerpApi.
        
        Args:
            query: Search query
            num_results: Number of results to fetch (SerpApi returns up to 100)
            location: Location for search (e.g., "Austin, Texas, United States")
            language: Language code (e.g., "en")
            country: Country code for Google domain (e.g., "us")
            **kwargs: Additional SerpApi parameters
        
        Returns:
            SerpChannelOp with results and usage info
        """
        start_time = time.time()
        
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
        
        # Add any additional parameters from kwargs
        params.update(kwargs)
        
        #logger.debug(f"[{self.name}] query='{query}', location='{location}', num={num_results}")
        
        try:
            # Perform search
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Check for errors
            if "error" in results:
                logger.error(f"[{self.name}] Error: {results['error']}")
                return self.create_channel_op([], time.time() - start_time)
            
            # Log search metadata
            search_info = results.get('search_information', {})
            if search_info:
                total_results = search_info.get('total_results', 0)
                time_taken = search_info.get('time_taken_displayed', 0)
              #  logger.info(f"[{self.name}] Found {total_results} total results in {time_taken}s")
            
            # Extract search hits
            hits = self._extract_search_hits(results)
            
            elapsed = time.time() - start_time
           # logger.info(f"[{self.name}] Returning {len(hits)} hits in {elapsed:.2f}s")
            
            return self.create_channel_op(hits, elapsed)
            
        except Exception as e:
            logger.exception(f"[{self.name}] Error in search: {e}")
            return self.create_channel_op([], time.time() - start_time)
    
    async def async_search(
        self,
        query: str,
        num_results: int = 10,
        location: str = "United States",
        language: str = "en",
        country: str = "us",
        **kwargs
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
                num_results,
                location,
                language,
                country,
                **kwargs
            )
        return result
    
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
            
            if self.is_link_valid(link) and self.is_link_to_website(link):
                # Build comprehensive metadata
                metadata = self._build_metadata(result)
                
                hit = self.create_search_hit(
                    link=link,
                    title=title,
                    metadata=metadata
                )
                hits.append(hit)
                
               # logger.debug(f"[{self.name}] Extracted: {title[:50]}... | {link}")
        
        # Extract other result types
        hits.extend(self._extract_ads(results))
        hits.extend(self._extract_featured_snippet(results))
        hits.extend(self._extract_people_also_ask(results))
        hits.extend(self._extract_knowledge_graph(results))
        hits.extend(self._extract_shopping_results(results))
        
      
        
        return hits
    
    def _build_metadata(self, result: Dict[str, Any]) -> str:
        """Build comprehensive metadata string from various result fields."""
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
    
    def _extract_ads(self, results: Dict[str, Any]) -> List[SearchHit]:
        """Extract ad results."""
        hits = []
        ads = results.get('ads', [])
        
        for ad in ads:
            link = ad.get('link', '')
            title = ad.get('title', '')
            
            if self.is_link_valid(link) and self.is_link_to_website(link):
                # Build ad metadata
                ad['snippet'] = ad.get('description', '')  # Normalize field name
                metadata = self._build_metadata(ad)
                
                hit = self.create_search_hit(
                    link=link,
                    title=f"[Ad] {title}",
                    metadata=metadata
                )
                hits.append(hit)
        
        return hits
    
    def _extract_featured_snippet(self, results: Dict[str, Any]) -> List[SearchHit]:
        """Extract featured snippet/answer box."""
        hits = []
        answer_box = results.get('answer_box', {})
        
        if not answer_box:
            return hits
        
        # Handle different answer box types
        if answer_box.get('type') == 'organic_result':
            link = answer_box.get('link', '')
            title = answer_box.get('title', '')
            
            if link and self.is_link_valid(link) and self.is_link_to_website(link):
                metadata = self._build_metadata(answer_box)
                
                hit = self.create_search_hit(
                    link=link,
                    title=f"[Featured] {title}",
                    metadata=metadata
                )
                hits.append(hit)
        
        # Handle direct answer types
        elif answer_box.get('answer'):
            answer_text = answer_box.get('answer', '')
            title = answer_box.get('title', 'Direct Answer')
            source = answer_box.get('source', {})
            link = source.get('link', '')
            
            if link and self.is_link_valid(link):
                hit = self.create_search_hit(
                    link=link,
                    title=f"[Answer] {title}",
                    metadata=answer_text
                )
                hits.append(hit)
        
        return hits
    
    def _extract_people_also_ask(self, results: Dict[str, Any]) -> List[SearchHit]:
        """Extract People Also Ask results."""
        hits = []
        related_questions = results.get('related_questions', [])
        
        for question in related_questions:
            link = question.get('link', '')
            title = question.get('question', '')
            snippet = question.get('snippet', '')
            
            if self.is_link_valid(link) and self.is_link_to_website(link):
                # Build PAA metadata
                metadata_parts = []
                if snippet:
                    metadata_parts.append(snippet)
                if question.get('title'):
                    metadata_parts.append(f"Page: {question['title']}")
                
                hit = self.create_search_hit(
                    link=link,
                    title=f"[PAA] {title}",
                    metadata=" | ".join(metadata_parts) if metadata_parts else ""
                )
                hits.append(hit)
        
        return hits
    
    def _extract_knowledge_graph(self, results: Dict[str, Any]) -> List[SearchHit]:
        """Extract knowledge graph results."""
        hits = []
        knowledge_graph = results.get('knowledge_graph', {})
        
        if not knowledge_graph:
            return hits
        
        # Main knowledge graph entity
        kg_link = knowledge_graph.get('website', '') or knowledge_graph.get('source', {}).get('link', '')
        kg_title = knowledge_graph.get('title', '')
        kg_description = knowledge_graph.get('description', '')
        
        if kg_link and self.is_link_valid(kg_link) and self.is_link_to_website(kg_link):
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
            
            hit = self.create_search_hit(
                link=kg_link,
                title=f"[Knowledge Graph] {kg_title}",
                metadata=" | ".join(metadata_parts)
            )
            hits.append(hit)
        
        # Knowledge graph articles
        kg_articles = knowledge_graph.get('articles', [])
        for article in kg_articles:
            link = article.get('link', '')
            title = article.get('title', '')
            snippet = article.get('snippet', '')
            
            if self.is_link_valid(link) and self.is_link_to_website(link):
                metadata = snippet or ""
                if article.get('date'):
                    metadata += f" | Date: {article['date']}"
                
                hit = self.create_search_hit(
                    link=link,
                    title=title,
                    metadata=metadata
                )
                hits.append(hit)
        
        return hits
    
    def _extract_shopping_results(self, results: Dict[str, Any]) -> List[SearchHit]:
        """Extract shopping results if present."""
        hits = []
        shopping_results = results.get('shopping_results', [])
        
        for item in shopping_results:
            link = item.get('link', '')
            title = item.get('title', '')
            
            if self.is_link_valid(link) and self.is_link_to_website(link):
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
                
                hit = self.create_search_hit(
                    link=link,
                    title=f"[Shopping] {title}",
                    metadata=" | ".join(metadata_parts)
                )
                hits.append(hit)
        
        return hits
    
    # ─── Additional Methods ───────────────────────────────────────────────
    
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
        # Remove api_key if present in custom params
        safe_params = {k: v for k, v in custom_params.items() if k not in ["api_key", "q"]}
        
        # Use the main search method with custom params
        return self.search(query, **safe_params)
    
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
        
        Args:
            query: Search query
            total_results: Total number of results desired
            location: Location for search
            language: Language code
            country: Country code
        
        Returns:
            SerpChannelOp with all paginated results
        """
        start_time = time.time()
        all_hits = []
        total_api_calls = 0
        
        # Calculate number of pages needed
        results_per_page = 100
        num_pages = (total_results + results_per_page - 1) // results_per_page
        
        for page in range(num_pages):
            start_index = page * results_per_page
            
            # Use main search method with start parameter
            op = self.search(
                query,
                num_results=min(results_per_page, total_results - start_index),
                location=location,
                language=language,
                country=country,
                start=start_index
            )
            
            all_hits.extend(op.results)
            total_api_calls += 1
            
            # Check if we have enough results
            if len(all_hits) >= total_results:
                break
            
            # Check if there are no more results
            if not op.results:
                break
        
        elapsed = time.time() - start_time
        cost = self.calculate_cost(total_api_calls)
        
        logger.info(f"[{self.name} Pagination] Returning {len(all_hits)} hits in {elapsed:.2f}s")
        
        return self.create_channel_op(all_hits[:total_results], elapsed, cost)


# ─── Demo Functions ───────────────────────────────────────────────────────

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
    """Demo the refactored SerpApi wrapper"""
    
    print("\n=== SerpAPI Refactored Demo ===")
    
    # Initialize with free plan for testing
    serpapi_searcher = SerpApiSearcher(plan="free")
    
    # Show channel info
    print(f"\nChannel: {serpapi_searcher.name}")
    print(f"Description: {serpapi_searcher.DESCRIPTION}")
    print(f"Required vars: {serpapi_searcher.REQUIRED_ENV_VARS}")
    print(f"Plan: {serpapi_searcher.plan} (${serpapi_searcher.COST_PER_SEARCH:.4f}/search)")
    
    # Test search
    query = "Python programming"
    print(f"\nSearching for: {query}")
    print("-" * 80)
    
    result = serpapi_searcher.search(
        query,
        location="United States",
        num_results=6
    )
    
    print(f"\nFound {len(result.results)} results")
    print(f"Cost: ${result.usage.cost:.4f}")
    print(f"Time: {result.elapsed_time:.2f}s")
    print("\nResults:")
    print("=" * 80)
    
    for i, hit in enumerate(result.results):
        print(f"\n{i+1}. TITLE: {hit.title}")
        print(f"   URL: {hit.link}")
        print(f"   CHANNEL: {hit.channel_name}")
        print(f"   METADATA: {hit.metadata}")
        print(f"   channel_rank: {hit.channel_rank}")
        print("-" * 80)
    
    # # Run async demo
    # print("\n\nRunning async demo...")
    # asyncio.run(_async_demo())


if __name__ == "__main__":
    main()

