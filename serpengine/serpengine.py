# serpengine.py

# to run python -m serpengine.serpengine

import os, re, time, logging, warnings, asyncio
from typing import List, Dict, Optional, Union
from dataclasses import asdict
from dotenv import load_dotenv

from .channel_manager import ChannelManager, ChannelRegistry
from .schemes import SearchHit, UsageInfo, SERPMethodOp, SerpEngineOp, ContextAwareSearchRequestObject

# ─── Setup ─────────────────────────────────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*found in sys.modules after import of package.*"
)

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class SERPEngine:
    """Main search orchestration engine."""
    
    def __init__(
        self,
        channels: List[str] = None,
        credentials: Dict[str, str] = None,
        auto_check_env: bool = True
    ):
        """
        Initialize SERPEngine with specified channels.
        
        Args:
            channels: List of channel names to initialize. If None, all available channels
                     with valid credentials will be initialized.
            credentials: Optional dict of credentials to override environment variables
            auto_check_env: If True, automatically check for required env vars
        """
        # Initialize channel manager
        self.channel_manager = ChannelManager(credentials)
        
        # Initialize channels
        self.available_channels = self.channel_manager.initialize_channels(
            channels, auto_check_env
        )
        
        if not self.available_channels:
            raise ValueError(
                "No search channels could be initialized. "
                "Please check your credentials and channel configuration."
            )
        
        logger.info(f"Successfully initialized channels: {self.available_channels}")
    
    def list_channels(self) -> Dict[str, Dict]:
        """List all available channels and their status."""
        return self.channel_manager.list_channels_status()
    
    def context_aware_collect(
        self,
        input: ContextAwareSearchRequestObject,
        **kwargs
    ) -> Union[Dict, SerpEngineOp]:
        """Context-aware search entry point."""
        return self.collect(query=input.query, **kwargs)
    
    def collect(
        self,
        query: str,
        regex_based_link_validation: bool             = True,
        allow_links_forwarding_to_files: bool          = True,
        keyword_match_based_link_validation: List[str] = None,
        num_urls: int                                  = 10,
        search_sources: List[str]                      = None,
        allowed_countries: List[str]                   = None,
        forbidden_countries: List[str]                 = None,
        allowed_domains: List[str]                     = None,
        forbidden_domains: List[str]                   = None,
        boolean_llm_filter_semantic: bool              = False,
        output_format: str                             = "object"
    ) -> Union[Dict, SerpEngineOp]:
        """
        Perform synchronous search across channels.
        
        Args:
            query: Search query
            search_sources: List of channels to use. If None, uses all available.
            num_urls: Number of results per channel
            output_format: "object" or "json"
            ... (other filter parameters)
            
        Returns:
            SerpEngineOp object or JSON dict
        """
        start_time = time.time()
        
        # Determine which channels to use
        sources = self._get_sources(search_sources)
        
        # Prepare validation conditions
        validation_conditions = {
            "regex_validation_enabled": regex_based_link_validation,
            "allow_file_links": allow_links_forwarding_to_files,
            "keyword_match_list": keyword_match_based_link_validation
        }
        
        # Run searches
        method_ops = self._run_search_methods(
            query, num_urls, sources,
            allowed_countries, forbidden_countries,
            allowed_domains, forbidden_domains,
            validation_conditions,
            boolean_llm_filter_semantic
        )
        
        # Aggregate results
        top_op = self._aggregate(method_ops, start_time)
        
        # Format output
        return self._format(top_op, output_format)
    
    async def collect_async(
        self,
        query: str,
        regex_based_link_validation: bool             = True,
        allow_links_forwarding_to_files: bool          = True,
        keyword_match_based_link_validation: List[str] = None,
        num_urls: int                                  = 10,
        search_sources: List[str]                      = None,
        allowed_countries: List[str]                   = None,
        forbidden_countries: List[str]                 = None,
        allowed_domains: List[str]                     = None,
        forbidden_domains: List[str]                   = None,
        boolean_llm_filter_semantic: bool              = False,
        output_format: str                             = "object"
    ) -> Union[Dict, SerpEngineOp]:
        """
        Perform async search across channels concurrently.
        """
        start_time = time.time()
        
        # Determine which channels to use
        sources = self._get_sources(search_sources)
        
        # Prepare validation conditions
        validation_conditions = {
            "regex_validation_enabled": regex_based_link_validation,
            "allow_file_links": allow_links_forwarding_to_files,
            "keyword_match_list": keyword_match_based_link_validation
        }
        
        # Run async searches
        method_ops = await self._run_search_methods_async(
            query, num_urls, sources,
            allowed_countries, forbidden_countries,
            allowed_domains, forbidden_domains,
            validation_conditions,
            boolean_llm_filter_semantic
        )
        
        # Aggregate results
        top_op = self._aggregate(method_ops, start_time)
        
        # Format output
        return self._format(top_op, output_format)
    
    def _get_sources(self, search_sources: Optional[List[str]]) -> List[str]:
        """Determine which channels to use for search."""
        if search_sources is None:
            return self.available_channels
        
        # Filter to only available channels
        sources = []
        for src in search_sources:
            if src in self.available_channels:
                sources.append(src)
            else:
                logger.warning(f"Requested channel '{src}' not available")
        
        return sources
    
    def _run_search_methods(
        self,
        query: str,
        num_urls: int,
        sources: List[str],
        allowed_countries: List[str],
        forbidden_countries: List[str],
        allowed_domains: List[str],
        forbidden_domains: List[str],
        validation_conditions: Dict,
        boolean_llm_filter_semantic: bool
    ) -> List[SERPMethodOp]:
        """Run search on each channel synchronously."""
        ops = []
        
        for channel_name in sources:
            try:
                # Execute search through channel manager
                op = self.channel_manager.execute_search(
                    channel_name, query, num_urls
                )
                
                # Apply filters
                op.results = self._apply_filters(
                    op.results,
                    allowed_countries, forbidden_countries,
                    allowed_domains, forbidden_domains,
                    validation_conditions
                )
                
                # Optional LLM filter
                if boolean_llm_filter_semantic:
                    op.results = self._filter_with_llm(op.results)
                
                ops.append(op)
                logger.info(f"Channel '{channel_name}' returned {len(op.results)} results")
                
            except Exception as e:
                logger.exception(f"Error running channel '{channel_name}': {e}")
        
        return ops
    
    async def _run_search_methods_async(
        self,
        query: str,
        num_urls: int,
        sources: List[str],
        allowed_countries: List[str],
        forbidden_countries: List[str],
        allowed_domains: List[str],
        forbidden_domains: List[str],
        validation_conditions: Dict,
        boolean_llm_filter_semantic: bool
    ) -> List[SERPMethodOp]:
        """Run search on each channel asynchronously."""
        # Create async tasks
        tasks = []
        for channel_name in sources:
            task = self.channel_manager.execute_search_async(
                channel_name, query, num_urls
            )
            tasks.append(task)
        
        # Run all tasks concurrently
        raw_ops = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_ops = []
        for i, op in enumerate(raw_ops):
            if isinstance(op, Exception):
                logger.exception(f"Async search failed", exc_info=op)
                continue
            
            # Apply filters
            op.results = self._apply_filters(
                op.results,
                allowed_countries, forbidden_countries,
                allowed_domains, forbidden_domains,
                validation_conditions
            )
            
            if boolean_llm_filter_semantic:
                op.results = self._filter_with_llm(op.results)
            
            processed_ops.append(op)
        
        return processed_ops
    
    def _apply_filters(
        self,
        results: List[SearchHit],
        allowed_countries: List[str],
        forbidden_countries: List[str],
        allowed_domains: List[str],
        forbidden_domains: List[str],
        validation_conditions: Dict
    ) -> List[SearchHit]:
        """Apply various filters to search results."""
        filtered = []
        
        for hit in results:
            link = hit.link
            
            # Domain filters
            if allowed_domains and not any(d in link.lower() for d in allowed_domains):
                continue
            if forbidden_domains and any(d in link.lower() for d in forbidden_domains):
                continue
            
            # Regex validation
            if validation_conditions.get("regex_validation_enabled"):
                pattern = r"^https?://([\w-]+\.)+[\w-]+(/[\w\-./?%&=]*)?$"
                if not re.match(pattern, link):
                    continue
            
            # File type filter
            if not validation_conditions.get("allow_file_links", True):
                file_extensions = (".pdf", ".doc", ".xls", ".zip", ".ppt")
                if any(link.lower().endswith(ext) for ext in file_extensions):
                    continue
            
            # Keyword matching
            keywords = validation_conditions.get("keyword_match_list") or []
            if keywords:
                combined = f"{hit.link} {hit.title} {hit.metadata}".lower()
                if not any(kw.lower() in combined for kw in keywords):
                    continue
            
            filtered.append(hit)
        
        return filtered
    
    def _filter_with_llm(self, hits: List[SearchHit]) -> List[SearchHit]:
        """Apply LLM-based semantic filtering."""
        try:
            from .myllmservice import MyLLMService
            svc = MyLLMService()
        except ImportError:
            logger.warning("LLM service not available, skipping semantic filter")
            return hits
        
        filtered = []
        for hit in hits:
            try:
                resp = svc.filter_simple(
                    semantic_filter_text=True,
                    string_data=f"{hit.title} {hit.metadata}"
                )
                if getattr(resp, "success", False):
                    filtered.append(hit)
            except Exception:
                logger.exception(f"LLM-filter failed on {hit.link}")
        
        return filtered
    
    def _aggregate(
        self,
        method_ops: List[SERPMethodOp],
        start_time: float
    ) -> SerpEngineOp:
        """Aggregate multiple method operations into one result."""
        all_hits = []
        total_cost = 0.0
        
        for op in method_ops:
            all_hits.extend(op.results)
            total_cost += op.usage.cost
        
        return SerpEngineOp(
            usage=UsageInfo(cost=total_cost),
            methods=method_ops,
            results=all_hits,
            elapsed_time=time.time() - start_time
        )
    
    @staticmethod
    def _format(top_op: SerpEngineOp, output_format: str):
        """Format output as JSON or object."""
        if output_format == "json":
            return {
                "usage": asdict(top_op.usage),
                "methods": [asdict(m) for m in top_op.methods],
                "results": [asdict(h) for h in top_op.results],
                "elapsed_time": top_op.elapsed_time
            }
        elif output_format == "object":
            return top_op
        else:
            raise ValueError("output_format must be 'json' or 'object'")


def main():
    """Demo the refactored SERPEngine."""
    print("=== Refactored SERPEngine Demo ===\n")
    
    # 1. Check available channels
    print("1. Checking available channels...")
    try:
        serp = SERPEngine(channels=[])
        channel_info = serp.list_channels()
        
        for name, info in channel_info.items():
            status = "✓ Ready" if info["initialized"] else f"✗ Missing: {info['missing_env']}"
            print(f"   {name}: {status}")
    except ValueError:
        pass
    
    # 2. Initialize with specific channels
    print("\n2. Initializing with specific channels...")
    try:
        serp = SERPEngine(channels=["google_scraper", "serpapi"])
        print(f"   Initialized: {serp.available_channels}")
    except ValueError as e:
        print(f"   Error: {e}")
    
    # 3. Run a search
    if 'serp' in locals() and serp.available_channels:
        print("\n3. Running search...")
        
        result = serp.collect(
            query="Python web scraping",
            num_urls=3,
            output_format="object"
        )
        
        print(f"   Total results: {len(result.results)}")
        print(f"   Total cost: ${result.usage.cost:.4f}")
        print(f"   Time: {result.elapsed_time:.2f}s")
        
        # Show first result
        if result.results:
            hit = result.results[0]
            print(f"\n   First result:")
            print(f"   Title: {hit.title}")
            print(f"   URL: {hit.link}")
    
    # 4. Test async search
    print("\n4. Testing async search...")
    
    async def test_async():
        try:
            serp = SERPEngine()
            result = await serp.collect_async(
                query="machine learning",
                num_urls=5
            )
            print(f"   Async search: {len(result.results)} results in {result.elapsed_time:.2f}s")
        except Exception as e:
            print(f"   Error: {e}")
    
    asyncio.run(test_async())


if __name__ == "__main__":
    main()