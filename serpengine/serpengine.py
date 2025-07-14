# serpengine.py

# to run python -m serpengine.serpengine

import os, re, time, logging, warnings, asyncio
from typing import List, Dict, Optional, Union
from dataclasses import asdict
from dotenv import load_dotenv

from .channel_manager import ChannelManager, ChannelRegistry
from .schemes import SearchHit, UsageInfo, SerpChannelOp, SerpEngineOp, ContextAwareSearchRequestObject

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
        num_of_links_per_channel: int                  = 10,
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
            num_of_links_per_channel: Number of results per channel
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
        channel_ops = self._run_search_channels(
            query, num_of_links_per_channel, sources,
            allowed_countries, forbidden_countries,
            allowed_domains, forbidden_domains,
            validation_conditions,
            boolean_llm_filter_semantic
        )
        
        # Aggregate results
        top_op = self._aggregate(channel_ops, start_time)
        
        # Format output
        return self._format(top_op, output_format)
    
    async def collect_async(
        self,
        query: str,
        regex_based_link_validation: bool             = True,
        allow_links_forwarding_to_files: bool          = True,
        keyword_match_based_link_validation: List[str] = None,
        num_of_links_per_channel: int                  = 10,
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
        channel_ops = await self._run_search_channels_async(
            query, num_of_links_per_channel, sources,
            allowed_countries, forbidden_countries,
            allowed_domains, forbidden_domains,
            validation_conditions,
            boolean_llm_filter_semantic
        )
        
        # Aggregate results
        top_op = self._aggregate(channel_ops, start_time)
        
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
    
    def _run_search_channels(
        self,
        query: str,
        num_of_links_per_channel: int,
        sources: List[str],
        allowed_countries: List[str],
        forbidden_countries: List[str],
        allowed_domains: List[str],
        forbidden_domains: List[str],
        validation_conditions: Dict,
        boolean_llm_filter_semantic: bool
    ) -> List[SerpChannelOp]:
        """Run search on each channel synchronously."""
        ops = []
        
        for channel_name in sources:
            try:
                # Execute search through channel manager
                # Channel will handle ranking automatically
                op = self.channel_manager.execute_search(
                    channel_name, query, num_of_links_per_channel
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
    
    async def _run_search_channels_async(
        self,
        query: str,
        num_of_links_per_channel: int,
        sources: List[str],
        allowed_countries: List[str],
        forbidden_countries: List[str],
        allowed_domains: List[str],
        forbidden_domains: List[str],
        validation_conditions: Dict,
        boolean_llm_filter_semantic: bool
    ) -> List[SerpChannelOp]:
        """Run search on each channel asynchronously."""
        # Create async tasks
        tasks = []
        for channel_name in sources:
            task = self.channel_manager.execute_search_async(
                channel_name, query, num_of_links_per_channel
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
        channel_ops: List[SerpChannelOp],
        start_time: float
    ) -> SerpEngineOp:
        """Aggregate multiple channel operations into one result."""
        all_hits = []
        total_cost = 0.0
        
        for op in channel_ops:
            all_hits.extend(op.results)
            total_cost += op.usage.cost
        
        return SerpEngineOp(
            usage=UsageInfo(cost=total_cost),
            channels=channel_ops,
            results=all_hits,
            elapsed_time=time.time() - start_time
        )
    
    @staticmethod
    def _format(top_op: SerpEngineOp, output_format: str):
        """Format output as JSON or object."""
        if output_format == "json":
            return {
                "usage": asdict(top_op.usage),
                "channels": [asdict(c) for c in top_op.channels],
                "results": [asdict(h) for h in top_op.results],
                "elapsed_time": top_op.elapsed_time
            }
        elif output_format == "object":
            return top_op
        else:
            raise ValueError("output_format must be 'json' or 'object'")


def main():
    """Demo the SERPEngine with base channel system."""
    print("=== SERPEngine Demo ===\n")
    
  
    
   
    print("\n1. Initializing with specific channels...")
    try:
        serp = SERPEngine(channels=["google_api", "serpapi"])
        print(f"   Initialized: {serp.available_channels}")
    except ValueError as e:
        print(f"   Error: {e}")
    
    # 3. Run a search
    if 'serp' in locals() and serp.available_channels:
        print("\n3. Running search...")
        
        result = serp.collect(
            query="Python web scraping",
            num_of_links_per_channel=3,
            output_format="object"
        )
        
        print(f"   Total results: {len(result.results)}")
        print(f"   Total cost: ${result.usage.cost:.4f}")
        print(f"   Time: {result.elapsed_time:.2f}s")
        
        # Show channel breakdown
        print("\n   Channel breakdown:")
        for channel in result.channels:
            print(f"   {channel.name}: {len(channel.results)} results, "
                  f"${channel.usage.cost:.4f}, {channel.elapsed_time:.2f}s")
        
        # Show first few results
        if result.results:
            print("\n   Sample results:")
            for i, hit in enumerate(result.results[:5]):
                print(f"\n   {i+1}. {hit.title}")
                print(f"      URL: {hit.link}")
                print(f"      Channel: {hit.channel_name} (rank #{hit.channel_rank})")
                if hit.metadata:
                    print(f"      Metadata: {hit.metadata[:100]}...")
        
        # Show results by channel
        print("\n   Results by channel:")
        for channel, hits in result.results_by_channel().items():
            print(f"   {channel}: {len(hits)} results")
            if hits:
                print(f"      Top result: {hits[0].title}")
    
    # 4. Test filters
    print("\n\n4. Testing with filters...")
    if 'serp' in locals() and serp.available_channels:
        filtered_result = serp.collect(
            query="Python tutorial",
            num_of_links_per_channel=10,
            allowed_domains=["python.org", "github.com"],
            regex_based_link_validation=True,
            allow_links_forwarding_to_files=False
        )
        
        print(f"   Filtered results: {len(filtered_result.results)}")
        print("   Domains found:")
        domains = set()
        for hit in filtered_result.results:
            domain = hit.link.split('/')[2] if '/' in hit.link else hit.link
            domains.add(domain)
        for domain in domains:
            print(f"      - {domain}")
    
    # 5. Test async search
    print("\n5. Testing async search...")
    
    async def test_async():
        try:
            serp = SERPEngine()
            result = await serp.collect_async(
                query="machine learning",
                num_of_links_per_channel=5
            )
            print(f"   Async search: {len(result.results)} results in {result.elapsed_time:.2f}s")
            
            # Show concurrent channel execution
            print("   Channels executed concurrently:")
            for channel in result.channels:
                print(f"      {channel.name}: {channel.elapsed_time:.2f}s")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    asyncio.run(test_async())
    
    # 6. Test JSON output format
    print("\n6. Testing JSON output format...")
    if 'serp' in locals() and serp.available_channels:
        json_result = serp.collect(
            query="data science",
            num_of_links_per_channel=2,
            output_format="json"
        )
        
        print("   JSON keys:", list(json_result.keys()))
        print(f"   Total results in JSON: {len(json_result['results'])}")
        print(f"   Channels in JSON: {[c['name'] for c in json_result['channels']]}")


if __name__ == "__main__":
    main()