#here is serpengine.py

# to run python -m serpengine.serpengine



import os, re, time, logging, warnings, asyncio
from typing import List, Dict, Optional, Union
from dataclasses import asdict
from dotenv import load_dotenv

from .google_searcher import GoogleSearcher
from .myllmservice import MyLLMService
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

_env_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
_env_cse_id    = os.getenv("GOOGLE_CSE_ID")



@staticmethod
def _format(top_op: SerpEngineOp, output_format: str):
    if output_format == "json":
        return {
            "usage":        asdict(top_op.usage),
            "methods":      [asdict(m) for m in top_op.methods],
            "results":      [asdict(h) for h in top_op.results],
            "elapsed_time": top_op.elapsed_time
        }
    elif output_format == "object":
        return top_op
    else:
        raise ValueError("output_format must be 'json' or 'object'")
    



class SERPEngine:
    def __init__(
        self,
        GOOGLE_SEARCH_API_KEY: Optional[str] = None,
        GOOGLE_CSE_ID: Optional[str]        = None
    ):
        # Validate API key
        key = GOOGLE_SEARCH_API_KEY or _env_api_key
        if not key:
            raise ValueError(
                "Missing environment variable 'GOOGLE_SEARCH_API_KEY'."
            )
        self.google_api_key = key

        # Validate CSE ID
        cx = GOOGLE_CSE_ID or _env_cse_id
        if not cx:
            raise ValueError(
                "Missing environment variable 'GOOGLE_CSE_ID'."
            )
        self.google_cse_id = cx

        self.searcher = GoogleSearcher()

    
    def context_aware_collect(
        self,
        input: ContextAwareSearchRequestObject,
        regex_based_link_validation: bool               = True,
        allow_links_forwarding_to_files: bool            = True,
        keyword_match_based_link_validation: List[str]   = None,
        num_urls: int                                    = 10,
        search_sources: List[str]                        = None,
        allowed_countries: List[str]                     = None,
        forbidden_countries: List[str]                   = None,
        allowed_domains: List[str]                       = None,
        forbidden_domains: List[str]                     = None,
        boolean_llm_filter_semantic: bool                = False,
        # output_format: str                               = "json"
        output_format                               = "object"
        
        
    ) -> Union[Dict, SerpEngineOp]:
        """
        Top-level entry: run each method, then aggregate into one SerpEngineOp.
        """
        start_time = time.time()
        sources = search_sources or [
            "google_search_via_api",
            "google_search_via_request_module"
        ]
        validation_conditions = {
            "regex_validation_enabled": regex_based_link_validation,
            "allow_file_links": allow_links_forwarding_to_files,
            "keyword_match_list": keyword_match_based_link_validation
        }

        # 1) Run each search source
        method_ops = self._run_search_methods(
            query,
            num_urls,
            sources,
            allowed_countries,
            forbidden_countries,
            allowed_domains,
            forbidden_domains,
            validation_conditions,
            boolean_llm_filter_semantic
        )

        # 2) Aggregate into a top-level operation
        top_op = self._aggregate(method_ops, start_time)

        if output_format == "json":
            return {
                "usage":       asdict(top_op.usage),
                "methods":     [asdict(m) for m in top_op.methods],
                "results":     [asdict(hit) for hit in top_op.results],
                "elapsed_time": top_op.elapsed_time,
            }
        elif output_format == "object":
            return top_op
        else:
            raise ValueError("Unsupported output_format. Use 'json' or 'object'.")
        

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
        output_format                                  = "object"
    ) -> Union[Dict, SerpEngineOp]:
        """
        Blocking (synchronous) entry point.
        Runs selected search sources, applies filters, aggregates into SerpEngineOp,
        then formats as JSON or returns the object itself.
        """
        start_time = time.time()

        # default sources if none provided
        sources = search_sources or [
            "google_search_via_api",
            "google_search_via_request_module"
        ]

        validation_conditions = {
            "regex_validation_enabled": regex_based_link_validation,
            "allow_file_links":        allow_links_forwarding_to_files,
            "keyword_match_list":      keyword_match_based_link_validation
        }

        # 1️⃣ run each source (synchronous path)
        method_ops = self._run_search_methods(
            query, num_urls, sources,
            allowed_countries, forbidden_countries,
            allowed_domains,  forbidden_domains,
            validation_conditions,
            boolean_llm_filter_semantic
        )

        # 2️⃣ aggregate results into a top-level operation
        top_op = self._aggregate(method_ops, start_time)

        # 3️⃣ format according to output_format ('json' or 'object')
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
        output_format                                  = "object"
    ) -> Union[Dict, SerpEngineOp]:
        """
        Non-blocking version: runs all requested search sources concurrently.
        """
        start_time = time.time()
        sources = search_sources or ["google_search_via_api",
                                     "google_search_via_request_module"]

        validation_conditions = {
            "regex_validation_enabled": regex_based_link_validation,
            "allow_file_links":        allow_links_forwarding_to_files,
            "keyword_match_list":      keyword_match_based_link_validation
        }

        # 1️⃣ run each source (async)
        method_ops = await self._run_search_methods_async(
            query, num_urls, sources,
            allowed_countries, forbidden_countries,
            allowed_domains,  forbidden_domains,
            validation_conditions,
            boolean_llm_filter_semantic
        )

        # 2️⃣ aggregate
        top_op = self._aggregate(method_ops, start_time)
        return self._format(top_op, output_format)


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
        """
        Calls each named source (API or scrape), applies filters & optional LLM,
        returns a list of SERPMethodOp.
        """
        ops: List[SERPMethodOp] = []

        for source in sources:
            try:
                if source == "google_search_via_api":
                    op = self.searcher.search_with_api(
                        query=query,
                        num_results=num_urls,
                        google_search_api_key=self.google_api_key,
                        cse_id=self.google_cse_id
                    )
                elif source == "google_search_via_request_module":
                    op = self.searcher.search(query)
                else:
                    logger.warning(f"Ignoring unknown source '{source}'")
                    continue

                # filter hits
                op.results = self._apply_filters(
                    results=op.results,
                    allowed_countries=allowed_countries,
                    forbidden_countries=forbidden_countries,
                    allowed_domains=allowed_domains,
                    forbidden_domains=forbidden_domains,
                    validation_conditions=validation_conditions
                )

                # optional semantic LLM filter
                if boolean_llm_filter_semantic:
                    op.results = self._filter_with_llm(op.results)

                ops.append(op)

            except Exception as e:
                logger.exception(f"Error running '{source}': {e}")

        return ops
    

    
    # ------------------------------------------------------------------ #
    #  ❸  NEW async runner                                               #
    # ------------------------------------------------------------------ #
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
        """
        Launches all requested search sources concurrently via asyncio.gather().
        """
        async_tasks = []

        for source in sources:
            if source == "google_search_via_api":
                async_tasks.append(
                    self.searcher.async_search_with_api(
                        query, num_urls,
                        google_search_api_key=self.google_api_key,
                        cse_id=self.google_cse_id
                    )
                )
            elif source == "google_search_via_request_module":
                async_tasks.append(self.searcher.async_search(query))
            else:
                logger.warning(f"Ignoring unknown source '{source}'")

        # run them concurrently
        raw_ops: List[SERPMethodOp] = await asyncio.gather(*async_tasks, return_exceptions=True)

        # post-process (filters, LLM) just like sync path
        processed_ops: List[SERPMethodOp] = []
        for op in raw_ops:
            if isinstance(op, Exception):
                logger.exception("Async search method raised", exc_info=op)
                continue

            op.results = self._apply_filters(
                results=op.results,
                allowed_countries=allowed_countries,
                forbidden_countries=forbidden_countries,
                allowed_domains=allowed_domains,
                forbidden_domains=forbidden_domains,
                validation_conditions=validation_conditions
            )

            if boolean_llm_filter_semantic:
                op.results = self._filter_with_llm(op.results)

            processed_ops.append(op)

        return processed_ops
    

    
    @staticmethod
    def _format(top_op: SerpEngineOp, output_format: str):
        if output_format == "json":
            return {
                "usage":        asdict(top_op.usage),
                "methods":      [asdict(m) for m in top_op.methods],
                "results":      [asdict(h) for h in top_op.results],
                "elapsed_time": top_op.elapsed_time
            }
        elif output_format == "object":
            return top_op
        else:
            raise ValueError("output_format must be 'json' or 'object'")

    def _aggregate(
        self,
        method_ops: List[SERPMethodOp],
        start_time: float
    ) -> SerpEngineOp:
        """
        Combines multiple SERPMethodOp into one SerpEngineOp,
        summing costs and concatenating all hits.
        """
        all_hits = []
        total_cost = 0.0

        for m in method_ops:
            all_hits.extend(m.results)
            total_cost += m.usage.cost

        elapsed = time.time() - start_time
        top_op = SerpEngineOp(
            usage=UsageInfo(cost=total_cost),
            methods=method_ops,
            results=all_hits,
            elapsed_time=elapsed
        )
        return top_op

    # ─── Filtering Helpers ──────────────────────────────────────────────────────

    def _apply_filters(
        self,
        results: List[SearchHit],
        allowed_countries: List[str],
        forbidden_countries: List[str],
        allowed_domains: List[str],
        forbidden_domains: List[str],
        validation_conditions: Dict
    ) -> List[SearchHit]:
        out = []
        for hit in results:
            link = hit.link

            if allowed_domains and not any(d in link.lower() for d in allowed_domains):
                continue
            if forbidden_domains and any(d in link.lower() for d in forbidden_domains):
                continue

            if validation_conditions.get("regex_validation_enabled"):
                pattern = r"^https?://([\w-]+\.)+[\w-]+(/[\w\-./?%&=]*)?$"
                if not re.match(pattern, link):
                    continue

            if not validation_conditions.get("allow_file_links", True):
                if any(link.lower().endswith(ext)
                       for ext in (".pdf", ".doc", ".xls", ".zip", ".ppt")):
                    continue

            kws = validation_conditions.get("keyword_match_list") or []
            if kws:
                combined = f"{hit.link} {hit.title} {hit.metadata}".lower()
                if not any(kw.lower() in combined for kw in kws):
                    continue

            out.append(hit)
        return out

    def _filter_with_llm(self, hits: List[SearchHit]) -> List[SearchHit]:
        svc = MyLLMService()
        kept = []
        for hit in hits:
            try:
                resp = svc.filter_simple(
                    semantic_filter_text=True,
                    string_data=f"{hit.title} {hit.metadata}"
                )
                if getattr(resp, "success", False):
                    kept.append(hit)
            except Exception:
                logger.exception(f"LLM-filter failed on {hit.link}")
        return kept

    



if __name__ == "__main__":
    serp_engine = SERPEngine()


    query="FÇ TEKSTİL SANAYİ VE DIŞ TİCARET ANONİM ŞİRKETİ"
    # query= "best food in USA"
    # query="3850763999"

    
    serp_engine_op = serp_engine.collect(
        query=query,
        num_urls=5,
        search_sources=["google_search_via_api"],
        regex_based_link_validation=False,             
        allow_links_forwarding_to_files=False,        
        output_format="object"  # or "json"
    )
    print(serp_engine_op)

    print(serp_engine_op.all_links())
    # for l in result_data:
    #     print(l)results



    # print(result_data.all_links())
