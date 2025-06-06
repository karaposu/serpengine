#here is serpengine.py

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules after import of package.*")


import os
import re
import time
import logging
import requests
from typing import List, Dict, Union, Optional
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

# Local imports (assuming these modules exist in your project)
from .google_searcher import GoogleSearcher
from .schemes import LinkSearch, OperationResult
from .myllmservice import MyLLMService

load_dotenv()
logging.basicConfig(level=logging.INFO)


google_search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
cse_id = os.getenv("GOOGLE_CSE_ID")



class SERPEngine:
    def __init__(self, GOOGLE_SEARCH_API_KEY=None, GOOGLE_CSE_ID=None):
        if not GOOGLE_SEARCH_API_KEY:
            if not google_search_api_key:
                raise ValueError(
                    "Missing environment variable 'GOOGLE_SEARCH_API_KEY'. "
                    "Please set it in your .env or system environment."
                )
            else:
                self.google_api_key = google_search_api_key
        else:
            self.google_api_key = GOOGLE_SEARCH_API_KEY


        if  not GOOGLE_CSE_ID:
            if not cse_id:
                raise ValueError(
                    "Missing environment variable 'GOOGLE_CSE_ID'. "
                    "Please set it in your .env or system environment."
                )
            else:
                 self.google_cse_id = cse_id
        else:
            self.google_cse_id = GOOGLE_CSE_ID
        
        self.google_api_key = google_search_api_key
        self.google_cse_id = cse_id
        # self.logger = logging.getLogger(self.__class__.__name__)
        self.searcher = GoogleSearcher()  # Our integrated GoogleSearcher

    def collect(self,
                query: str,
                # New parameters:
                regex_based_link_validation: bool = True,
                allow_links_forwarding_to_files: bool = True,
                keyword_match_based_link_validation: Optional[List[str]] = None,
                # Existing parameters:
                num_urls: int = 10,
                search_sources: List[str] = None,
                allowed_countries: List[str] = None,
                forbidden_countries: List[str] = None,
                allowed_domains: List[str] = None,
                forbidden_domains: List[str] = None,
                boolean_llm_filter_semantic: bool = False,
                output_format: str = "json"
                ) -> Union[Dict, List[LinkSearch]]:
        """
        Collect links based on the input query and filters.

        :param query: The search query.
        :param regex_based_link_validation: If True, apply regex-based link validation logic.
        :param allow_links_forwarding_to_files: If False, exclude links that appear to be file downloads.
        :param keyword_match_based_link_validation: A list of keywords that must appear in the link/title/metadata.
        :param num_urls: Number of links to retrieve.
        :param search_sources: e.g. ["google_search_via_api", "google_search_via_request_module"].
        :param allowed_countries: List of country codes to allow (placeholder logic).
        :param forbidden_countries: List of country codes to forbid (placeholder logic).
        :param allowed_domains: List of domains to allow.
        :param forbidden_domains: List of domains to block.
        :param boolean_llm_filter_semantic: If True, apply an LLM-based semantic filter.
        :param output_format: "json" or "linksearch".
        """
        

        # logger.debug(f"link collection started")
        start_time = time.time()
        errors: List[str] = []

        if search_sources is None:
            search_sources = ["google_search_via_api", "google_search_via_request_module"]

        linksearch_results: List[LinkSearch] = []

        # 1. Gather results from each source via helper methods
        try:
            if "google_search_via_api" in search_sources:
                api_results = self._fetch_links_from_api(query, num_urls)
                
 
                logger.debug(f"api_results:")
                for i,e in enumerate(api_results):
                    logger.debug(f"    {i}: {e.link}")
                
                linksearch_results.extend(api_results)
        except Exception as e:
            error_msg = f"Error in google_search_via_api: {str(e)}"
            logger.exception(error_msg)
            errors.append(error_msg)

        try:
            if "google_search_via_request_module" in search_sources:
                scrape_results = self._fetch_links_from_scrape(query, num_urls)
                logger.debug(f"scrape_results: {scrape_results}")
                linksearch_results.extend(scrape_results)
        except Exception as e:
            error_msg = f"Error in google_search_via_request_module: {str(e)}"
            logger.exception(error_msg)
            errors.append(error_msg)

        # 2. Apply filters
        try:
            linksearch_results = self._apply_filters(
                results=linksearch_results,
                allowed_countries=allowed_countries,
                forbidden_countries=forbidden_countries,
                allowed_domains=allowed_domains,
                forbidden_domains=forbidden_domains,
                validation_conditions={
                    "regex_validation_enabled": regex_based_link_validation,
                    "allow_file_links": allow_links_forwarding_to_files,
                    "keyword_match_list": keyword_match_based_link_validation
                }
            )
        except Exception as e:
            error_msg = f"Error in apply_filters: {str(e)}"
            logger.exception(error_msg)
            errors.append(error_msg)

        # 3. Optional LLM-based filtering
        if boolean_llm_filter_semantic:
            try:
                linksearch_results = self._filter_with_llm(linksearch_results)
            except Exception as e:
                error_msg = f"Error in LLM-based filtering: {str(e)}"
                logger.exception(error_msg)
                errors.append(error_msg)

        end_time = time.time()
        operation_result = OperationResult(
            total_time=(end_time - start_time),
            errors=errors
        )

        
        self.print(operation_result, api_results)

       

        # 4. Return results
        if output_format == "json":
            return {
                "operation_result": asdict(operation_result),
                "results": [asdict(item) for item in linksearch_results]
            }
        elif output_format == "linksearch":
            return linksearch_results
        else:
            raise ValueError("Unsupported output format. Choose 'json' or 'linksearch'.")

    # --------------------------------------------------------------------
    # Helper Methods for Gathering Links (Refactored)
    # --------------------------------------------------------------------
    def _fetch_links_from_api(self, query: str, num_urls: int) -> List[LinkSearch]:
        """
        Fetches links from the Google Custom Search API, then
        converts them into LinkSearch objects.
        """
        api_links = self.searcher.search_with_api(
            query=query,
            num_results=num_urls,
            google_search_api_key=self.google_api_key,
            cse_id=self.google_cse_id
        )
        return [LinkSearch(link=link, title="", metadata="") for link in api_links]

    def _fetch_links_from_scrape(self, query: str, num_urls: int) -> List[LinkSearch]:
  
        scraped_links = self.searcher.search(query)
     
        scraped_links = scraped_links[:num_urls]  # Limit to num_urls
        return [LinkSearch(link=link, title="", metadata="") for link in scraped_links]

    # --------------------------------------------------------------------
    # Filtering Methods
    # --------------------------------------------------------------------
    def _apply_filters(self,
                       results: List[LinkSearch],
                       allowed_countries: List[str],
                       forbidden_countries: List[str],
                       allowed_domains: List[str],
                       forbidden_domains: List[str],
                       validation_conditions: Dict) -> List[LinkSearch]:
        """
        Filter links based on countries, domains, and custom validation conditions.
        """
        filtered_results: List[LinkSearch] = []
        for result in results:
            link = result.link

            # -- Placeholder for country checks
            if allowed_countries:
                pass
            if forbidden_countries:
                pass

            # -- Allowed domains
            if allowed_domains and not any(domain.lower() in link.lower() for domain in allowed_domains):
                continue

            # -- Forbidden domains
            if forbidden_domains and any(domain.lower() in link.lower() for domain in forbidden_domains):
                continue

            # -- Additional validation conditions
            if validation_conditions:
                if not self._validate_link(result, validation_conditions):
                    continue

            filtered_results.append(result)
        return filtered_results

    def _validate_link(self, result: LinkSearch, validation_conditions: Dict) -> bool:
        """
        Validate link based on custom conditions:
         - regex_validation_enabled: apply a link-format regex check
         - allow_file_links: if False, skip links that appear to be file downloads
         - keyword_match_list: if provided, the link/title/metadata must contain at least one keyword
        """
        link = result.link
        metadata = result.metadata or ""
        title = result.title or ""

        # 1) Regex-based link validation
        if validation_conditions.get("regex_validation_enabled", False):
            # A more realistic link checking regex pattern:
            #  - Must start with http or https
            #  - Must have a valid domain (e.g. example.com)
            #  - Optionally allow some path/query characters
            pattern = r"^https?:\/\/([\w-]+\.)+[\w-]+(\/[\w\-./?%&=]*)?$"
            if not re.match(pattern, link):
                return False

        # 2) Allow or disallow file links
        if not validation_conditions.get("allow_file_links", True):
            # Example check for common file extensions
            file_exts = (".pdf", ".doc", ".docx", ".ppt", ".pptx", 
                         ".xls", ".xlsx", ".zip", ".rar", ".7z")
            lower_link = link.lower()
            if any(lower_link.endswith(ext) for ext in file_exts):
                return False

        # 3) Keyword-based link validation
        keyword_list = validation_conditions.get("keyword_match_list", [])
        if keyword_list:
            combined_text = f"{link} {title} {metadata}".lower()
            # Must match at least one keyword
            if not any(kw.lower() in combined_text for kw in keyword_list):
                return False

        return True
    
    def print(self,operation_result, api_results):

        print(f" ")  
        print(f"Results:")   

        print(f"Total Time : {operation_result.total_time}")   
        # for e in results:
        print(f"API Search Results: ")   
      
        for i,e in enumerate(api_results):
             print(f"    {i}: {e.link}")   
                  


    def _filter_with_llm(self, results: List[LinkSearch]) -> List[LinkSearch]:
        """
        Filter links using a language model (placeholder).
        For each link, we pass the combined title+metadata to an LLM service
        to decide if we keep it.
        """
        my_llm_service = MyLLMService()
        filtered_links = []

        for link_obj in results:
            # Combine title + metadata for semantic filtering
            string_data = (link_obj.title or "") + " " + (link_obj.metadata or "")

            try:
                # Perform categorization
                llm_result = my_llm_service.filter_simple(
                    semantic_filter_text=True,  # e.g., always pass True or param here
                    string_data=string_data
                )

                logger.debug("LLM filter result: %s", llm_result)
                if llm_result.success:
                    # If LLM approves, keep the link
                    filtered_links.append(link_obj)
                else:
                    logger.info("Link filtered out by LLM: %s", link_obj.link)
            except Exception as e:
                logger.exception(f"LLM filtering error for link='{link_obj.link}': {e}")
                # Decide whether to keep or skip the link in case of exception:
                # filtered_links.append(link_obj)  # or skip, depending on your policy

        return filtered_links


if __name__ == "__main__":
    serp_engine = SERPEngine()

    result_data = serp_engine.collect(
        query="best food in USA",
        num_urls=5,
        search_sources=["google_search_via_api"],
        regex_based_link_validation=False,             
        allow_links_forwarding_to_files=False,        
        output_format="json"  # or "linksearch"
    )
    print(result_data)
