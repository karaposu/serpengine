"""
internal_test_file.py

Demonstrates how to test various configurations of LinkSearchAgent by
providing different parameters and observing the returned results.
"""

import logging
from link_finder_engine.link_finder_engine import LinkFinderEngine  # Update to your actual import path
# If your file is named differently, e.g. "my_link_searcher.py", adjust the import.

logging.basicConfig(level=logging.INFO)

def test_basic_search():
    """
    Test a simple, minimal configuration:
    - default regex validation (True)
    - file links allowed
    - no specific keyword matching
    """
    engine = LinkFinderEngine()
    result = engine.collect(
        query="test query",
        search_sources=["google_search_via_request_module"],  # Only scraping Google
        num_urls=3,
        output_format="json"
    )
    print("\n===== Test: Basic Search (Scraping) =====")
    print(result)


def test_api_search_only():
    """
    Test using only the Google Search API with additional parameters.
    - Regex validation disabled
    - No file links
    - Keyword matching required
    """
    agent = LinkFinderEngine()
    result = agent.collect(
        query="STM32 Microprocessor",
        regex_based_link_validation=False,
        allow_links_forwarding_to_files=False,
        keyword_match_based_link_validation=["STM32", "microprocessor"],
        search_sources=["google_search_via_api"],  # Only Google API
        num_urls=5,
        output_format="json"
    )
    print("\n===== Test: API Search Only =====")
    print(result)


def test_api_and_scraping_with_domains():
    """
    Test combined search sources (API + scraping).
    - Both regex validation and file-link allowance toggled
    - Restrict to allowed_domains
    """
    agent = LinkFinderEngine()
    result = agent.collect(
        query="example query",
        regex_based_link_validation=True,
        allow_links_forwarding_to_files=False,
        keyword_match_based_link_validation=["example"],
        search_sources=["google_search_via_api", "google_search_via_request_module"],
        allowed_domains=["example.com"],
        num_urls=5,
        output_format="json"
    )
    print("\n===== Test: API + Scraping with Allowed Domains =====")
    print(result)


def test_llm_filtering():
    """
    Test using LLM-based semantic filtering after gathering links.
    Note: The MyLLMService.filter_simple must be implemented or mocked for a real test.
    """
    agent = LinkFinderEngine()
    result = agent.collect(
        query="some interesting query",
        search_sources=["google_search_via_request_module"],
        boolean_llm_filter_semantic=True,  # Engage LLM filtering
        num_urls=3,
        output_format="json"
    )
    print("\n===== Test: LLM Filtering =====")
    print(result)


if __name__ == "__main__":
    """
    Manually execute each test function. In practice, you could integrate these into a
    formal test suite (e.g., unittest, pytest) for automated testing.
    """
    test_basic_search()
    test_api_search_only()
    test_api_and_scraping_with_domains()
    test_llm_filtering()
