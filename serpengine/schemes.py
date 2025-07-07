from typing import List
from dataclasses import dataclass


from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class ContextAwareSearchRequestObject:
    
    query: str
    identifier_context: Optional[str] = None 
    what_to_impute: Optional[Any] = None 
    task_purpose: Optional[str] = None 
   

@dataclass
class SearchHit:
    """One individual search result."""
    link: str
    metadata: str
    title: str

@dataclass
class UsageInfo:
    """Billing information for the operation."""
    cost: float

@dataclass
class SERPMethodOp:
    """
    Operation details for a single search method.

    - name: identifier for the method (e.g., 'google_api', 'scrape')
    - results: list of SearchHit from this method
    - usage: usage stats for this method
    - elapsed_time: time spent in seconds for this method
    """
    name: str
    results: List[SearchHit]
    usage: UsageInfo
    elapsed_time: float

@dataclass
class SerpEngineOp:
    """
    Combined search results from all methods + aggregate stats.

    - methods: list of SERPMethodOp, one per search source
    - usage: aggregate usage stats (sum of costs)
    - results: combined list of all SearchHit
    - elapsed_time: total time across all methods (in seconds)
    """
    methods: List[SERPMethodOp]
    usage: UsageInfo
    results: List[SearchHit]
    elapsed_time: float
    
    def all_links(self) -> List[str]:
        return [hit.link for hit in self.results]
