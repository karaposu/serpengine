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
    channel_name: str =None
    channel_rank: int =None

@dataclass
class UsageInfo:
    """Billing information for the operation."""
    cost: float


@dataclass
class SerpChannelOp:
    """
    Operation details for a single search channel.

    - name: identifier for the channel (e.g., 'google_api', 'serpapi')
    - results: list of SearchHit from this channel
    - usage: usage stats for this channel
    - elapsed_time: time spent in seconds for this channel
    """
    name: str
    results: List[SearchHit]
    usage: UsageInfo
    elapsed_time: float


@dataclass
class SerpEngineOp:
    """
    Combined search results from all channels + aggregate stats.

    - channels: list of SerpChannelOp, one per search channel
    - usage: aggregate usage stats (sum of costs)
    - results: combined list of all SearchHit
    - elapsed_time: total time across all channels (in seconds)
    """
    channels: List[SerpChannelOp]
    usage: UsageInfo
    results: List[SearchHit]
    elapsed_time: float
    
    def all_links(self) -> List[str]:
        return [hit.link for hit in self.results]
    
    def results_by_channel(self) -> dict:
        """Group results by channel name."""
        channel_results = {}
        for hit in self.results:
            if hit.channel_name not in channel_results:
                channel_results[hit.channel_name] = []
            channel_results[hit.channel_name].append(hit)
        return channel_results