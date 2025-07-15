# schemas.py (renamed from schemes.py)

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class SearchHit:
    """Represents a single search result."""
    link: str
    metadata: str = ""
    title: str = ""
    channel_name: str = ""
    channel_rank: Optional[int] = None
    
    def __post_init__(self):
        """Validate required fields."""
        if not self.link:
            raise ValueError("SearchHit requires a non-empty link")


@dataclass
class UsageInfo:
    """Tracks API usage and costs."""
    cost: float = 0.0
    requests_made: int = 1
    credits_used: Optional[float] = None
    
    def __add__(self, other: 'UsageInfo') -> 'UsageInfo':
        """Allow adding UsageInfo objects together."""
        return UsageInfo(
            cost=self.cost + other.cost,
            requests_made=self.requests_made + other.requests_made,
            credits_used=(self.credits_used or 0) + (other.credits_used or 0)
        )


@dataclass
class SerpChannelOp:
    """Results from a single search channel."""
    name: str
    results: List[SearchHit]
    usage: UsageInfo
    elapsed_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if the operation was successful."""
        return self.error is None and len(self.results) > 0
    
    @property
    def result_count(self) -> int:
        """Get the number of results."""
        return len(self.results)


@dataclass
class SerpEngineOp:
    """Aggregated results from multiple search channels."""
    usage: UsageInfo
    channels: List[SerpChannelOp]
    results: List[SearchHit]
    elapsed_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def results_by_channel(self) -> Dict[str, List[SearchHit]]:
        """
        Group results by their channel.
        
        Returns:
            Dict mapping channel names to their results
        """
        by_channel = {}
        for hit in self.results:
            if hit.channel_name not in by_channel:
                by_channel[hit.channel_name] = []
            by_channel[hit.channel_name].append(hit)
        return by_channel
    
    @property
    def total_results(self) -> int:
        """Get total number of results across all channels."""
        return len(self.results)
    
    @property
    def successful_channels(self) -> List[str]:
        """Get list of channels that returned results."""
        return [ch.name for ch in self.channels if ch.success]
    
    @property
    def failed_channels(self) -> List[str]:
        """Get list of channels that failed."""
        return [ch.name for ch in self.channels if not ch.success]
    
    def get_channel_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for each channel.
        
        Returns:
            Dict with channel statistics
        """
        stats = {}
        for channel in self.channels:
            stats[channel.name] = {
                'result_count': channel.result_count,
                'elapsed_time': channel.elapsed_time,
                'cost': channel.usage.cost,
                'success': channel.success,
                'error': channel.error
            }
        return stats


@dataclass
class ContextAwareSearchRequestObject:
    """Request object for context-aware searches."""
    query: str
    context: Optional[str] = None
    search_type: str = "general"  # general, academic, news, images, etc.
    user_location: Optional[str] = None
    user_language: str = "en"
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate search request."""
        if not self.query or not self.query.strip():
            raise ValueError("Search query cannot be empty")
        
        valid_search_types = ["general", "academic", "news", "images", "video", "shopping"]
        if self.search_type not in valid_search_types:
            raise ValueError(f"search_type must be one of: {valid_search_types}")


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    user_agent: str = "SERPEngine/1.0"
    verify_ssl: bool = True
    proxy: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    
    def to_request_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for requests library."""
        kwargs = {
            'timeout': self.timeout,
            'verify': self.verify_ssl,
            'headers': {'User-Agent': self.user_agent, **self.headers}
        }
        if self.proxy:
            kwargs['proxies'] = {'http': self.proxy, 'https': self.proxy}
        return kwargs


# Optional: Add type aliases for clarity
SearchResults = List[SearchHit]
ChannelResults = Dict[str, List[SearchHit]]