# serpengine/base_channel.py

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

from .schemes import SearchHit, UsageInfo, SerpChannelOp

load_dotenv()
logger = logging.getLogger(__name__)


class BaseSearchChannel(ABC):
    """
    Abstract base class for all search channels.
    Provides common functionality and enforces interface.
    """
    
    # Channel metadata - override in subclasses
    CHANNEL_NAME = "base"
    REQUIRED_ENV_VARS = []
    DESCRIPTION = "Base search channel"
    
    # Pricing info - override if channel has costs
    COST_PER_SEARCH = 0.0
    FREE_SEARCHES_PER_DAY = None
    
    def __init__(self, credentials: Dict[str, str] = None):
        """
        Initialize the search channel.
        
        Args:
            credentials: Optional dict to override environment variables
        """
        self.name = self.CHANNEL_NAME
        self.credentials = credentials or {}
        self._loaded_credentials = {}
        
        # Load and validate credentials
        self.load_credentials()
        self.validate_credentials()
        
        # Initialize channel-specific resources
        self.initialize()
        
        logger.info(f"[{self.name}] Initialized: {self.DESCRIPTION}")
    
    def load_credentials(self) -> Dict[str, str]:
        """
        Load credentials from environment or provided dict.
        
        Returns:
            Dict of loaded credentials
        """
        for var_name in self.REQUIRED_ENV_VARS:
            # First check provided credentials, then environment
            value = self.credentials.get(var_name) or os.getenv(var_name)
            if value:
                self._loaded_credentials[var_name] = value
            else:
                self._loaded_credentials[var_name] = None
        
        return self._loaded_credentials
    
    def validate_credentials(self):
        """
        Validate that all required credentials are present.
        Raises ValueError if any are missing.
        """
        missing = []
        for var_name in self.REQUIRED_ENV_VARS:
            if not self._loaded_credentials.get(var_name):
                missing.append(var_name)
        
        if missing:
            raise ValueError(
                f"[{self.name}] Missing required credentials: {missing}. "
                f"Set environment variables or pass in credentials dict."
            )
    
    def get_credential(self, var_name: str) -> Optional[str]:
        """Get a specific credential value."""
        return self._loaded_credentials.get(var_name)
    
    @abstractmethod
    def initialize(self):
        """
        Initialize channel-specific resources.
        Called after credentials are loaded and validated.
        """
        pass
    
    @abstractmethod
    def search(self, query: str, num_results: int = 10, **kwargs) -> SerpChannelOp:
        """
        Perform a search.
        
        Args:
            query: Search query
            num_results: Number of results to fetch
            **kwargs: Channel-specific parameters
            
        Returns:
            SerpChannelOp with results
        """
        pass
    
    @abstractmethod
    async def async_search(self, query: str, num_results: int = 10, **kwargs) -> SerpChannelOp:
        """
        Perform an async search.
        
        Args:
            query: Search query
            num_results: Number of results to fetch
            **kwargs: Channel-specific parameters
            
        Returns:
            SerpChannelOp with results
        """
        pass
    
    def calculate_cost(self, num_searches: int = 1) -> float:
        """
        Calculate the cost for a given number of searches.
        Override for complex pricing models.
        
        Args:
            num_searches: Number of searches
            
        Returns:
            Total cost
        """
        if self.FREE_SEARCHES_PER_DAY is not None:
            # Simple model: first N searches are free
            if num_searches <= self.FREE_SEARCHES_PER_DAY:
                return 0.0
            else:
                paid_searches = num_searches - self.FREE_SEARCHES_PER_DAY
                return paid_searches * self.COST_PER_SEARCH
        else:
            # All searches have the same cost
            return num_searches * self.COST_PER_SEARCH
    
    def is_link_valid(self, link: str) -> bool:
        """
        Check if a link is valid.
        Can be overridden for channel-specific validation.
        
        Args:
            link: URL to validate
            
        Returns:
            True if valid
        """
        if not link:
            return False
        return link.startswith(("http://", "https://"))
    
    def is_link_to_website(self, link: str) -> bool:
        """
        Check if link leads to a website (not a file).
        Can be overridden for channel-specific logic.
        
        Args:
            link: URL to check
            
        Returns:
            True if it's a website
        """
        file_extensions = [
            '.pdf', '.doc', '.docx', '.ppt', '.pptx',
            '.xls', '.xlsx', '.zip', '.rar', '.tar', '.gz'
        ]
        lower_link = link.lower()
        return not any(lower_link.endswith(ext) for ext in file_extensions)
    
    def create_search_hit(
        self, 
        link: str, 
        title: str = "", 
        metadata: str = "",
        **kwargs
    ) -> SearchHit:
        """
        Create a SearchHit with channel info pre-populated.
        
        Args:
            link: URL
            title: Page title
            metadata: Additional metadata
            **kwargs: Any additional fields
            
        Returns:
            SearchHit object
        """
        return SearchHit(
            link=link,
            title=title,
            metadata=metadata,
            channel_name=self.name,
            channel_rank=None,  # Will be set by channel manager
            **kwargs
        )
    
    def create_channel_op(
        self,
        results: List[SearchHit],
        elapsed_time: float,
        cost: Optional[float] = None
    ) -> SerpChannelOp:
        """
        Create a SerpChannelOp with standard fields.
        Automatically sets channel_rank for all results.
        
        Args:
            results: List of SearchHit objects
            elapsed_time: Time taken for search
            cost: Optional cost override
            
        Returns:
            SerpChannelOp object
        """
        # Set channel ranking for all results
        for rank, hit in enumerate(results, 1):
            hit.channel_rank = rank
            # Ensure channel_name is set
            if not hit.channel_name:
                hit.channel_name = self.name
        
        if cost is None:
            cost = self.calculate_cost(1)
        
        return SerpChannelOp(
            name=self.name,
            results=results,
            usage=UsageInfo(cost=cost),
            elapsed_time=elapsed_time
        )
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}')>"