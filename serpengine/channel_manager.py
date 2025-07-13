# serpengine/channel_manager.py

import os
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ChannelRegistry:
    """Registry of available search channels and their requirements."""
    
    CHANNELS = {
        "google_api": {
            "module": "serpengine.google_search_api",
            "class": "GoogleSearchAPI",
            "required_env": ["GOOGLE_SEARCH_API_KEY", "GOOGLE_CSE_ID"],
            "description": "Google Custom Search API (paid after 100 queries/day)"
        },
        "google_scraper": {
            "module": "serpengine.google_searcher",
            "class": "GoogleSearcher",
            "required_env": [],
            "description": "Google HTML scraper (free but may be blocked)"
        },
        "serpapi": {
            "module": "serpengine.serpapi_searcher",
            "class": "SerpApiSearcher",
            "required_env": ["SERPAPI_API_KEY"],
            "description": "SerpAPI service (paid)"
        },
        "dataforseo": {
            "module": "serpengine.dataforseo_search",
            "class": "DataForSEOSearcher",
            "required_env": ["DATAFORSEO_USERNAME", "DATAFORSEO_PASSWORD"],
            "description": "DataForSEO SERP API (paid, $0.002 per SERP)"
        }
    }


class ChannelManager:
    """Manages search channel initialization and lifecycle."""
    
    def __init__(self, credentials: Dict[str, str] = None):
        """
        Initialize the channel manager.
        
        Args:
            credentials: Optional dict of credentials to override environment variables
        """
        self.credentials = credentials or {}
        self.searchers = {}
        self.available_channels = []
    
    def initialize_channels(
        self, 
        channels: List[str] = None, 
        auto_check_env: bool = True
    ) -> List[str]:
        """
        Initialize requested channels.
        
        Args:
            channels: List of channel names to initialize. If None, all channels.
            auto_check_env: If True, check for required environment variables.
            
        Returns:
            List of successfully initialized channel names.
        """
        # If no channels specified, try all
        if channels is None:
            channels = list(ChannelRegistry.CHANNELS.keys())
            logger.info("No channels specified, attempting to initialize all available channels")
        
        # Initialize each channel
        for channel in channels:
            if channel not in ChannelRegistry.CHANNELS:
                logger.warning(f"Unknown channel '{channel}', skipping")
                continue
            
            if self._initialize_channel(channel, auto_check_env):
                self.available_channels.append(channel)
        
        return self.available_channels
    
    def _initialize_channel(self, channel_name: str, auto_check_env: bool) -> bool:
        """
        Initialize a single channel.
        
        Returns:
            True if successful, False otherwise.
        """
        channel_info = ChannelRegistry.CHANNELS[channel_name]
        
        # Check required environment variables
        if auto_check_env:
            missing_vars = []
            for var in channel_info["required_env"]:
                value = self.credentials.get(var) or os.getenv(var)
                if not value:
                    missing_vars.append(var)
            
            if missing_vars:
                logger.warning(
                    f"Cannot initialize '{channel_name}': "
                    f"Missing environment variables: {missing_vars}"
                )
                return False
        
        try:
            # Dynamically import the module
            module_name = channel_info["module"]
            class_name = channel_info["class"]
            
            module = __import__(module_name, fromlist=[class_name])
            searcher_class = getattr(module, class_name)
            
            # Initialize with credentials
            init_kwargs = self._get_init_kwargs(channel_name)
            
            # Create instance
            self.searchers[channel_name] = searcher_class(**init_kwargs)
            logger.info(f"Initialized channel '{channel_name}': {channel_info['description']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize channel '{channel_name}': {e}")
            return False
    
    def _get_init_kwargs(self, channel_name: str) -> Dict[str, Any]:
        """Get initialization kwargs for a specific channel."""
        init_kwargs = {}
        
        if channel_name == "google_api":
            init_kwargs["api_key"] = (
                self.credentials.get("GOOGLE_SEARCH_API_KEY") or 
                os.getenv("GOOGLE_SEARCH_API_KEY")
            )
            init_kwargs["cse_id"] = (
                self.credentials.get("GOOGLE_CSE_ID") or 
                os.getenv("GOOGLE_CSE_ID")
            )
        elif channel_name == "serpapi":
            init_kwargs["api_key"] = (
                self.credentials.get("SERPAPI_API_KEY") or 
                os.getenv("SERPAPI_API_KEY")
            )
        elif channel_name == "dataforseo":
            init_kwargs["username"] = (
                self.credentials.get("DATAFORSEO_USERNAME") or 
                os.getenv("DATAFORSEO_USERNAME")
            )
            init_kwargs["password"] = (
                self.credentials.get("DATAFORSEO_PASSWORD") or 
                os.getenv("DATAFORSEO_PASSWORD")
            )
        # google_scraper doesn't need credentials
        
        return init_kwargs
    
    def get_searcher(self, channel_name: str):
        """Get a searcher instance by channel name."""
        return self.searchers.get(channel_name)
    
    def list_channels_status(self) -> Dict[str, Dict[str, Any]]:
        """
        List all channels and their status.
        
        Returns:
            Dict with channel info and initialization status.
        """
        result = {}
        
        for name, info in ChannelRegistry.CHANNELS.items():
            result[name] = {
                "description": info["description"],
                "required_env": info["required_env"],
                "initialized": name in self.available_channels,
                "missing_env": []
            }
            
            # Check missing env vars
            for var in info["required_env"]:
                if not (self.credentials.get(var) or os.getenv(var)):
                    result[name]["missing_env"].append(var)
        
        return result
    
    def execute_search(self, channel_name: str, query: str, num_results: int):
        """
        Execute search on a specific channel.
        
        Args:
            channel_name: Name of the channel to use
            query: Search query
            num_results: Number of results to fetch
            
        Returns:
            SERPMethodOp with results
        """
        searcher = self.searchers.get(channel_name)
        if not searcher:
            raise ValueError(f"Channel '{channel_name}' not initialized")
        
        # Call the appropriate search method based on channel
        if channel_name == "google_api":
            return searcher.search(query=query, num_results=num_results)
        elif channel_name == "google_scraper":
            return searcher.search(query=query, num_results=num_results)
        elif channel_name == "serpapi":
            return searcher.search(query=query, num_results=num_results)
        elif channel_name == "dataforseo":
            return searcher.search_live(query=query, num_results=num_results)
        else:
            raise ValueError(f"No search method defined for '{channel_name}'")
    
    async def execute_search_async(self, channel_name: str, query: str, num_results: int):
        """
        Execute async search on a specific channel.
        
        Args:
            channel_name: Name of the channel to use
            query: Search query
            num_results: Number of results to fetch
            
        Returns:
            SERPMethodOp with results
        """
        searcher = self.searchers.get(channel_name)
        if not searcher:
            raise ValueError(f"Channel '{channel_name}' not initialized")
        
        # Call the appropriate async search method
        if channel_name == "google_api":
            return await searcher.async_search(query=query, num_results=num_results)
        elif channel_name == "google_scraper":
            return await searcher.async_search(query=query, num_results=num_results)
        elif channel_name == "serpapi":
            return await searcher.async_search(query=query, num_results=num_results)
        elif channel_name == "dataforseo":
            return await searcher.async_search_live(query=query, num_results=num_results)
        else:
            raise ValueError(f"No async search method defined for '{channel_name}'")