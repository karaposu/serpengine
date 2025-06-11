# here is schemes.py

from typing import List, Dict, Union
from dataclasses import dataclass, field, asdict
# -----------------------------------------------------------------------------------------
# Dataclasses for the final results
# -----------------------------------------------------------------------------------------
@dataclass
class LinkSearch:
    """Represents a single search result with link, metadata, and title."""
    link: str
    metadata: str
    title: str

@dataclass
class OperationResult:
    """Holds details about the operation's outcome."""
    total_time: float
    errors: List[str] = field(default_factory=list)