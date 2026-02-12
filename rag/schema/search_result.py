from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SearchResult:
    id: int
    score: float
    text: str
    metadata: Dict[str, Any]
    # 用于调试：知道是哪路召回的
    source: str = "unknown" 