from abc import ABC, abstractmethod
from typing import List
from rag.schema.search_result import SearchResult

class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        pass
