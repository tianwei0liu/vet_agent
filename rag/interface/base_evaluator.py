from abc import ABC, abstractmethod
from typing import Dict

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, test_file: str, top_k: int = 5) -> Dict[str, float]:
        """
        加载测试集，运行检索，计算 Hit Rate 和 MRR。
        """
        pass
