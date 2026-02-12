from abc import ABC, abstractmethod
from typing import List


class BaseIndexer(ABC):
    
    @abstractmethod
    def connect(self) -> None:
        """建立数据库连接"""
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, overwrite = False) -> None:
        """创建集合，定义 Dense 和 Sparse 的 Schema"""
        pass

    @abstractmethod
    def index_data(self, collection_name: str, data_path: str, batch_size: int = 64) -> None:
        """
        主入口：读取数据 -> 生成向量 -> 批量写入
        """
        pass
