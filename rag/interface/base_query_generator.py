from abc import ABC, abstractmethod

class BaseQueryGenerator(ABC):
    @abstractmethod
    def generate_test_set(self, source_file: str, sample_size: int = 50, output_file: str = "./rag/data/generated_test_set.json", max_workers: int = 20) -> None:
        """
        从源数据中采样，使用 LLM 生成查询，并保存为测试集文件。
        """
        pass
