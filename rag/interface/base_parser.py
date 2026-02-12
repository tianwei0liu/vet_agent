from abc import ABC, abstractmethod
from typing import List

class BaseParser(ABC):
    """
    RAG Pipeline 的第一个环节：Parser
    职责：读取原始数据 -> 清洗/结构化 -> 保存中间格式
    """

    @abstractmethod
    def parse(self, input_path: str, output_path: str) -> str:
        """
        执行解析逻辑
        :param input_path: 原始数据路径
        :param output_path: 结构化数据保存路径
        :return: 保存文件的最终路径
        """
        pass
