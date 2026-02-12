import logging
from rag.impl.retriever import Retriever
from rag.impl.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # 配置你的数据路径
    TEST_SET_FILE = "./rag/data/generated_test_set.json"

    try:
        # 假设 HybridRetriever 已经可以 import
        retriever = Retriever(use_reranker=True)
        evaluator = Evaluator(retriever)
        evaluator.evaluate(TEST_SET_FILE, top_k=10, with_species_filter=True, max_workers=20)
    except Exception as e:
        logger.error(f"Error: {e}")
