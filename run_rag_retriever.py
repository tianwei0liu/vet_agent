import logging
from rag.impl.retriever import Retriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 单元测试入口 ---
if __name__ == "__main__":
    retriever = Retriever(model_cache_dir="./rag/model_cache", use_reranker=True)
    
    # 测试 Case: 包含模糊描述和具体症状
    test_query = "cat having trouble eating and making weird mouth movements"
    test_query = "my hamster has crusty growths"
    species = "hamster"
    
    results = retriever.search(test_query, species_filter=species, limit=3)
    
    logger.info("\n" + "="*50)
    logger.info(f"Query: {test_query}")
    logger.info("="*50)
    
    for i, res in enumerate(results):
        logger.info(f"\nRank {i+1} | Score: {res.score:.4f} | Source: {res.source}")
        logger.info(f"Species: {res.metadata.get('species')}")
        logger.info(f"Breed: {res.metadata.get('specific_breed')}")
        logger.info(f"Condition: {res.metadata.get('condition')}")
        logger.info(f"Text: {res.text}")
