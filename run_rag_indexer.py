import logging
from rag.impl.indexer import Indexer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    ##################################################################################################################################
    # mkdir -p ./rag/qdrant_storage
    # sudo usermod -aG docker $USER
    # newgrp docker
    # docker ps
    # docker run -d --name pet_vectordb -p 6333:6333 -p 6334:6334 -v $(pwd)/rag/qdrant_storage:/qdrant/storage:z qdrant/qdrant:latest
    # http://localhost:6333/dashboard
    ##################################################################################################################################
    # 配置
    DATA_FILE = "./rag/data/pet_health_symptoms_dataset.json"
    COLLECTION_NAME = "pet_health_hybrid"
    
    # 初始化 Indexer
    indexer = Indexer(url="http://localhost:6333", model_cache_dir="./rag/model_cache")
    
    try:
        indexer.connect()
        indexer.create_collection(COLLECTION_NAME, overwrite=True)
        
        # 为了演示，确保你有这个文件，或者替换为你的真实路径
        indexer.index_data(COLLECTION_NAME, DATA_FILE)
        logger.info("Indexing successful!")
        
    except Exception as e:
        logger.critical(f"Fatal error in main loop: {e}")