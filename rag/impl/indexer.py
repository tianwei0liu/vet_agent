import json
import logging
import time
from typing import List, Dict, Any, Generator, Optional
from pathlib import Path
from itertools import islice

from pydantic import BaseModel, Field, ValidationError
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from rag.interface.base_indexer import BaseIndexer
from rag.schema.pet_record import PetRecord

logger = logging.getLogger(__name__)

# --- 3. 核心实现类 (Concrete Implementation) ---
class Indexer(BaseIndexer):
    def __init__(self, url: str = "http://localhost:6333", timeout : int = 120.0, api_key: Optional[str] = None, model_cache_dir: str = "./rag/model_cache"):
        self.url = url
        self.timeout = timeout
        self.api_key = api_key
        self.client: Optional[QdrantClient] = None
        
        # 初始化模型 (Lazy Loading 可以在 connect 中做，但在 __init__ 中做可以提前暴露模型加载错误)
        logger.info("Initializing Embedding Models (FastEmbed ONNX)...")
        # Dense Model: BGE-Small (高效且强大)
        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=model_cache_dir)
        # Sparse Model: SPLADE (专门用于稀疏检索)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", cache_dir=model_cache_dir)
        logger.info("Models initialized successfully.")

    def connect(self) -> None:
        try:
            self.client = QdrantClient(url=self.url, timeout = self.timeout, api_key=self.api_key)
            # 简单的心跳检测
            self.client.get_collections()
            logger.info(f"Successfully connected to Qdrant at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Could not connect to Qdrant: {e}")

    def create_collection(self, collection_name: str, overwrite=False) -> None:
        if not self.client:
            raise ConnectionError("Client not connected. Call connect() first.")

        if self.client.collection_exists(collection_name):
            if not overwrite:
                logger.info(f"Collection '{collection_name}' already exists. Overwrite disabled. Skipping creation.")
                return
            else:
                logger.warning(f"Collection '{collection_name}' exists. Overwriting existing collection...")
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted collection '{collection_name}'.")

        logger.info(f"Creating collection '{collection_name}' with Hybrid Search Schema...")
        
        # 定义 Schema
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                # Named Vector: "dense"
                "dense": models.VectorParams(
                    size=384,  # BGE-small 的维度
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                # Named Sparse Vector: "sparse"
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False, # 生产环境如果内存紧张设为 True
                    )
                )
            }
        )
        logger.info(f"Collection '{collection_name}' created.")

    def _batch_iterator(self, iterable, size):
        """Helper to chunk data into batches"""
        iterator = iter(iterable)
        for first in iterator:
            yield [first] + list(islice(iterator, size - 1))

    def index_data(self, collection_name: str, data_path: str, batch_size: int = 64) -> None:
        if not self.client:
            raise ConnectionError("Client not connected.")

        path = Path(data_path)
        if not path.exists():
            logger.error(f"Data file not found: {path}")
            raise FileNotFoundError(f"{path} does not exist")

        # 1. 读取并校验数据
        logger.info(f"Loading data from {path}...")
        valid_records: List[PetRecord] = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                for item in raw_data:
                    try:
                        valid_records.append(PetRecord(**item))
                    except ValidationError as ve:
                        logger.warning(f"Skipping invalid record ID {item.get('id', 'unknown')}: {ve}")
        except json.JSONDecodeError:
            logger.error("Invalid JSON file format.")
            return

        total_records = len(valid_records)
        logger.info(f"Found {total_records} valid records. Starting indexing...")

        # 2. 批量处理
        start_time = time.time()
        indexed_count = 0

        for batch in self._batch_iterator(valid_records, batch_size):
            try:
                # 准备文本列表
                dense_texts = [r.dense_search_content for r in batch]
                sparse_texts = [r.sparse_search_content for r in batch]
                
                # 生成向量 (Generator 转 List)
                # FastEmbed 返回的是 generator，需要 list() 消费掉
                dense_embeddings = list(self.dense_model.embed(dense_texts))
                sparse_embeddings = list(self.sparse_model.embed(sparse_texts))

                # 构建 Points
                points = []
                for i, record in enumerate(batch):
                    # 1. 处理 Dense: numpy array -> list
                    dense_vec = dense_embeddings[i].tolist()
                    # 2. 处理 Sparse: FastEmbed Object -> Qdrant Model
                    # FastEmbed 返回的对象包含 .indices 和 .values (都是 numpy array)
                    sparse_vec = models.SparseVector(
                        indices=sparse_embeddings[i].indices.tolist(),
                        values=sparse_embeddings[i].values.tolist()
                    )
                    
                    points.append(models.PointStruct(
                        id=record.id,
                        vector={
                            "dense": dense_vec,
                            "sparse": sparse_vec
                        },
                        payload=record.model_dump()
                    ))

                # 写入 Qdrant
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                indexed_count += len(batch)
                logger.info(f"Indexed {indexed_count}/{total_records} records...")

            except Exception as e:
                logger.error(f"Error indexing batch starting at ID {batch[0].id}: {e}")
                # 生产环境这里通常会把失败的 batch ID 写入死信队列 (DLQ)

        elapsed = time.time() - start_time
        logger.info(f"Indexing completed. {indexed_count} records processed in {elapsed:.2f} seconds.")
