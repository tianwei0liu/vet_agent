import logging
from typing import List, Dict, Any, Optional, Union
from rag.interface.base_retriever import BaseRetriever
from rag.schema.search_result import SearchResult

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)


class Retriever(BaseRetriever):
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "pet_health_hybrid",
        timeout: float = 5.0,
        model_cache_dir: str = "./rag/model_cache",
        use_reranker: bool = True,
    ):

        self.collection_name = collection_name
        self.use_reranker = use_reranker

        # 1. 连接数据库
        self.client = QdrantClient(url=url, timeout=timeout)

        # 2. 加载 Embedding 模型
        logger.info("Loading Embedding Models for Retrieval...")
        self.dense_model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5", cache_dir=model_cache_dir
        )
        self.sparse_model = SparseTextEmbedding(
            model_name="Qdrant/bm25", cache_dir=model_cache_dir
        )

        # 3. 加载 Reranker 模型
        if self.use_reranker:
            logger.info("Loading Reranker (ms-marco-TinyBERT-L-2-v2)...")
            self.ranker = Ranker(
                model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=model_cache_dir
            )

    def _get_dense_vector(self, text: str) -> List[float]:
        return list(self.dense_model.embed([text]))[0].tolist()

    def _get_sparse_vector(self, text: str) -> models.SparseVector:
        embedding = list(self.sparse_model.embed([text]))[0]
        return models.SparseVector(
            indices=embedding.indices.tolist(), values=embedding.values.tolist()
        )

    def _reciprocal_rank_fusion(
        self, dense_results: List[Any], sparse_results: List[Any], k: int = 40
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (倒数排名融合)
        """
        fused_scores = {}
        weight_dense = 1.0
        weight_sparse = 1.0

        # 处理 Dense
        for rank, hit in enumerate(dense_results):
            if hit.id not in fused_scores:
                fused_scores[hit.id] = {"hit": hit, "score": 0.0, "sources": []}
            fused_scores[hit.id]["score"] += 1 / (k + rank + 1) * weight_dense
            fused_scores[hit.id]["sources"].append("dense")

        # 处理 Sparse
        for rank, hit in enumerate(sparse_results):
            if hit.id not in fused_scores:
                fused_scores[hit.id] = {"hit": hit, "score": 0.0, "sources": []}
            fused_scores[hit.id]["score"] += 1 / (k + rank + 1) * weight_sparse
            fused_scores[hit.id]["sources"].append("sparse")

        return sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

    def _build_qdrant_filter(
        self, filters: Optional[Dict[str, Any]]
    ) -> Optional[models.Filter]:
        """
        通用 Filter 构建器：将字典转换为 Qdrant 的精确匹配 Filter。
        不做任何值校验，完全信任调用方。
        """
        if not filters:
            return None

        # 允许高级用户直接传入构建好的 models.Filter 对象
        if isinstance(filters, models.Filter):
            return filters

        must_conditions = []
        for key, value in filters.items():
            if value is None:
                continue

            # 默认使用 MatchValue (Exact Match)
            # 如果未来需要 Range (如 age > 5)，调用方可以传 models.Filter 对象进来
            must_conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )

        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,  # 变更点：通用 filters
        limit: int = 10,
        recall_limit: int = 40,
    ) -> List[SearchResult]:
        """
        通用混合检索接口
        :param filters: 过滤条件字典，例如 {"species": "cat", "source": "book"}
        """
        logger.info(f"Searching for: '{query}' | Filters: {filters}")

        # Step 0: 构建 Filter
        qdrant_filter = self._build_qdrant_filter(filters)

        # Step 1: 向量化
        dense_vec = self._get_dense_vector(query)
        sparse_vec = self._get_sparse_vector(query)

        # Step 2: 多路召回
        dense_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec,
            using="dense",
            limit=recall_limit,
            query_filter=qdrant_filter,  # Qdrant 原生支持 filter
            with_payload=True,
        ).points

        sparse_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_vec,
            using="sparse",
            limit=recall_limit,
            query_filter=qdrant_filter,
            with_payload=True,
        ).points

        logger.info(f"Recall: Dense={len(dense_hits)}, Sparse={len(sparse_hits)}")

        # Step 3: RRF 融合
        fused_results = self._reciprocal_rank_fusion(dense_hits, sparse_hits)

        # 准备 Rerank 候选
        candidates = []
        for item in fused_results:
            hit = item["hit"]
            text_content = hit.payload.get("text", "")
            candidates.append({"id": hit.id, "text": text_content, "meta": hit.payload})

        # Step 4: Rerank
        final_results = []
        if self.use_reranker and candidates:
            # logger.info("Reranking candidates...") # 减少日志噪音
            rerank_request = RerankRequest(query=query, passages=candidates)
            rerank_results = self.ranker.rerank(rerank_request)

            for res in rerank_results[:limit]:
                final_results.append(
                    SearchResult(
                        id=res["id"],
                        score=res["score"],
                        text=res["text"],
                        metadata=res["meta"],
                        source="reranked",
                    )
                )
        else:
            for item in fused_results[:limit]:
                hit = item["hit"]
                final_results.append(
                    SearchResult(
                        id=hit.id,
                        score=item["score"],
                        text=hit.payload.get("text", ""),
                        metadata=hit.payload,
                        source="+".join(item["sources"]),
                    )
                )

        return final_results
