import logging
from typing import List
from state.diagnosis import DiagnosisState
from rag.impl.retriever import Retriever

# 假设我们在 main.py 或某个地方初始化了全局的 retriever 实例
# 或者在这里 lazy init (生产环境通常通过依赖注入传递)
# 为了 Demo 简单，我们这里假设能引用到 retriever 实例
# 更好的做法是在 graph config 里传递 retriever，或者在这里实例化

logger = logging.getLogger(__name__)

# 初始化 Retriever (注意：在实际生产中，这通常是在系统启动时完成的单例)
# 这里为了代码独立性，演示一下初始化，实际应该复用全局实例
_retriever_instance = None


def get_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        # 确保路径和你项目一致
        _retriever_instance = Retriever(
            model_cache_dir="./rag/model_cache", use_reranker=True
        )
    return _retriever_instance


def diagnosis_retriever_node(state: DiagnosisState):
    """
    执行检索：接收 search_queries，调用 Retriever，返回 retrieved_docs
    """
    logger.info("--- Entering Retrieve Node ---")

    # 1. 获取输入
    queries = state.get("search_queries", [])
    pet_profile = state.get("pet_profile")

    # 守卫逻辑
    if not queries:
        logger.warning("No search queries found in state. Skipping retrieval.")
        return {"retrieved_docs": []}

    try:
        retriever = Retriever(url = "http://localhost:6333", 
                            collection_name = "pet_health_hybrid",
                            timeout = 5.0,
                            model_cache_dir = "./rag/model_cache",
                            use_reranker = True)
        all_results = []

        # 2. 执行多路检索 (Serial Execution for Demo, Parallel for Production)
        # 因为我们采用了“全量混合”策略，每个 Query 都要查一遍
        for q in queries:
            # 这里的 limit 可以稍微大一点，给 Reranker 更多候选
            # 关键：传入 species_filter !!
            species_filter = None
            if pet_profile and pet_profile.species:
                # 处理 Enum 取 value 的逻辑
                species_filter = getattr(
                    pet_profile.species, "value", str(pet_profile.species)
                )

            logger.info(f"Executing Search: '{q}' | Filter: {species_filter}")

            results = retriever.search(
                query=q,
                filters={"species": species_filter},
                limit=10,  # 这里的 limit 是单次检索的召回量
            )
            all_results.extend(results)

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        # 即使失败，也尽量不要 Crash 整个流程，返回空列表让后续节点处理(比如 Query Rewrite)
        return {"retrieved_docs": []}

    # 3. 结果去重 (Deduplication)
    # 多个 Query 可能会搜出同一个文档，按 ID 去重
    unique_docs_map = {res.id: res for res in all_results}
    unique_docs = list(unique_docs_map.values())

    # 4. (可选) 二次重排序 Logic
    # 如果你的 retriever.search 内部已经做了 Rerank (你的代码里似乎做过了)，
    # 这里只需要按 score 排序截断即可。
    # 如果 retriever.search 只是返回了召回结果，这里应该调用 Reranker。

    # 既然你的 Retriever.search 已经包含了 Reranker 逻辑 (Source 221)，
    # 我们这里主要做的是 "Merge Sort"
    unique_docs.sort(key=lambda x: x.score, reverse=True)

    # 取最终 Top K (比如 Top 5 给 LLM 阅读)
    final_docs = unique_docs[:5]

    # 5. 格式化输出
    # 将 SearchResult 对象转化为 LLM 易读的字符串格式
    formatted_docs = []
    for doc in final_docs:
        # 组装一个清晰的文档字符串
        meta = doc.metadata or {}
        keywords_raw = meta.get("symptom_keywords", [])
        if isinstance(keywords_raw, list):
            keywords_str = ", ".join(keywords_raw)
        else:
            keywords_str = str(keywords_raw)
        doc_str = (
            f"Doc ID: {doc.id}\n"
            f"Source: {doc.source}\n" # SearchResult 类本身有 source 属性，所以 doc.source 是对的
            f"Species: {meta.get('species', 'unknown')}\n"       # <--- 修正
            f"Breed: {meta.get('specific_breed', 'unknown')}\n"  # <--- 修正
            f"Symptoms: {doc.text}\n"
            f"Symptom_keywords: {keywords_str}\n"                 # <--- 修正 (单引号问题也在此解决)
            f"Diagnosis: {meta.get('condition', 'unknown')}\n"    # <--- 修正
        )
        formatted_docs.append(doc_str)

    logger.info(
        f"Retrieval Complete. Found {len(formatted_docs)} unique relevant docs."
    )

    # 6. 更新 State
    return {"retrieved_docs": formatted_docs}
