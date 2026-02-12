import json
import os
import logging
import concurrent.futures
from typing import Dict, List, Any
from collections import defaultdict

from rag.interface.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

class Evaluator(BaseEvaluator):
    def __init__(self, retriever_instance):
        self.retriever = retriever_instance

    def _evaluate_single_case(self, case: Dict[str, Any], top_k: int, with_species_filter: bool) -> Dict[str, Any]:
        """
        处理单个测试用例的原子函数（用于线程池调用）
        """
        query = case.get("query", "")
        target_id = case.get("id")
        difficulty = case.get("difficulty", "Unknown") # 获取难度标签
        
        # 1. 执行检索
        try:
            if with_species_filter:
                species = case.get("species")
                # 注意：确保你的 retriever.search 方法支持 species_filter 参数
                results = self.retriever.search(query=query, filters={"species": species}, limit=top_k)
            else:
                results = self.retriever.search(query=query, limit=top_k)
        except Exception as e:
            logger.error(f"Error searching for query ID {target_id}: {e}")
            return {
                "difficulty": difficulty,
                "hit": False,
                "mrr": 0.0,
                "error": True
            }

        # 2. 计算指标
        retrieved_ids = [res.id for res in results]
        
        is_hit = target_id in retrieved_ids
        mrr_val = 0.0
        
        if is_hit:
            rank = retrieved_ids.index(target_id) + 1
            mrr_val = 1.0 / rank

        return {
            "difficulty": difficulty,
            "hit": is_hit,
            "mrr": mrr_val,
            "error": False
        }

    def evaluate(self, test_file: str, top_k: int = 10, with_species_filter: bool = True, max_workers: int = 20) -> Dict[str, Dict[str, float]]:
        """
        并行执行评测，并按难度分层统计指标
        :param max_workers: 并发线程数
        """
        logger.info(f"Starting evaluation using {test_file} @ Top-{top_k} | Filter={with_species_filter} | Workers={max_workers}...")
        
        if not os.path.exists(test_file):
            logger.error(f"Test file {test_file} not found.")
            return {}

        with open(test_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)

        total_cases = len(test_cases)
        
        # 初始化统计容器
        # 结构: stats[category] = {'hits': 0, 'mrr_sum': 0.0, 'count': 0}
        # Categories 包括: 'Total', 'Easy', 'Medium', 'Hard', etc.
        stats = defaultdict(lambda: {'hits': 0, 'mrr_sum': 0.0, 'count': 0})

        # --- 并行执行 ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_case = {
                executor.submit(self._evaluate_single_case, case, top_k, with_species_filter): case 
                for case in test_cases
            }
            
            processed_count = 0
            
            # 处理结果 (as_completed 会在任务完成时立刻返回)
            for future in concurrent.futures.as_completed(future_to_case):
                processed_count += 1
                result = future.result()
                
                if result.get("error"):
                    continue

                difficulty = result["difficulty"]
                hit = result["hit"]
                mrr = result["mrr"]

                # 1. 更新总榜 (Total)
                stats["Total"]["count"] += 1
                if hit:
                    stats["Total"]["hits"] += 1
                    stats["Total"]["mrr_sum"] += mrr

                # 2. 更新分榜 (Difficulty)
                stats[difficulty]["count"] += 1
                if hit:
                    stats[difficulty]["hits"] += 1
                    stats[difficulty]["mrr_sum"] += mrr

                # 进度日志 (每完成 10% 或 20条 打印一次)
                if processed_count % max_workers == 0 or processed_count == total_cases:
                    current_acc = stats["Total"]["hits"] / stats["Total"]["count"] if stats["Total"]["count"] else 0
                    logger.info(f"Progress: {processed_count}/{total_cases} | Current Total Hit Rate: {current_acc:.2%}")

        # --- 汇总与输出 ---
        logger.info("\n" + "=" * 60)
        logger.info(f"{'Category':<15} | {'Count':<8} | {'Hit Rate @'+str(top_k):<15} | {'MRR':<10}")
        logger.info("-" * 60)

        final_metrics = {}

        # 确保输出顺序：Total -> Easy -> Medium -> Hard -> Others
        categories = ["Total", "Easy", "Medium", "Hard"]
        #如果有其他未知的难度标签，也加进来
        existing_keys = list(stats.keys())
        for k in existing_keys:
            if k not in categories:
                categories.append(k)

        for cat in categories:
            if cat not in stats:
                continue
            
            data = stats[cat]
            count = data["count"]
            if count == 0:
                continue
                
            hit_rate = data["hits"] / count
            avg_mrr = data["mrr_sum"] / count
            
            # 存入返回结果
            final_metrics[cat] = {
                "hit_rate": round(hit_rate, 4),
                "mrr": round(avg_mrr, 4),
                "count": count
            }
            
            logger.info(f"{cat:<15} | {count:<8} | {hit_rate:.2%} | {avg_mrr:.4f}")

        logger.info("=" * 60 + "\n")
        
        return final_metrics
