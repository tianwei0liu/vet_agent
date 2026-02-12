import json
import os
import random
import logging
import concurrent.futures  # <--- 新增引入
from typing import List, Dict, Any, Optional

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage

from rag.interface.base_query_generator import BaseQueryGenerator

logger = logging.getLogger(__name__)

class QueryGenerator(BaseQueryGenerator):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "deepseek-chat"):
        # 1. 获取 API Key
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API Key is missing. Set DEEPSEEK_API_KEY env var.")
        
        # 2. 初始化 LangChain Chat Model
        self.llm = ChatDeepSeek(
            api_key=self.api_key,
            model=model_name,
            temperature=0.7,
            model_kwargs={"response_format": {"type": "json_object"}} 
        )

    def _call_llm(self, record: Dict) -> List[Dict[str, str]]:
        """
        私有方法：调用 DeepSeek 生成 3 个不同难度的 Query
        """
        # ... (Prompt 内容保持你提供的版本不变) ...
        prompt_content = f"""
        You are an AI data generator simulating diverse pet owners seeking online veterinary advice. 
        Your task is to generate 3 distinct search queries based on the provided Ground Truth Pet Record.
        
        ### The Ground Truth Record
        - Species: {record.get('species')}
        - Specific Breed: {record.get('specific_breed')}
        - Condition/Diagnosis: {record.get('condition')}
        - Key Symptoms: {record.get('symptom_keywords')}
        - Owner's Observation: "{record.get('text')}"

        ### Task Instructions
        Generate 3 search queries in English that a user might type into a search bar. The queries must strictly correspond to the Ground Truth Record but vary in linguistic style and difficulty.

        ### Difficulty Definitions & Examples
        
        **1. Easy (Keyword-centric)**
        - Style:  Standard sentence, describing the symptoms clearly.
        - Goal: Test Sparse Retriever (SPLADE/BM25).
        - Example (for "The rabbit has digestive issue"): "My rabbit has diarrhea."

        **2. Medium (Natural Language Description)**
        - Style: Standard sentence, describing the observation clearly without medical jargon. NO explicit medical terms from the 'condition' or 'symptom_keywords' field.
        - Goal: Test Dense Retriever (BGE) semantic understanding.
        - Example (for "The rabbit has digestive issue"): "My rabbit has runny droppings and looks listless."

        **3. Hard (Implicit & Noisy - The Challenge)**
        - Style: Emotional, indirect, uses slang or colloquiums, or describes the *effect* of the symptom rather than the symptom itself. NO explicit medical terms from the 'condition' or 'symptom_keywords' field.
        - Goal: Test Reranker & Reasoner capabilities.
        - Example (for "The rabbit has digestive issue"): "Bunny’s got a bit of a poopy butt and he looks really hunched up and out of it. Is this bad?" 
          *(Note: Uses "Bunny" instead of Rabbit, "poopy butt" instead of diarrhea, implies Trauma)*

        ### Output Format
        You must output a strictly valid JSON object. Do not output markdown code blocks.
        
        {{
            "queries": [
                {{
                    "difficulty": "Easy",
                    "query": "..."
                }},
                {{
                    "difficulty": "Medium",
                    "query": "..."
                }},
                {{
                    "difficulty": "Hard",
                    "query": "..."
                }}
            ]
        }}
        """

        messages = [
            SystemMessage(content="You are a helpful data generation assistant. You must output valid JSON."),
            HumanMessage(content=prompt_content)
        ]

        try:
            # LangChain Invoke (这里是线程安全的)
            response = self.llm.invoke(messages)
            content = response.content
            
            # --- 鲁棒性清洗 ---
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            if isinstance(result, dict) and "queries" in result:
                return result["queries"]
            elif isinstance(result, list):
                return result
            else:
                logger.warning(f"Unexpected JSON structure for ID {record.get('id')}")
                return []

        except Exception as e:
            logger.error(f"LLM generation failed for ID {record.get('id')}: {e}")
            return []

    def generate_test_set(self, source_file: str, sample_size: int = 50, output_file: str = "./rag/data/generated_test_set.json", max_workers: int = 20) -> None:
        """
        生成测试集（并行版）
        :param max_workers: 并发线程数。建议设置为 5-10，取决于你的 API Rate Limit。
        """
        logger.info(f"Loading source data from {source_file}...")
        
        path = source_file
        if not os.path.exists(path):
            logger.error(f"Source file not found: {path}")
            return

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 随机采样
        real_sample_size = min(len(data), sample_size)
        sampled_records = random.sample(data, real_sample_size)
        
        logger.info(f"Sampled {real_sample_size} records. Starting PARALLEL generation with {max_workers} workers...")
        
        eval_dataset = []
        
        # --- 并行执行核心逻辑 ---
        # 使用 ThreadPoolExecutor 管理线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 1. 提交任务：建立 Future 到 Record 的映射，以便后续追踪是谁的结果
            future_to_record = {
                executor.submit(self._call_llm, record): record 
                for record in sampled_records
            }
            
            # 2. 获取结果：as_completed 会在任务完成时立刻 yield，而不是按顺序等待
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_record):
                record = future_to_record[future]
                completed_count += 1
                
                try:
                    queries = future.result() # 获取返回值，如果线程抛出异常这里会 raise
                    
                    # 将结果加入数据集
                    for q in queries:
                        eval_dataset.append({
                            "id": record["id"],
                            "species": record.get("species"),
                            "original_text": record.get("text"),
                            "query": q.get("query"),
                            "difficulty": q.get("difficulty", "Unknown")
                        })
                    
                    # 进度日志
                    if completed_count % max_workers == 0:
                        logger.info(f"Progress: {completed_count}/{real_sample_size} records processed...")
                        
                except Exception as e:
                    logger.error(f"Worker exception for record ID {record.get('id')}: {e}")
        
        # --- 保存结果 ---
        # 按 ID 排序一下，因为并行执行后顺序会乱
        eval_dataset.sort(key=lambda x: x['id'])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_dataset, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Successfully generated {len(eval_dataset)} test cases. Saved to {output_file}")
