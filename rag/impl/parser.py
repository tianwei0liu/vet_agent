import os
import json
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from rag.interface.base_parser import BaseParser
from rag.schema.pet_record import PetRecord

# Retry logic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

# LangChain Imports
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

logger = logging.getLogger(__name__)

class Parser(BaseParser):
    def __init__(
        self,
        model_name: str = "deepseek-chat",
        temperature: float = 0,
        batch_size: int = 20, # 保持较小的 batch size 以保证精度
        max_workers: int = 20,
    ):
        """
        初始化解析器，使用 LangChain 组件，支持并发和重试
        """
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Environment variable DEEPSEEK_API_KEY not found.")

        # 1. 初始化 LLM
        self.llm = ChatDeepSeek(
            model=model_name,
            api_key=api_key, # Langchain DeepSeek 使用 api_key 参数
            temperature=temperature,
            # 强制 JSON 模式
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        self.batch_size = batch_size
        self.max_workers = max_workers
        self.chain = self._build_chain()

    def _build_chain(self):
        """
        构建 LCEL 处理链
        """
        system_template = """
        You are a veterinary data structuring expert.
        Your task is to extract structured information from the provided user observations.

        ### Taxonomy Rules (CRITICAL):
        You must classify the 'species' into one of the following 8 EXACT categories (all lowercase):
        1. "dog" (includes puppy, cocker spaniel, etc.)
        2. "cat" (includes kitten, siamese cat, etc.)
        3. "rabbit"
        4. "ferret"
        5. "guinea_pig" (use underscore instead of space for guinea pig)
        6. "hamster"
        7. "bird" (includes cockatiel, canary, budgie, etc.)
        8. "unknown" (use this if the species is not specified or generic like "pet")

        ### Output Requirements:
        1. Return a valid JSON object with a single key "results", containing a list of objects.
        2. Each object must contain:
            - "id": The original ID from input (Must match exactly).
            - "species": One of the 8 categories above (lowercase).
            - "specific_breed": The specific breed or animal type mentioned (e.g., "puppy", "cockatiel", "canary", "kitten"). 
             * IMPORTANT: If no specific breed is mentioned, REPEAT the value from 'species'. 
             * MUST be lowercase.
            - "symptom_keywords": A list of standardized medical keywords (e.g., "vomiting", "lethargy"). All lowercase.

        ### Examples:
        Input: "My kitten is vomiting."
        Output: {{"id": 1, "species": "cat", "specific_breed": "kitten", "symptom_keywords": ["vomiting"]}}

        Input: "The hamster looks sad."
        Output: {{"id": 2, "species": "hamster", "specific_breed": "hamster", "symptom_keywords": ["depression"]}}

        Input: "My pet is not eating."
        Output: {{"id": 3, "species": "unknown", "specific_breed": "unknown", "symptom_keywords": ["anorexia"]}}

        Do not miss any records. If {batch_size} records are input, {batch_size} records must be output.
        """
        human_template = "Input Data (JSON):\n{batch_data}"

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", human_template)]
        )

        parser = JsonOutputParser()

        # 构建链：Prompt -> LLM -> JSON Parser
        return prompt | self.llm | parser

    # --- 核心升级：增加重试装饰器 ---
    @retry(
        stop=stop_after_attempt(3), # 最多重试3次
        wait=wait_random_exponential(min=1, max=10), # 指数退避
        reraise=True
    )
    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        处理单个批次，带有重试机制
        """
        try:
            if len(batch) == 0:
                return []

            batch_ids = [item['id'] for item in batch]
            
            # 将 batch 转换为 json string
            batch_json_str = json.dumps(batch, ensure_ascii=False)

            # 调用链，传入 batch_size 用于 prompt 提示
            response = self.chain.invoke({
                "batch_data": batch_json_str,
                "batch_size": len(batch)
            })

            results = response.get("results", [])

            # 【关键校验】输入输出数量必须一致，否则视为失败，触发重试
            if len(results) != len(batch):
                logger.warning(f"Batch size mismatch! Input: {len(batch)}, Output: {len(results)}. Retrying...")
                raise ValueError("Output count mismatch")
            
            validated_results = []
            input_map = {item['id']: item['text'] for item in batch}
            for item in results:
                original_text = input_map.get(item['id'])
                record_data = {
                    "id": item['id'],
                    "text": original_text,
                    "species": item.get("species"),
                    "specific_breed": item.get("specific_breed"),
                    "symptom_keywords": item.get("symptom_keywords")
                }
                # data validation
                record = PetRecord(**record_data)
                validated_results.append(record.model_dump())
            if len(validated_results) != len(batch):
                logger.warning(f"Batch size mismatch! Input: {len(batch)}, Validated: {len(validated_results)}. Retrying...")
                raise ValueError("Validation count mismatch")
            return results

        except Exception as e:
            logger.error(f"Error processing batch {batch_ids[0]} - {batch_ids[-1]}: {e}")
            raise e # 抛出异常以触发 tenacity 重试

    def load_raw_data(self, input_path: str) -> pd.DataFrame:
        """
        读取 CSV 并预处理
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_csv(input_path)

        required_cols = ["text", "condition", "record_type"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV missing required columns: {required_cols}")
        
        # 1. 过滤 Owner Observation
        df_owner = df[df['record_type'].str.strip() == 'Owner Observation'].copy()
        
        # 2. 生成唯一 ID (锚点)
        if "id" not in df_owner.columns:
            df_owner["id"] = range(len(df_owner))
        
        logger.info(f"Loaded raw data. Filtered {len(df_owner)} 'Owner Observation' records from {len(df)} total.")
        return df_owner

    def parse(
        self,
        input_path: str,
        output_path: str,
    ) -> str:
        """
        核心 Pipeline 方法 (并行版)
        """
        logger.info(f"--- [Parser] Starting pipeline on {input_path} ---")

        # 1. Load
        df = self.load_raw_data(input_path)
        # 只取需要的字段发送给 LLM
        records = df[["id", "text"]].to_dict(orient="records")
        
        # 切分 Batches
        batches = [records[i : i + self.batch_size] for i in range(0, len(records), self.batch_size)]
        total_batches = len(batches)
        
        logger.info(f"Processing {len(records)} records in {total_batches} batches with {self.max_workers} threads...")

        all_results = []
        
        # 2. Extract (Parallel Processing)
        # 使用 ThreadPoolExecutor 进行并发请求
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_batch = {executor.submit(self._process_batch, batch): batch for batch in batches}
            
            # 使用 tqdm 监控进度
            for future in tqdm(as_completed(future_to_batch), total=total_batches, desc="Parallel Extraction"):
                try:
                    batch_result = future.result()
                    all_results.extend(batch_result)
                except Exception as e:
                    # 如果重试多次后依然失败，记录错误但不中断整个流程 (或者根据需求抛出异常)
                    logger.critical(f"A batch failed after retries: {e}")
                    # 在生产环境中，这里应该把失败的 batch ID 写入 fail_log

        # 3. Merge & Format
        if not all_results:
            raise RuntimeError("Extraction failed. No data returned from LLM.")

        extracted_df = pd.DataFrame(all_results)
        
        # 确保 ID 类型一致
        extracted_df["id"] = extracted_df["id"].astype(int)

        # 合并原始数据 (Inner Join 确保只有成功提取的数据才被保留)
        final_df = pd.merge(df, extracted_df, on="id", how="inner", validate="one_to_one")

        # 4. Save as JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_json(output_path, orient="records", indent=2, force_ascii=False)

        logger.info(f"--- [Parser] Success! Processed {len(final_df)} records. Data saved to {output_path} ---")
        return output_path
