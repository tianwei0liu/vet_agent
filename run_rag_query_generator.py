import os
import logging
from rag.impl.query_generator import QueryGenerator

# os.environ["LANGCHAIN_TRACING_V2"] = "false"
# os.environ["LANGCHAIN_API_KEY"] = ""

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # 配置你的数据路径
    SOURCE_DATA = "./rag/data/pet_health_symptoms_dataset.json" # 请根据实际路径修改
    TEST_SET_FILE = "./rag/data/generated_test_set.json"
    
    # --- 1. 生成数据 (需要设置 DEEPSEEK_API_KEY) ---
    generator = QueryGenerator()
    generator.generate_test_set(
        source_file=SOURCE_DATA,
        sample_size=200, 
        output_file=TEST_SET_FILE,
        max_workers = 20
    )
    