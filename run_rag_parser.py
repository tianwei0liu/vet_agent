import os
# os.environ["LANGCHAIN_TRACING_V2"] = "false"
# os.environ["LANGCHAIN_API_KEY"] = ""
import json
import logging
from rag.interface.base_parser import BaseParser
from rag.impl.parser import Parser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 模拟在 RAG Pipeline 中的调用 ---

if __name__ == "__main__":
    # 配置路径
    INPUT_FILE = os.path.join(os.getcwd(), "rag/data/pet-health-symptoms-dataset.csv")
    OUTPUT_FILE = os.path.join(os.getcwd(), "rag/data/pet_health_symptoms_dataset.json")
    try:
        # 1. 实例化 Parser
        parser = Parser(batch_size=20, max_workers=20)

        # 2. 运行解析
        json_path = parser.parse(input_path=INPUT_FILE, output_path=OUTPUT_FILE)

        # 3. 验证输出
        with open(OUTPUT_FILE, "r") as f:
            data = json.load(f)
            logging.info(f"\npreview first record:\n{json.dumps(data[0], indent=2)}")

    except Exception as e:
        logging.critical(f"\nPipeline Execution Failed: {e}")
    