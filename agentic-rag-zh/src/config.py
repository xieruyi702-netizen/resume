"""全局配置：路径、分块/检索参数、LLM 设置（.env 三行切换模型）。"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CRUD_DIR = DATA_DIR / "crud"                 # CRUD-RAG 原始数据
KB_DIR = DATA_DIR / "kb"                     # 构建产物
CHROMA_DIR = KB_DIR / "chroma"
PARENTS_PATH = KB_DIR / "parents.json"
BM25_PATH = KB_DIR / "bm25.pkl"

# 语料
CORPUS_PATH = DATA_DIR / "kb_docs.jsonl"     # 全量知识库文档 {doc_id, text}
GOLDEN_PATH = ROOT / "evals" / "golden.jsonl"
HISTORY_DIR = ROOT / "evals" / "history"

# 模型
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
USE_RERANK = os.getenv("USE_RERANK", "0") == "1"

# 分块（父子两段：child 检索、parent 喂 LLM）
PARENT_CHUNK = 1024
CHILD_CHUNK = 256
CHILD_OVERLAP = 32

# 检索
CANDIDATE_K = 20   # 每路召回数
TOP_K = 5          # 融合后返回块数
RRF_K = 60         # RRF 平滑常数（原论文经验值）

# LLM（OpenAI 兼容）
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
LLM_TEMPERATURE = 0.2
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

# Judge（LLM-as-Judge，默认用与被测不同源的 GLM，避免同源偏好）
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "glm-4-flash")

# Agent
MAX_AGENT_STEPS = 4
MAX_PLAN_STEPS = 3      # PAE：规划步数上限

# 上下文压缩（检索证据按预算裁剪；可用 --compress / 环境变量覆盖）
USE_COMPRESS = os.getenv("USE_COMPRESS", "0") == "1"
RETRIEVAL_BUDGET_CHARS = int(os.getenv("RETRIEVAL_BUDGET_CHARS", "3000"))
MEMORY_BUDGET_CHARS = int(os.getenv("MEMORY_BUDGET_CHARS", "600"))

# 失败记忆
MEMORY_COLLECTION = "failure_memory"
MEMORY_TOP_K = 2
MEMORY_MIN_SIM = 0.60   # 余弦相似度阈值，低于则不注入（防污染第一道闸）

# Chroma 里 child 向量 collection 固定名
KB_COLLECTION = "kb_children"
