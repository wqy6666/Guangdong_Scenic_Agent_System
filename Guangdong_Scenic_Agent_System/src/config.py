# config.py
from pathlib import Path
import torch
import os

# =========项目根路径=========
ROOT_PATH = Path(__file__).parent.parent

# =========文件路径配置=========
VECTOR_FILE = str(ROOT_PATH / "src" / "vector_array.npy")
VECTOR_NPY_FILE = VECTOR_FILE
META_FILE = str(ROOT_PATH / "src" / "chunk_meta.pkl")

DB_PATH = str(ROOT_PATH / "shujuku" / "travel_final2.db")
DB_TABLE_NAME = "scenic_spots"

# =========模型路径=========
EMBED_MODEL_PATH = str(ROOT_PATH / "model" / "bge-small-zh-v1.5")
RERANKER_MODEL_PATH = str(ROOT_PATH / "model" / "bge-reranker-base")

# =========检索超参数=========
TOP_K_VECTOR = 20
TOP_K_KEYWORD = 10
RERANK_TOP = 5

# =========LLM模板【仅此一处】，无真实密钥=========
DEFAULT_LLM_TEMPLATE = {
    "api_key": "",
    "base_url": "https://api.deepseek.com",
    "model_name": "deepseek-v4-flash"
}

# =========管理员账户名：该用户名的账户拥有管理后台权限，API密钥与普通用户一样自行在界面配置=========
ADMIN_USERNAME = "admin"

# Public deployment switches are opt-in so existing local/LAN usage keeps working.
PUBLIC_MODE = os.environ.get("SCENIC_PUBLIC_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}

# =========推理设备=========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 推理设备：{DEVICE}")
