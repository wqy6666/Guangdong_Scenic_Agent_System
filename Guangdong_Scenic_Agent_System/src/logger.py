import logging
import os
from pathlib import Path
# src/logger.py
from safe_guard import logger

__all__ = ["logger"]

# 创建日志对象
logger = logging.getLogger("travel_agent")
logger.setLevel(logging.INFO)
# 禁止向上传递日志，避免父logger二次打印
logger.propagate = False

# 防止重复添加handler
if not logger.handlers:
    #控制台输出
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    #文件日志输出
    log_path = Path(__file__).parent.parent / "logs" / "run.log"
    os.makedirs(log_path.parent,exist_ok=True)
    file_handler = logging.FileHandler(log_path,encoding="utf‑8")
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
