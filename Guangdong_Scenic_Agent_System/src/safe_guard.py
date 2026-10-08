# src/safe_guard.py
import logging
import os
import re
import functools
from datetime import datetime
from pathlib import Path
import tiktoken
from service_errors import LLMServiceError

# ====================== 日志配置 ======================
LOG_DIR = Path(__file__).parent.parent / "logs"
os.makedirs(LOG_DIR, exist_ok=True)

AGENT_LOGGER_NAME = "travel_agent"

class DailyAgentLogHandler(logging.FileHandler):
    """每日独立日志文件：logs/YYYYMMDD_agent.log，跨天后首条日志自动切换到新文件"""

    def __init__(self, log_dir, encoding: str = "utf-8"):
        self._log_dir = Path(log_dir)
        self._today = datetime.now().strftime("%Y%m%d")
        super().__init__(str(self._log_dir / f"{self._today}_agent.log"), encoding=encoding, delay=True)

    def emit(self, record):
        today = datetime.now().strftime("%Y%m%d")
        if today != self._today:
            self._today = today
            if self.stream:
                self.stream.close()
                self.stream = None
            self.baseFilename = os.fspath(self._log_dir / f"{self._today}_agent.log")
        super().emit(record)


def setup_logger(debug: bool = False):
    """初始化全局日志（每日独立日志文件），程序启动仅调用一次"""
    logger = logging.getLogger(AGENT_LOGGER_NAME)
    if logger.handlers:
        return logger
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    logger.propagate = False

    fmt_str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    console_fmt = logging.Formatter(fmt_str, datefmt=date_fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # 每日一个独立日志文件，跨天自动切新文件，不再集中写 run.log
    file_handler = DailyAgentLogHandler(LOG_DIR)
    file_handler.setFormatter(logging.Formatter(fmt_str, datefmt=date_fmt))
    logger.addHandler(file_handler)
    return logger

# 全局logger实例
logger = setup_logger(debug=False)

def set_log_debug():
    logger.setLevel(logging.DEBUG)

def set_log_info():
    logger.setLevel(logging.INFO)

# ====================== 安全工具函数 ======================
def safe_truncate(text: str, max_len: int = 60) -> str:
    """日志打印安全截断，防止超长文本刷屏"""
    if not text:
        return ""
    s = str(text)
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s

@functools.lru_cache(maxsize=1)
def _token_encoder():
    return tiktoken.get_encoding("cl100k_base")


def safe_count_tokens(text: str) -> int:
    """token计算，自带异常兜底，出错返回0"""
    try:
        return len(_token_encoder().encode(text))
    except Exception as e:
        logger.warning(f"safe_count_tokens error: {str(e)}")
        return 0

# ====================== 通用异常捕获装饰器 ======================
def safe_call(default_ret, log_ctx: str = ""):
    """
    装饰器：捕获函数全部运行时异常，打印日志，返回default_ret兜底值，不向上抛异常
    :param default_ret: 发生异常时候返回的兜底对象
    :param log_ctx: 业务上下文标识，日志用于区分是哪个业务
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except LLMServiceError:
                raise
            except Exception as e:
                ctx_info = f"[{log_ctx}]" if log_ctx else ""
                logger.error(
                    f"{ctx_info} {func.__name__} raise exception: {type(e).__name__}",
                    exc_info=True
                )
                return default_ret
        return wrapper
    return decorator

# ---------------- 预定义常用兜底常量，业务直接引用 ----------------
SAFE_EMPTY_LIST = []
SAFE_EMPTY_DICT = {}
SAFE_EMPTY_STR = ""

# 结构化查询单景区兜底字典
SAFE_SPOT_DETAIL_DICT = {
    "spot_name": "",
    "address": "暂无地址信息",
    "ticket": "暂无门票信息",
    "open_time": "暂无开放时间信息",
    "city": None,
    "level": "暂无景区等级信息",
    "telephone": "暂无联系电话",
}

# ====================== 输出格式边界（确定性净化） ======================
def sanitize_answer_text(text: str) -> str:
    """
    最终输出净化：剥离LLM习惯性输出的markdown符号，保证控制台/记忆/协议统一为纯文本。
    提示词里的格式约束是概率性的，这里是确定性的兜底边界。
    """
    if not text:
        return text
    s = str(text)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)                 # **粗体**
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", s)      # *行内强调*
    s = re.sub(r"^\s*#{1,6}\s+", "", s, flags=re.MULTILINE)  # 标题井号
    s = s.replace("`", "")                                   # 代码标记
    s = re.sub(r"\n{3,}", "\n\n", s)                         # 压缩多余空行
    return s.strip()
