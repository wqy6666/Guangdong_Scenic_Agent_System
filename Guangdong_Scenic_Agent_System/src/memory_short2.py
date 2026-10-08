# src/memory_short2.py
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from safe_guard import safe_call, logger, SAFE_EMPTY_LIST, safe_count_tokens
STORE_DIR = "./session_short2_store"
RECALL_RATIO_LIMIT = 0.2  # 召回内容上限：总上下文token的 1/5
os.makedirs(STORE_DIR, exist_ok=True)

class Short2IndexItem:
    """内存只保存轻量索引元数据，完整问答保存在磁盘jsonl文件"""
    def __init__(self, memory_id: str, session_id: str, create_dt: str, file_name: str, user_snippet: str):
        self.memory_id = memory_id
        self.session_id = session_id
        self.create_dt = create_dt
        self.file_name = file_name
        self.user_snippet = user_snippet  # 摘要用于调试，不用于LLM

class MemoryShort2:
    def __init__(self):
        self.memory_index: Dict[str, Short2IndexItem] = dict()

    def _get_day_file(self) -> str:
        dt_str = datetime.now().strftime("%Y%m%d")
        return os.path.join(STORE_DIR, f"short2_{dt_str}.jsonl")

    @safe_call(default_ret=None, log_ctx="Short2Memory-save_overflow_round")
    def save_overflow_round(self, session_id: str, round_item: Dict):
        """保存单条被短期记忆一挤出的对话轮次"""
        mem_id = str(uuid.uuid4())
        dt_now = datetime.now().isoformat()
        file_path = self._get_day_file()
        record = {
            "memory_id": mem_id,
            "session_id": session_id,
            "create_time": dt_now,
            "round_data": round_item
        }
        # jsonl 追加一行
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        snippet = round_item.get("content", "")[:120]
        idx_item = Short2IndexItem(
            memory_id=mem_id,
            session_id=session_id,
            create_dt=dt_now,
            file_name=os.path.basename(file_path),
            user_snippet=snippet
        )
        self.memory_index[mem_id] = idx_item

    def save_batch_overflow(self, session_id: str, round_list: List[Dict]):
        """批量保存多条溢出记录"""
        for item in round_list:
            self.save_overflow_round(session_id, item)

    @safe_call(default_ret=SAFE_EMPTY_LIST, log_ctx="Short2Memory-recall_by_session")
    def recall_by_session(self, session_id: str, total_context_token: int) -> List[Dict]:
        """
        按会话召回溢出历史
        :param session_id:会话ID
        :param total_context_token:当前常驻上下文总token，用于计算1/5上限
        :return:召回的完整对话轮次列表，严格token截断
        """
        candidate_idx = [v for v in self.memory_index.values() if v.session_id == session_id]
        if not candidate_idx:
            return []
        max_allow_token = int(total_context_token * RECALL_RATIO_LIMIT)
        collected: List[Dict] = []
        used = 0
        for idx in candidate_idx:
            file_path = os.path.join(STORE_DIR, idx.file_name)
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("memory_id") == idx.memory_id:
                        raw_round = obj["round_data"]
                        round_text = json.dumps(raw_round, ensure_ascii=False)
                        item_token = safe_count_tokens(round_text)
                        if used + item_token > max_allow_token:
                            return collected
                        collected.append(raw_round)
                        used += item_token
                        break
        return collected

short2_memory = MemoryShort2()
