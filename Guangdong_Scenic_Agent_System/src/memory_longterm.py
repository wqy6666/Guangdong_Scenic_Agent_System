# src/memory_longterm.py
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict
from safe_guard import safe_call, logger, SAFE_EMPTY_LIST, safe_count_tokens
LONGTERM_FILE = "./long_term_memory.json"
RECALL_RATIO_LIMIT = 0.25  # 召回最多占总上下文 1/4

if not os.path.exists(LONGTERM_FILE):
    with open(LONGTERM_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False)

class LongTermMemory:
    def __init__(self):
        self.store: List[Dict] = []
        self._load()

    def _load(self):
        try:
            with open(LONGTERM_FILE, "r", encoding="utf-8") as f:
                self.store = json.load(f)
        except Exception as e:
            logger.error(f"[LongTermMemory] long_term.json加载失败，重置为空列表 err={str(e)}", exc_info=True)
            self.store = []

    def _persist(self):
        try:
            with open(LONGTERM_FILE, "w", encoding="utf-8") as f:
                json.dump(self.store, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[LongTermMemory] _persist保存失败 err={str(e)}", exc_info=True)

    @safe_call(default_ret=None, log_ctx="LongTermMemory-save_memory")
    def save_memory(self, session_id: str, source_query: str, content: str, tag: List[str]):
        """保存一条规划/方案长期记忆"""
        mem = {
            "mem_id": str(uuid.uuid4()),
            "session_id": session_id,
            "create_time": datetime.now().isoformat(),
            "source_query": source_query,
            "content": content,
            "tag": tag,
            "invalid": False
        }
        self.store.append(mem)
        self._persist()

    def get_valid_memory(self) -> List[Dict]:
        """获取全部未失效记忆，上层在这里做embedding相似度排序"""
        return [x for x in self.store if x["invalid"] is False]

    @safe_call(default_ret=SAFE_EMPTY_LIST, log_ctx="LongTermMemory-recall_memory")
    def recall_memory(self, sorted_candidates: List[Dict], total_context_token: int) -> List[Dict]:
        """
        接收上层已经做过相似度排序的记忆列表，执行token上限截断
        :param sorted_candidates:上层emb相似度排序完成的记忆列表
        :param total_context_token:当前上下文总token，计算1/4上限
        :return:截断后的记忆片段
        """
        max_allow = int(total_context_token * RECALL_RATIO_LIMIT)
        collected: List[Dict] = []
        used = 0
        for item in sorted_candidates:
            piece_token = safe_count_tokens(item["content"])
            if used + piece_token > max_allow:
                break
            collected.append(item)
            used += piece_token
        return collected

longterm_memory = LongTermMemory()
