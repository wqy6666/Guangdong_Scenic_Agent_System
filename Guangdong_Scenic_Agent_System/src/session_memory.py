# session_memory.py
from typing import Dict, List, Tuple, Optional
import tiktoken
MAX_ROUND = 5                # 短期记忆一最大轮数(user+assistant算一轮对话，2条message)
MAX_TOTAL_TOKENS = 1800

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

class SessionMemory:
    def __init__(self):
        # key:session_id, value:对话消息列表 [{"role":"user","content":"xxx"}, ...]
        self.session_store: Dict[str, List[Dict]] = {}
        # 新增：保存每个session本轮滑动窗口溢出出来的消息，每次触发窗口后读取清空
        self._session_overflow_buffer: Dict[str, List[Dict]] = {}

    def get_history(self, session_id: str) -> List[Dict]:
        """【原有接口，完全保留】获取当前内存内滑动窗口的历史（送入LLM的最近上下文）"""
        if session_id not in self.session_store:
            self.session_store[session_id] = []
        return self.session_store[session_id].copy()

    def add_turn(self, session_id: str, user_text: str, ai_text: str):
        """【原有接口，完全保留】增加一轮：user+assistant"""
        self.add_message(session_id, "user", user_text)
        self.add_message(session_id, "assistant", ai_text)

    def add_message(self, session_id: str, role: str, content: str):
        """【原有接口，完全保留】添加单条消息，添加后自动执行滑动窗口挤出逻辑，溢出存入内部buffer"""
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        self.session_store[session_id] = history
        # 执行滑动窗口，溢出的消息存入本session的buffer
        overflow = self._slide_window(session_id)
        if session_id not in self._session_overflow_buffer:
            self._session_overflow_buffer[session_id] = []
        self._session_overflow_buffer[session_id].extend(overflow)

    def pop_overflow(self, session_id: str) -> List[Dict]:
        """【新增对外接口】取出当前session所有被挤出的消息，取出后清空buffer；上层调用交给short2磁盘存储"""
        buf = self._session_overflow_buffer.get(session_id, [])
        self._session_overflow_buffer[session_id] = []
        return buf

    def clear_session(self, session_id: str):
        """【原有接口，完全保留】清除会话内存+溢出缓存"""
        if session_id in self.session_store:
            del self.session_store[session_id]
        if session_id in self._session_overflow_buffer:
            del self._session_overflow_buffer[session_id]

    def _calc_total_token(self, msg_list: List[Dict]) -> int:
        total = 0
        for m in msg_list:
            total += count_tokens(m["content"])
        return total

    def _slide_window(self, session_id: str) -> List[Dict]:
        """
        内部滑动窗口逻辑
        约束：消息列表总消息数 <= MAX_ROUND*2；同时总token <= MAX_TOTAL_TOKENS
        注意：一轮对话 = user + assistant，2条message；MAX_ROUND=5轮最多内存放10条message
        返回：被挤出去的旧消息列表
        """
        msg_list = self.session_store[session_id]
        overflow = []
        max_msg_count = MAX_ROUND * 2
        while len(msg_list) > max_msg_count or self._calc_total_token(msg_list) > MAX_TOTAL_TOKENS:
            if not msg_list:
                break
            popped = msg_list.pop(0)
            overflow.append(popped)
        self.session_store[session_id] = msg_list
        return overflow

# 全局单例，和你原来一模一样，上层完全不用修改 import
memory_manager = SessionMemory()
