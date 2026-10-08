import os
import json
from typing import List, Dict, Optional
from collections import OrderedDict
from file_store import write_json

# ===================== 常量预设，后续可微调 =====================
GLOBAL_MEM_CHAR_BASE_LIMIT = 6000
GLOBAL_MEM_CHAR_FLOAT_BUFFER = 1200
GLOBAL_MEM_MIN_RESERVE_PARAGRAPH = 3
PARAGRAPH_SEP = "====SEG_PARA===="
SHORT2_INDEX_FILENAME = "short2_index.json"
GLOBAL_MEM_FILENAME = "global_memory.txt"
LONG_MEM_FILENAME = "long_memory.json"
PARAS_PER_PREV_LOAD = 6   # 上拉向前翻阅每次加载6条
# ==============================================================


class LruGlobalCache:
    """全局记忆内存LRU缓存，按字符计数，最小保留最近N段"""
    def __init__(self, base_limit: int, float_buf: int, min_reserve: int):
        self.base_limit = base_limit
        self.float_buf = float_buf
        self.max_allow = base_limit + float_buf
        self.min_reserve = min_reserve
        # OrderedDict 维护访问时序 key: para_id, value: {"text": str, "access_ts": float}
        self.cache = OrderedDict()

    def get_total_chars(self) -> int:
        total = 0
        for item in self.cache.values():
            total += len(item["text"])
        return total

    def get_para(self, para_id: int) -> Optional[str]:
        if para_id in self.cache:
            # 更新访问时间，移到末尾（最近访问）
            val = self.cache.pop(para_id)
            import time
            val["access_ts"] = time.time()
            self.cache[para_id] = val
            return val["text"]
        return None

    def add_para(self, para_id: int, text: str):
        import time
        self.cache[para_id] = {
            "text": text,
            "access_ts": time.time()
        }
        # LRU淘汰逻辑
        self._prune()

    def _prune(self):
        """超限执行淘汰，至少保留最近N段"""
        while self.get_total_chars() > self.max_allow and len(self.cache) > self.min_reserve:
            # 弹出最久未访问
            old_para_id, _ = self.cache.popitem(last=False)
            print(f"[LRU淘汰] 段落{old_para_id}移出内存缓存")


class SessionMemoryManager:
    def __init__(self, session_folder: str):
        self.session_folder = session_folder
        self.global_mem_path = os.path.join(session_folder, GLOBAL_MEM_FILENAME)
        self.short2_index_path = os.path.join(session_folder, SHORT2_INDEX_FILENAME)
        self.long_mem_path = os.path.join(session_folder, LONG_MEM_FILENAME)

        # LRU实例
        self.lru_cache = LruGlobalCache(
            base_limit=GLOBAL_MEM_CHAR_BASE_LIMIT,
            float_buf=GLOBAL_MEM_CHAR_FLOAT_BUFFER,
            min_reserve=GLOBAL_MEM_MIN_RESERVE_PARAGRAPH
        )
        # 加载短2索引
        self.short2_index: List[Dict] = self._load_short2_index()

        # --------长期记忆：懒加载，会话初始化不读磁盘--------
        self._long_mem_loaded = False
        self._long_mem_data: Dict = {
            "hobby": None,
            "price": None,
            "time": None,
            "spot_type": None
        }

    @staticmethod
    def escape_sep(text: str) -> str:
        """分隔符转义，写入全局记忆前调用"""
        return text.replace(PARAGRAPH_SEP, "__SEG_ESCAPE__")

    @staticmethod
    def unescape_sep(text: str) -> str:
        """读取文本后还原"""
        return text.replace("__SEG_ESCAPE__", PARAGRAPH_SEP)

    def _load_short2_index(self) -> List[Dict]:
        """加载短期记忆2索引，轻量校验，标记坏索引"""
        if not os.path.exists(self.short2_index_path):
            return []
        with open(self.short2_index_path, "r", encoding="utf-8") as f:
            index_list = json.load(f)
        # 获取全局记忆最大段落编号
        max_para_id = self.get_max_paragraph_id()
        for item in index_list:
            pid = item["para_id"]
            if pid > max_para_id:
                item["is_bad"] = True
            else:
                if "is_bad" not in item:
                    item["is_bad"] = False
        return index_list

    def get_max_paragraph_id(self) -> int:
        """获取全局记忆最大段落编号"""
        if not os.path.exists(self.global_mem_path):
            return -1
        with open(self.global_mem_path, "r", encoding="utf-8") as f:
            raw = f.read()
        paragraphs = raw.split(PARAGRAPH_SEP)
        valid = [p.strip() for p in paragraphs if p.strip()]
        if len(valid) == 0:
            return -1
        return len(valid) - 1

    def append_dialogue_paragraph(self, user_q: str, agent_a: str, keywords: List[str]):
        """
        新增一轮对话段落
        写入顺序：先全局记忆磁盘，成功后追加短2索引
        """
        content = f"用户：{user_q}\n助手：{agent_a}"
        escaped_content = self.escape_sep(content)
        para_text = escaped_content + PARAGRAPH_SEP
        # 1.写入全局记忆
        with open(self.global_mem_path, "a", encoding="utf-8") as f:
            f.write(para_text)
            f.flush()
            os.fsync(f.fileno())
        # 获取新段落id
        new_para_id = self.get_max_paragraph_id()
        # 2.追加索引记录
        new_index_item = {
            "para_id": new_para_id,
            "keywords": keywords,
            "is_bad": False
        }
        self.short2_index.append(new_index_item)
        # 保存索引文件
        write_json(self.short2_index_path, self.short2_index)
        # 加载到LRU缓存
        self.lru_cache.add_para(new_para_id, content)

    def read_paragraph_by_id(self, para_id: int) -> Optional[str]:
        """按段落编号读取段落，优先LRU缓存，没有就读磁盘"""
        cache_res = self.lru_cache.get_para(para_id)
        if cache_res is not None:
            return cache_res
        # 磁盘读取
        if not os.path.exists(self.global_mem_path):
            return None
        with open(self.global_mem_path, "r", encoding="utf-8") as f:
            raw = f.read()
        para_list = raw.split(PARAGRAPH_SEP)
        valid_para = [self.unescape_sep(p.strip()) for p in para_list if p.strip()]
        if para_id < 0 or para_id >= len(valid_para):
            return None
        para_text = valid_para[para_id]
        # 载入LRU缓存，触发淘汰
        self.lru_cache.add_para(para_id, para_text)
        return para_text

    def search_short2_index(self, query_keywords: List[str]) -> List[Dict]:
        """短期记忆2检索：关键词匹配，返回非坏索引列表"""
        hit = []
        for idx_item in self.short2_index:
            if idx_item["is_bad"]:
                continue
            # 简单关键词交集匹配，后续可优化检索逻辑
            item_kw = set(idx_item["keywords"])
            q_kw = set(query_keywords)
            if item_kw & q_kw:
                hit.append(idx_item)
        return hit

    def load_recent_paras(self, count: int = 5) -> List[str]:
        """打开会话时，预加载最新N段用于前端展示"""
        max_pid = self.get_max_paragraph_id()
        res = []
        start_id = max(0, max_pid - count + 1)
        for pid in range(start_id, max_pid + 1):
            p_text = self.read_paragraph_by_id(pid)
            if p_text:
                res.append(p_text)
        return res

    def load_prev_paras(self, oldest_show_para_id: int) -> tuple[List[str], List[int]]:
        """
        上拉向前翻阅历史对话
        :param oldest_show_para_id: 当前界面已经展示的最旧段落编号
        :return: (文本列表,对应段落id列表),无更多返回两个空list
        """
        max_pid = self.get_max_paragraph_id()
        if max_pid < 0:
            return [], []
        end = oldest_show_para_id - 1
        start = end - (PARAS_PER_PREV_LOAD - 1)
        if start < 0:
            start = 0
        if end < 0:
            return [], []
        para_texts = []
        para_ids = []
        for pid in range(start, end + 1):
            txt = self.read_paragraph_by_id(pid)
            if txt is not None:
                para_texts.append(txt)
                para_ids.append(pid)
        return para_texts, para_ids

    # ========== 长期记忆懒加载接口 ==========
    def load_context(self) -> Dict:
        path = os.path.join(self.session_folder, "session_context.json")
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else {}

    def save_context(self, context: Dict):
        # Persist conversation state, never model credentials.
        keys = ("latest_subject", "candidate_spot_list", "city", "level", "allow_cross_city",
                "focus_mode", "focus_keep_round", "user_preference", "session_alias_map",
                "last_intent_type", "list_pages")
        write_json(os.path.join(self.session_folder, "session_context.json"),
                   {key: context[key] for key in keys if key in context})

    def _lazy_load_long_memory(self):
        """推理时懒加载，只加载一次"""
        if self._long_mem_loaded:
            return
        if os.path.exists(self.long_mem_path):
            with open(self.long_mem_path, "r", encoding="utf-8") as f:
                disk_data = json.load(f)
            # 合并磁盘，保留四类key
            for k in ["hobby", "price", "time", "spot_type"]:
                if k in disk_data:
                    self._long_mem_data[k] = disk_data[k]
        self._long_mem_loaded = True

    def get_long_memory(self) -> Dict:
        """获取长期记忆，触发懒加载"""
        self._lazy_load_long_memory()
        return self._long_mem_data.copy()

    def update_long_memory(self, new_data: Dict) -> bool:
        """
        更新长期记忆，只有发生变化才写磁盘
        new_data 允许key: hobby/price/time/spot_type
        return True:数据发生变更并写入磁盘；False:无变更，不写盘
        """
        self._lazy_load_long_memory()
        changed = False
        for k, v in new_data.items():
            if k in self._long_mem_data and self._long_mem_data[k] != v:
                self._long_mem_data[k] = v
                changed = True
        if changed:
            write_json(self.long_mem_path, self._long_mem_data)
        return changed


if __name__ == "__main__":
    # 简易测试代码
    # 1. 创建账号
    from user_account import UserAccount
    user = UserAccount.create_user("test_user_01")
    sid = user.create_new_session("demo_session")
    session_path = os.path.join(user.user_folder, sid)
    mem_mgr = SessionMemoryManager(session_path)

    # 写入测试对话
    mem_mgr.append_dialogue_paragraph(
        user_q="介绍丹霞山",
        agent_a="丹霞山是广东5A景区",
        keywords=["丹霞山", "5A", "广东"]
    )
    mem_mgr.append_dialogue_paragraph(
        user_q="白云山门票多少钱",
        agent_a="白云山门票5元",
        keywords=["白云山", "门票"]
    )
    # 检索示例
    hits = mem_mgr.search_short2_index(["白云山"])
    for hit_item in hits:
        para_content = mem_mgr.read_paragraph_by_id(hit_item["para_id"])
        print("命中段落：", para_content)
