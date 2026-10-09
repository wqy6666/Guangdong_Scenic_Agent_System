# -*- coding: utf-8 -*-
"""Deterministic intent and entity classification for the agent router."""
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
DB_PATH = str(Path(__file__).resolve().parent.parent / "shujuku" / "travel_final2.db")
SCENIC_ALIAS_MAP = {
    "白云山": "广州市白云山风景区",
    "长隆": "广州市长隆旅游度假区",
    "长隆欢乐世界": "广州市长隆旅游度假区",
    "长隆水上乐园": "广州市长隆旅游度假区",
    "长隆野生动物世界": "广州市长隆旅游度假区",
    "长隆大马戏": "广州市长隆旅游度假区",
    #【修复】新增丹霞山别名映射
    "丹霞山": "韶关市丹霞山景区",
}
CHINESE_NUM_TO_DIGIT = {
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10"
}
# 序号指代："第2个" / "第二个" / "第 3 名"
ORDINAL_PATTERN = re.compile(r"第\s*([0-9一二三四五六七八九十]{1,3})\s*[个名名家]")
# 【新增】疑似未收录景区检测：带景区类后缀的名词短语（用于虚构/未收录景区的边界明示）
UNKNOWN_SPOT_PATTERN = re.compile(r"([\u4e00-\u9fa5]{2,10})(?:风景区|旅游区|度假区|景区|景点)")
# 通用修饰前缀（命中则不算具体景区名，如"5A景区""亲子景区"）
UNKNOWN_SPOT_GENERIC_PREFIX = re.compile(
    r"(5A|4A|3A|A级|[0-9]A|级|的|些|个|家|亲子|温泉|海滨|森林|国家|省|市|县|区|免费|收费|著名|热门|适合|好玩)$"
)
# 【修复】泛化词过滤改为"包含即过滤"：原 $ 结尾锚定拦不住"A级旅游景区"切出的前缀"级旅游"（以"游"结尾）
UNKNOWN_SPOT_GENERIC_WORDS = ("5A", "4A", "3A", "A级", "0A", "级", "的", "些", "个", "家",
                              "亲子", "温泉", "海滨", "森林", "国家", "省", "市", "县", "区",
                              "免费", "收费", "著名", "热门", "适合", "好玩")
# 【新增】清单分页续传触发词（仅在上一轮清单分页未展示完且本轮无新实体时生效）
LIST_CONTINUE_WORDS = ("继续", "剩下", "其余", "下一页", "还有吗", "还有哪些", "接着", "后面的")
# 景区类后缀（解析链归一化用）
_SPOT_SUFFIXES = ("风景名胜区", "旅游度假区", "风景区", "旅游区", "度假区", "景区", "景点")


def _strip_spot_affixes(name: str, cities: Tuple[str, ...]) -> str:
    """去城市前缀 + 循环去景区类后缀，得到景区核心名"""
    n = (name or "").strip()
    for c in cities:
        for cf in _city_forms(c):
            if n.startswith(cf):
                n = n[len(cf):]
                break
    changed = True
    while changed:
        changed = False
        for suf in _SPOT_SUFFIXES:
            if n.endswith(suf) and len(n) > len(suf):
                n = n[:-len(suf)]
                changed = True
    return n


def resolve_spot_mention(mention: str) -> Optional[str]:
    """
    【实体解析链】把非标准景区名解析到词典标准名：
    原样 → 别名映射 → 去"市"前缀/景区后缀后的核心名相等比对 → 核心名包含比对（取最长核心）。
    解析失败返回None（调用方据此走未收录/澄清边界）。
    """
    if not mention or not mention.strip():
        return None
    cities, spots_tuple, _ = load_entities()
    spot_names = [s[0] for s in spots_tuple if s[0]]
    name = mention.strip()
    if name in spot_names:
        return name
    std, mapped = scenic_alias_normalize(name)
    if mapped and std in spot_names:
        return std
    core = _strip_spot_affixes(name, list(cities))
    if not core or len(core) < 2:
        return None
    for cand in spot_names:
        if _strip_spot_affixes(cand, list(cities)) == core:
            return cand
    best, best_len = None, 0
    for cand in spot_names:
        cand_core = _strip_spot_affixes(cand, list(cities))
        if cand_core and (cand_core in core or core in cand_core) and len(cand_core) > best_len:
            best, best_len = cand, len(cand_core)
    return best


def spot_content_exists(phrase: str) -> bool:
    """
    【存在性查询·工具证据】识别层不得宣布"数据库没有"——是否收录必须由本查询给出证据：
    对景区名（及其去前缀/后缀核心名）在词典与内容库中做检索，任一命中即视为有相关资料。
    """
    if not phrase or not phrase.strip():
        return False
    p = phrase.strip()
    terms = [p]
    try:
        cities, _, _ = load_entities()
        core = _strip_spot_affixes(p, list(cities))
        if core and len(core) >= 2 and core != p:
            terms.append(core)
    except Exception:
        pass
    conn = sqlite3.connect(DB_PATH)
    try:
        for term in terms:
            row = conn.execute(
                "SELECT COUNT(*) FROM tourism_spots_cleaned WHERE spot_name LIKE ? OR cleaned_content LIKE ?",
                (f"%{term}%", f"%{term}%")
            ).fetchone()
            if row and row[0] > 0:
                return True
        return False
    except Exception:
        return False
    finally:
        conn.close()
def scenic_alias_normalize(name: str, session_alias_map: Dict[str, str] | None = None) -> Tuple[str, bool]:
    if not name:
        return name, False
    if session_alias_map is None:
        session_alias_map = dict()
    raw_name = name.strip()
    if raw_name in session_alias_map:
        return session_alias_map[raw_name], True
    std_name = SCENIC_ALIAS_MAP.get(raw_name, raw_name)
    changed = std_name != raw_name
    return std_name, changed
ROUTE_STRUCTURED_DETAIL = "structured_detail"
ROUTE_SCENIC_LIST = "scenic_list"
ROUTE_SCENIC_COUNT = "scenic_count"
ROUTE_RECOMMENDATION = "recommendation"
ROUTE_RAG = "rag"
ROUTE_CLARIFY = "clarify"
ROUTE_AGENT_PROFILE = "agent_profile"
ROUTE_GREETING = "greeting"
AGENT_PROFILE_KEYWORDS = (
    "你是谁", "你叫什么", "你是做什么的", "你是干什么的", "介绍一下你", "自我介绍",
    "你能做什么", "你可以做什么", "能帮我什么", "可以帮我什么", "你能帮我什么",
    "怎么使用", "如何使用", "使用方法", "怎么用你",
)
AGENT_PROFILE_ANSWER = (
    "我是广东省A级旅游景区智能客服助手。\n"
    "我可以帮你查询景区介绍、门票、开放时间、地址、等级和联系电话，"
    "也可以按城市列出或统计景区，并提供游玩推荐、攻略和路线建议。\n"
    "例如：丹霞山门票多少钱？广州有哪些5A景区？"
)
GREETING_PATTERN = re.compile(
    r"^(?:你好|您好|嗨|哈喽|hello|hi|早上好|上午好|中午好|下午好|晚上好|在吗)(?:啊|呀|哇)?[!！。,.，?？\s]*$",
    re.IGNORECASE,
)
GREETING_ANSWER = (
    "你好，我是广东省A级旅游景区智能客服助手。"
    "你可以直接告诉我想查询的城市或景区，例如“丹霞山门票多少钱”或“广州有哪些5A景区”。"
)
DETAIL_FIELD_KEYWORDS = {
    "ticket": ("门票",),
    "open_time": ("开放时间",),
    "address": ("地址",),
    "level": ("等级",),
    "telephone": ("电话",),
}
NON_STRUCTURED_WORDS = ("停车场", "停车", "停车位", "观光车", "接驳车", "卫生间", "厕所", "充电桩", "充电宝")
LIST_KEYWORDS = ("哪些", "有哪些", "列出", "名单", "清单", "都有什么", "全部")
COUNT_KEYWORDS = ("多少个", "有多少", "一共有", "数量", "总数", "几个")
RECOMMENDATION_KEYWORDS = (
    "推荐", "攻略", "路线", "行程", "预算", "亲子", "老人", "对比", "挑选", "适合",
    "怎么安排", "怎么玩",
)
RAG_KEYWORDS = (
    "介绍", "概况", "简介", "特色", "历史", "文化", "好玩", "景点", "交通", "怎么去",
)
PRONOUNS = (
    "它", "这个景区", "该景区", "这里", "那里", "那", "这", "这两个", "哪个",
    "那儿", "此地", "那边", "该地方"
)
CONTEXT_CONTINUE_WORDS = (
    "介绍一下", "讲一下", "再说下", "再讲讲", "详细介绍", "详细说说",
    "继续介绍", "接着说", "再介绍下"
)
ALL_PRONOUN_TRIGGER = PRONOUNS + CONTEXT_CONTINUE_WORDS
SHORT_CONTINUE_WORDS = ("需要", "好的", "也行", "可以", "那就", "接着")
CROSS_CITY_KEYWORDS = ("周边", "周边游", "短途", "出发", "附近")
SUPPORTED_LEVELS = {"3A", "4A", "5A"}
MEMORY_TRIGGER_WORDS = ("请记住", "记住我", "我的条件", "我希望你记住", "我的要求")
def _contains_any(text: str, keywords: Tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)
def _city_forms(city: str) -> Tuple[str, ...]:
    """
    数据库原始值是【广州】(不带市)
    需要同时生成：广州、广州市，两种匹配形式用于用户问句实体抽取
    """
    normalized = city.removesuffix("市")
    forms = dict.fromkeys((city, normalized, normalized + "市"))
    return tuple(sorted(forms, key=len, reverse=True))
def _spot_forms(spot_name: str, cities: Tuple[str, ...]) -> Tuple[str, ...]:
    forms = {spot_name}
    simplified = spot_name
    for city in cities:
        for city_form in _city_forms(city):
            if simplified.startswith(city_form):
                simplified = simplified[len(city_form):]
                break
        if simplified != spot_name:
            break
    # 【修复】后缀迭代剥离：一次性replace会留下复合残体（"星湖旅游景区"去"旅游"→"星湖景区"，
    # 永远产不出裸名"星湖"）——逐层剥到稳定，每一层形态都进匹配表
    changed = True
    while changed:
        changed = False
        for suf in ("旅游度假区", "风景区", "旅游区", "度假区", "景区", "旅游"):
            if simplified.endswith(suf) and len(simplified) > len(suf):
                forms.add(simplified)
                simplified = simplified[:-len(suf)]
                changed = True
    # 【新增】尾部修饰词剥离（古龙峡"生态"）：剥掉后生成裸名进匹配表
    for tail_word in ("生态", "森林", "滨海", "山水", "温泉", "漂流"):
        if simplified.endswith(tail_word) and len(simplified) > len(tail_word) + 1:
            forms.add(simplified[:-len(tail_word)])
    # 无景区后缀的名称（珠海市圆明新园）：城市前缀剥除后的简写也要生成
    if simplified != spot_name and len(simplified) >= 2:
        forms.add(simplified)
    return tuple(form for form in forms if len(form) >= 2)
def _normalize_level(value: str | None) -> str | None:
    if not value:
        return None
    return value.upper().replace("级", "")
@lru_cache(maxsize=1)
def load_entities() -> Tuple[
    Tuple[str, ...],
    Tuple[Tuple[str, str | None], ...],
    Tuple[str, ...],
]:
    conn = sqlite3.connect(DB_PATH)
    try:
        city_rows = conn.execute(
            "SELECT DISTINCT city FROM scenic_spots WHERE city IS NOT NULL AND city != ''"
        ).fetchall()
        spot_rows = conn.execute(
            """
            SELECT DISTINCT spot_name, city
            FROM scenic_spots
            WHERE spot_name IS NOT NULL AND spot_name != ''
            """
        ).fetchall()
        level_rows = conn.execute(
            "SELECT DISTINCT level FROM scenic_spots WHERE level IS NOT NULL AND level != ''"
        ).fetchall()
    finally:
        conn.close()
    cities = tuple(sorted({row[0] for row in city_rows if row[0]}, key=len, reverse=True))
    spots = tuple(
        sorted(
            {(row[0], row[1]) for row in spot_rows if row[0]},
            key=lambda item: len(item[0]),
            reverse=True,
        )
    )
    levels = tuple(sorted({_normalize_level(row[0]) for row in level_rows if row[0]}, key=len, reverse=True))
    return cities, spots, levels
def clear_entity_cache() -> None:
    load_entities.cache_clear()
def extract_entities(user_query: str, session_alias_map: Dict[str, str] | None = None) -> Dict[str, Any]:
    if session_alias_map is None:
        session_alias_map = dict()
    cities, spots, database_levels = load_entities()
    city = None
    for candidate in cities:
        if any(form and form in user_query for form in _city_forms(candidate)):
            city = candidate
            break
    query_text = user_query
    for alias, std_name in session_alias_map.items():
        if alias in query_text:
            query_text = query_text.replace(alias, std_name)
    spot_matches = [
        (form, candidate)
        for candidate, candidate_city in spots
        if city is None or candidate_city is None or candidate_city == city
        for form in _spot_forms(candidate, cities)
        if form in query_text
    ]
    spot_name = max(spot_matches, key=lambda item: len(item[0]))[1] if spot_matches else None
    # 【修复】复合问句可能包含跨城景区（"告诉我丹霞山门票，再列出广州的5A景区"），
    # 同城无命中时放开城市限制再匹配一次，否则跨城景区实体会被城市过滤丢掉
    if spot_name is None:
        cross_matches = [
            (form, candidate)
            for candidate, candidate_city in spots
            for form in _spot_forms(candidate, cities)
            if form in query_text
        ]
        spot_name = max(cross_matches, key=lambda item: len(item[0]))[1] if cross_matches else None
    if spot_name:
        spot_name, _ = scenic_alias_normalize(spot_name, session_alias_map)
    # 【修复】等级识别前归一化：全角→半角、压缩数字与A之间空白，兼容"5 A"/"５Ａ"等写法
    level_probe = user_query.translate(str.maketrans("０１２３４５６７８９Ａａ", "0123456789Aa"))
    level_probe = re.sub(r"([0-9一二三四五])\s+(?=[Aa])", r"\1", level_probe)
    explicit_level = None
    invalid_level = None
    level_list = []
    for lv_match in re.findall(r"([0-9一二三四五])\s*[Aa](?:级)?", level_probe):
        raw_lv = CHINESE_NUM_TO_DIGIT.get(lv_match, lv_match if lv_match.isdigit() else None)
        if raw_lv is None:
            continue
        lv_val = _normalize_level(raw_lv + "A")
        if lv_val not in level_list:
            level_list.append(lv_val)
    if level_list:
        explicit_level = level_list[0]
        if explicit_level not in SUPPORTED_LEVELS:
            invalid_level = explicit_level
    level = explicit_level
    if level is None:
        upper_query = user_query.upper()
        level = next((item for item in database_levels if item and item in upper_query), None)
    fields = []
    for field_name, core_words in DETAIL_FIELD_KEYWORDS.items():
        for w in core_words:
            if w in user_query:
                fields.append(field_name)
                break
    has_pronoun = _contains_any(user_query, ALL_PRONOUN_TRIGGER)
    allow_cross_city: bool = _contains_any(user_query, CROSS_CITY_KEYWORDS)
    trigger_memory_instruction = _contains_any(user_query, MEMORY_TRIGGER_WORDS)
    # 【新增】免费/0元预算语义信号：供清单/数量/推荐路由做门票维度过滤
    ticket_free: bool = bool(re.search(r"(免费|不要钱|(?<![0-9])0元|零元)", user_query))
    # 【新增】预算上限提取："不超过150元"/"150元之内"→budget_max=150
    budget_max = None
    m_budget = re.search(r"(?:不超过|不得超过|不能超过|不要超过|最多|低于|小于)\s*(\d{1,5})\s*元|(\d{1,5})\s*元\s*(?:以内|之内|以下)", user_query)
    if m_budget:
        budget_max = int(m_budget.group(1) or m_budget.group(2))
    # 【新增】疑似未收录景区：仅在词典未命中任何景区时检测，避免与已识别实体冲突
    unknown_spot = None
    if spot_name is None:
        m_unknown = UNKNOWN_SPOT_PATTERN.search(user_query)
        if m_unknown:
            prefix = m_unknown.group(1)
            if len(prefix) >= 2 and not any(g in prefix for g in UNKNOWN_SPOT_GENERIC_WORDS):
                unknown_spot = m_unknown.group(0)
    return {
        "city": city,
        "spot_name": spot_name,
        "level": level,
        "level_list": level_list,
        "invalid_level": invalid_level,
        "fields": fields,
        "has_pronoun": has_pronoun,
        "allow_cross_city": allow_cross_city,
        "trigger_memory_instruction": trigger_memory_instruction,
        "unknown_spot": unknown_spot,
        "ticket_free": ticket_free,
        "budget_max": budget_max
    }
def _decision(route: str, answer_type: str, confidence: float, reason: List[str], entities: Dict[str, Any], answer_hint: str = None) -> Dict[str, Any]:
    return {
        "route": route,
        "answer_type": answer_type,
        "confidence": confidence,
        "reason": reason,
        "entities": entities,
        "answer_hint": answer_hint,
    }
def _empty_entities() -> Dict[str, Any]:
    return {
        "city": None,
        "spot_name": None,
        "level": None,
        "level_list": [],
        "invalid_level": None,
        "fields": [],
        "has_pronoun": False,
        "allow_cross_city": False,
        "trigger_memory_instruction": False,
        "unknown_spot": None,
        "ticket_free": False,
        "budget_max": None
    }
def _is_standalone_greeting(query: str) -> bool:
    return bool(GREETING_PATTERN.fullmatch(query))
def classify_query(user_query: str, session_ctx: Dict[str, Any]) -> Dict[str, Any]:
    query = " ".join(user_query.strip().split())
    session_alias_map = session_ctx.get("session_alias_map", dict())
    if _contains_any(query, AGENT_PROFILE_KEYWORDS):
        # 【修复】混合提问（"你是谁？另外白云山门票多少钱"）：同时含业务实体/意图词时不再提前返回，
        # 继续按业务意图分类，身份介绍由上层在最终回答前拼接
        probe_entities = extract_entities(query, session_alias_map=session_alias_map)
        business_hit = (
            bool(probe_entities["city"])
            or bool(probe_entities["spot_name"])
            or _contains_any(query, LIST_KEYWORDS + COUNT_KEYWORDS + RECOMMENDATION_KEYWORDS + RAG_KEYWORDS)
        )
        if not business_hit:
            return _decision(
                ROUTE_AGENT_PROFILE,
                "agent_intro",
                0.99,
                ["识别到助手身份、能力或使用方式咨询"],
                _empty_entities(),
            )
    if _is_standalone_greeting(query):
        return _decision(
            ROUTE_GREETING,
            "greeting",
            0.99,
            ["识别到独立问候语"],
            _empty_entities(),
        )
    raw_entities = extract_entities(query, session_alias_map=session_alias_map)
    user_has_new_entity = bool(raw_entities["city"]) or bool(raw_entities["spot_name"])
    ctx_latest_subject = session_ctx.get("latest_subject")
    ctx_candidate_list = session_ctx.get("candidate_spot_list", [])
    fill_reasons = []
    filled_entities = raw_entities.copy()
    # ==========【新增】序号指代解析："第2个"按candidate_spot_list下标回填景区实体 ==========
    ordinal_match = ORDINAL_PATTERN.search(query)
    if ordinal_match and (not user_has_new_entity) and ctx_candidate_list:
        ord_raw = ordinal_match.group(1)
        ord_digit = CHINESE_NUM_TO_DIGIT.get(ord_raw, ord_raw if ord_raw.isdigit() else None)
        if ord_digit is not None:
            ord_index = int(ord_digit) - 1
            if 0 <= ord_index < len(ctx_candidate_list):
                filled_entities["spot_name"] = ctx_candidate_list[ord_index]
                fill_reasons.append(f"序号指代回填：第{ord_digit}个 → {ctx_candidate_list[ord_index]}")
            else:
                return _decision(
                    ROUTE_CLARIFY,
                    "clarification",
                    0.80,
                    [f"序号{ord_digit}超出候选范围1~{len(ctx_candidate_list)}"] + fill_reasons,
                    filled_entities,
                    answer_hint=f"当前候选列表共有{len(ctx_candidate_list)}个景区，请输入1到{len(ctx_candidate_list)}之间的序号。",
                )
    # ==========【清单续传】"继续/剩下的景区呢/下一页"：上一轮分页清单未展示完 → 直接续传，不走澄清/规划器 ==========
    _list_pages = session_ctx.get("list_pages")
    _list_city = str(_list_pages.get("city") or "").casefold() if isinstance(_list_pages, dict) else ""
    _query_city = str(raw_entities.get("city") or "").casefold()
    _same_list_city = not _query_city or _query_city == _list_city
    _list_level = _list_pages.get("level") if isinstance(_list_pages, dict) else None
    _query_level = raw_entities.get("level")
    _same_list_level = not _query_level or not _list_level or _query_level == _list_level
    if (_list_pages and _same_list_city and _same_list_level
            and not filled_entities["spot_name"]
            and _contains_any(query, LIST_CONTINUE_WORDS)):
        filled_entities["list_continue"] = True
        filled_entities["city"] = _list_pages.get("city")
        if not filled_entities.get("level") and _list_level:
            filled_entities["level"] = _list_level
        fill_reasons.append(f"清单续传：{_list_pages.get('city')} 从第{_list_pages.get('offset', 0) + 1}家继续")
        return _decision(
            ROUTE_SCENIC_LIST,
            "scenic_list",
            0.92,
            ["识别到清单分页续传"] + fill_reasons,
            filled_entities,
        )
    # 省略式追问（"周末呢？""周一的安排你有明确资料吗？"）视同代词，回填上一轮主体景区
    ellipsis_continuation = bool(re.search(r"(?:呢|吗|么)\s*[?？!！。]?\s*$", query))
    # 序号指代已回填景区时跳过代词分支，否则"介绍一下第2个"会被当成纯代词提问拦截
    if (raw_entities["has_pronoun"] or ellipsis_continuation) and (not user_has_new_entity) and (not filled_entities["spot_name"]):
        # 【新增】清单类追问（"那4A呢"）：上一轮是城市清单，追问的是同城市换等级的清单
        last_intent = session_ctx.get("last_intent_type", "")
        list_followup = (filled_entities["level"] is not None
                         or _contains_any(query, LIST_KEYWORDS) or _contains_any(query, COUNT_KEYWORDS))
        if (last_intent in (ROUTE_SCENIC_LIST, ROUTE_SCENIC_COUNT)
                and session_ctx.get("city") and list_followup):
            filled_entities["city"] = session_ctx.get("city")
            fill_reasons.append(f"清单类追问：继承城市={session_ctx.get('city')}（上一轮为{last_intent}），等级沿用本轮识别")
        elif len(ctx_candidate_list) >= 2:
            return _decision(
                ROUTE_CLARIFY,
                "clarification",
                0.70,
                ["纯代词提问，候选列表多个景区，请求用户确认具体景点"] + fill_reasons,
                filled_entities,
                answer_hint="当前存在多个候选景点，请问你指的是哪一个景点？"
            )
        elif ctx_latest_subject is None:
            return _decision(
                ROUTE_CLARIFY,
                "clarification",
                0.65,
                ["纯代词提问，但当前会话无任何历史主体，禁止进入RAG，请求用户输入景区名称"],
                filled_entities,
                answer_hint="没有找到上文提到的景区，请你告诉我需要查询的景区名称。"
            )
        else:
            filled_entities["spot_name"] = ctx_latest_subject
            fill_reasons.append(f"代词回填上一轮唯一主体 latest_subject={ctx_latest_subject}")
    if filled_entities.get("city"):
        if session_ctx.get("city") != filled_entities["city"]:
            session_ctx["level"] = None
            fill_reasons.append("检测到新城市，清空历史level")
        # 【修复】城市写回会话上下文：推荐追问的城市继承分支依赖session_ctx["city"]
        session_ctx["city"] = filled_entities["city"]
    if filled_entities["invalid_level"]:
        return _decision(
            ROUTE_CLARIFY,
            "clarification",
            0.98,
            [f"检测到不支持的景区等级：{filled_entities['invalid_level']}"] + fill_reasons,
            filled_entities,
        )
    # ==========【新增】疑似未收录景区：识别层只表达"未能识别"，"无数据"结论由存在性查询给出 ==========
    unknown_spot = filled_entities.get("unknown_spot")
    if unknown_spot:
        hint = (f"未能识别该景区「{unknown_spot}」，无法确认其信息。"
                "请确认景区名称是否正确，或告诉我城市，我为您列出该市的A级景区。")
        return _decision(
            ROUTE_CLARIFY,
            "clarification",
            0.90,
            [f"检测到疑似未收录景区「{unknown_spot}」"] + fill_reasons,
            filled_entities,
            answer_hint=hint,
        )
    # ==========【核心修复】推荐会话补充条件：本轮无实体，但是上一轮是recommendation，继承city，不走clarify ==========
    last_intent = session_ctx.get("last_intent_type", "")
    if filled_entities["spot_name"] is None and filled_entities["city"] is None:
        if last_intent == ROUTE_RECOMMENDATION and session_ctx.get("city"):
            inherit_city = session_ctx.get("city")
            filled_entities["city"] = inherit_city
            fill_reasons.append(f"推荐会话补充筛选条件，继承上一轮城市:{inherit_city}")
            return _decision(
                ROUTE_RECOMMENDATION,
                "recommendation",
                0.86,
                ["推荐会话补充筛选条件，继承上一轮城市"] + fill_reasons,
                filled_entities,
            )
        # 普通情况，既无城市无景区，不是推荐补充会话，直接澄清；不再支持全省广东推荐
        return _decision(
            ROUTE_CLARIFY,
            "clarification",
            0.72,
            ["缺少明确城市、景区或问题类型"] + fill_reasons,
            filled_entities,
            answer_hint="请您提供广东省下面具体的地级市（例如广州、韶关）或者具体景区名称，我再为您处理。"
        )
    # =====================================================================================================
    has_list = _contains_any(query, LIST_KEYWORDS)
    has_count = _contains_any(query, COUNT_KEYWORDS)
    has_recommendation = _contains_any(query, RECOMMENDATION_KEYWORDS)
    if has_recommendation:
        return _decision(
            ROUTE_RECOMMENDATION,
            "recommendation",
            0.91,
            ["识别到推荐、攻略、路线或对比意图"] + fill_reasons,
            filled_entities,
        )
    # 【新增】预算/免费类推荐快速通道：城市+免费语义+无景区实体+非清单/数量问句
    # （"广州门票0元的景区"——推荐关键词表覆盖不到的预算语义，识别层直接定性，省一次Judge）
    if (filled_entities["city"] and not filled_entities["spot_name"]
            and not has_list and not has_count
            and re.search(r"(免费|不要钱|(?<![0-9])0元|零元)", query)):
        return _decision(
            ROUTE_RECOMMENDATION,
            "recommendation",
            0.90,
            ["识别到城市+免费/0元预算推荐意图"] + fill_reasons,
            filled_entities,
        )
    # 【新增】复合意图：景区字段查询与城市清单/数量同现（"告诉我丹霞山门票，再列出广州的5A景区"），
    # 不再只走清单路由丢掉前半问，改交LLM‑Judge拆分子意图
    if filled_entities["spot_name"] and filled_entities["city"] and (has_list or has_count):
        return _decision(
            ROUTE_RAG,
            "text",
            0.82,
            ["识别到景区字段查询与城市清单复合意图，交由LLM‑Judge拆分子意图"] + fill_reasons,
            filled_entities,
        )
    if filled_entities["city"] and (has_count or has_list or filled_entities["level"]):
        route = ROUTE_SCENIC_COUNT if has_count and not has_list else ROUTE_SCENIC_LIST
        return _decision(
            route,
            "scenic_count" if route == ROUTE_SCENIC_COUNT else "scenic_list",
            0.95,
            ["识别到有效城市", "识别到清单或数量查询"] + fill_reasons,
            filled_entities,
        )
    if filled_entities["spot_name"]:
        return _decision(
            ROUTE_RAG,
            "text",
            0.82,
            ["识别到明确景区实体，交由LLM‑Judge做字段语义解析，判断是否走结构化SQL"] + fill_reasons,
            filled_entities
        )
    return _decision(
        ROUTE_RAG,
        "text",
        0.82,
        ["识别为景区知识问答"] + fill_reasons,
        filled_entities,
    )
