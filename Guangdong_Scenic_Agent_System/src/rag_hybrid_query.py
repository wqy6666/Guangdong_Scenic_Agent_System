# -*- coding: utf-8 -*-
import json
import re
import copy
import torch
import numpy as np
import pickle
import sqlite3
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from safe_guard import safe_call, logger, SAFE_EMPTY_LIST, SAFE_SPOT_DETAIL_DICT
from llm_answer import generate_answer_by_api, build_chat_messages, OUTPUT_FORMAT_RULE
from llm_judge_call import call_llm_route_judge
from memory_short2 import short2_memory
from service_errors import LLMServiceError

# ========== 全部本地工具函数【放在最前面，向量加载代码之前，避免NameError】 ==========
def parse_ticket_max_num(ticket_str: str):
    import re
    if not ticket_str:
        return None
    s = ticket_str.strip()
    if s == "暂无" or "暂无" in s[:4]:
        return None
    # 【修复】"免费"语义上就是0元：参与预算比较（budget_max=0时免费景区获得加成）
    if "免费" in s:
        return 0
    nums = re.findall(r"\d+", s)
    if not nums:
        return None
    num_list = [int(x) for x in nums]
    return max(num_list)


def _update_session_subject_by_spots(session_ctx: dict, spots: list):
    if spots is None or len(spots) == 0:
        return
    spot_names = []
    for s in spots:
        if isinstance(s, dict) and "spot_name" in s:
            spot_names.append(s["spot_name"])
        elif isinstance(s, str):
            spot_names.append(s)
    spot_names = list(dict.fromkeys(spot_names))
    if len(spot_names) == 1:
        session_ctx["latest_subject"] = spot_names[0]
        session_ctx["candidate_spot_list"] = []
        session_ctx["focus_mode"] = False
        session_ctx["focus_keep_round"] = 0
    elif len(spot_names) >= 2:
        session_ctx["candidate_spot_list"] = spot_names
        session_ctx["latest_subject"] = None
        session_ctx["focus_mode"] = True
        session_ctx["focus_keep_round"] = 5


def _aggregate_spot_from_docs(doc_list: list) -> list:
    spot_cnt = dict()
    for doc in doc_list:
        sp = doc.get("spot_name", "")
        if sp:
            spot_cnt[sp] = spot_cnt.get(sp, 0) + 1
    if len(spot_cnt) == 0:
        return []
    sorted_spots = sorted(spot_cnt.items(), key=lambda x: x[1], reverse=True)
    return [s[0] for s in sorted_spots]


def _query_result(answer: str, docs: list, return_metadata: bool, route: str, answer_type: str, spots=None):
    if spots is None:
        spots = []
    meta = {
        "route": route,
        "answer_type": answer_type,
        "spots": spots
    }
    if return_metadata:
        return answer, docs, meta
    else:
        return answer


def _attach_route_metadata(result, return_meta: bool, route_info: dict):
    if not return_meta:
        return result
    ans, docs, meta = result
    meta["confidence"] = route_info.get("confidence")
    meta["reason"] = route_info.get("reason", [])
    return ans, docs, meta
# =====================================================================================

# =========【修改】移除旧的全局长期记忆 import =========
# from memory_longterm import longterm_memory
from config import (
    VECTOR_NPY_FILE,
    META_FILE,
    DB_PATH,
    RERANKER_MODEL_PATH,
    EMBED_MODEL_PATH,
    TOP_K_VECTOR,
    TOP_K_KEYWORD,
    RERANK_TOP
)
from route_classifier import (
    AGENT_PROFILE_ANSWER,
    GREETING_ANSWER,
    ROUTE_AGENT_PROFILE,
    ROUTE_CLARIFY,
    ROUTE_RAG,
    ROUTE_RECOMMENDATION,
    ROUTE_GREETING,
    ROUTE_SCENIC_COUNT,
    ROUTE_SCENIC_LIST,
    classify_query,
    RAG_KEYWORDS,
    load_entities,
    PRONOUNS,
    CONTEXT_CONTINUE_WORDS,
    scenic_alias_normalize,
    resolve_spot_mention
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("正在加载向量库与模型，请稍候......")
vector_array = np.load(VECTOR_NPY_FILE)
with open(META_FILE, "rb") as f:
    meta_data = pickle.load(f)


def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
    model = AutoModel.from_pretrained(EMBED_MODEL_PATH).to(DEVICE)
    return tokenizer, model


emb_tokenizer, emb_model = load_embedding_model()
rerank_model = CrossEncoder(RERANKER_MODEL_PATH, device=DEVICE)
print("✅所有模型加载完成，可以开始对话！")


def _llm_merge_multi_result(user_query: str, sub_answers: list, llm_cfg: dict) -> str:
    """把多个子任务回答合并为一个回答"""
    from llm_answer import generate_answer_by_api
    split_text = "\n====分割====\n"
    content_block = split_text.join(sub_answers)
    merge_prompt = (
        f"用户原始提问：{user_query}\n\n"
        "下面是各个景区分别查询得到的结果，请把下面多段内容整理合并成通顺完整的回答，不要遗漏信息，不要编造内容：\n"
        f"{content_block}\n\n"
        "直接输出整理后的回答，不要多余开场白。"
    )
    messages = [
        {"role": "system", "content": "你负责合并多段查询结果，只使用给定内容，禁止编造信息。"},
        {"role": "user", "content": merge_prompt}
    ]
    return generate_answer_by_api(messages, llm_cfg=llm_cfg)



def _normalize_city_name(city: str) -> str:
    """
    数据库scenic_spots.city字段存储格式：不带"市"，例如：广州、韶关、汕头
    输入：广州市 / 广州 → 统一输出：广州
    """
    if not city:
        return city
    city = city.strip()
    if city.endswith("市"):
        city = city.removesuffix("市")
    return city

def post_filter_docs(doc_list: list, city: str | None, allow_cross_city: bool, spot_whitelist: list[str]) -> list:
    """
    检索文档后置过滤
    :param doc_list: 原始检索chunk列表
    :param city: 当前会话城市
    :param allow_cross_city: 是否允许跨城市检索
    :param spot_whitelist: 指定景区白名单
    :return: 过滤后的文档列表
    """
    if not doc_list:
        return []
    out_docs = []
    for doc in doc_list:
        doc_spot = doc.get("spot_name", "")
        doc_city = doc.get("city", "")

        # 如果有景区白名单，优先匹配景区名称
        if spot_whitelist and len(spot_whitelist) > 0:
            if doc_spot in spot_whitelist:
                out_docs.append(doc)
            continue

        # 城市过滤逻辑
        if city and not allow_cross_city:
            if doc_city == city:
                out_docs.append(doc)
        else:
            # 允许跨城市，全部保留
            out_docs.append(doc)
    return out_docs


def extract_common_requirement(raw_query: str, spot_name_list: list[str]) -> str:
    """
    【新增工具函数】
    从原始用户问句移除全部景区实体名称，提取公共查询约束条件。
    """
    if not raw_query:
        return ""
    text = raw_query
    for spot in spot_name_list:
        text = text.replace(spot, "")
    text = re.sub(r"\s+(和|与|以及|还有|及)\s+", "，", text)
    text = re.sub(r"[，,、]{2,}", "，", text)
    text = text.strip(" ，,、")
    return text


@safe_call(default_ret=None, log_ctx="get_query_embedding")
def get_query_embedding(text: str, tokenizer, model):
    instruction = "为这个句子生成表示以用于检索："
    input_text = instruction + text
    encoded = tokenizer(input_text, max_length=512, truncation=True, padding=True, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**encoded)
        vec = out.last_hidden_state[:, 0]
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
    return vec.cpu().numpy()


@safe_call(default_ret=SAFE_EMPTY_LIST, log_ctx="vector_search_numpy")
def vector_search_numpy(q_vec, vec_lib, meta, top_k):
    score = np.dot(vec_lib, q_vec.T).squeeze(-1)
    top_k = min(int(top_k), score.size)
    if top_k <= 0:
        return []
    candidate_idx = np.argpartition(-score, top_k - 1)[:top_k]
    top_idx = candidate_idx[np.argsort(-score[candidate_idx])]
    res = []
    for idx in top_idx:
        res.append(meta[idx])
    return res


@safe_call(default_ret=SAFE_EMPTY_LIST, log_ctx="keyword_search_db")
def keyword_search_db(query_text: str, top_k: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    sql = """
        SELECT spot_name, city, cleaned_content, source_url, source_domain
        FROM tourism_spots_cleaned
        WHERE spot_name LIKE ? OR cleaned_content LIKE ?
        LIMIT ?
    """
    query_pattern = f"%{query_text}%"
    cur.execute(sql, (query_pattern, query_pattern, top_k))
    rows = cur.fetchall()
    conn.close()
    out = []
    for name, city, content, source_url, source_domain in rows:
        out.append({
            "spot_name": name,
            "city": city,
            "chunk_text": content or f"景区名称：{name}",
            "source_url": source_url,
            "source_domain": source_domain,
        })
    return out


@safe_call(default_ret=SAFE_EMPTY_LIST, log_ctx="rerank")
def rerank(query: str, doc_list: list, top_n: int):
    if not doc_list:
        return []
    text_list = []
    for doc in doc_list:
        if isinstance(doc, dict):
            text_list.append(doc["chunk_text"])
        else:
            text_list.append(str(doc))
    score_pair = [[query, text] for text in text_list]
    scores = rerank_model.predict(score_pair)
    ranked = sorted(zip(scores, doc_list), key=lambda x: -x[0])
    out = [item[1] for item in ranked[:top_n]]
    return out


@safe_call(default_ret=SAFE_SPOT_DETAIL_DICT, log_ctx="get_single_spot_detail")
def get_single_spot_detail(spot_name: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    sql = """
        SELECT
            s.spot_name,
            s.address,
            MAX(c.location_val) AS location_val,
            MAX(c.ticket_val) AS ticket_val,
            MAX(c.open_time_val) AS open_time_val,
            s.city,
            s.level,
            s.tel
        FROM scenic_spots AS s
        LEFT JOIN scenic_cleaned AS c ON c.spot_name = s.spot_name
        WHERE s.spot_name = ?
        GROUP BY s.spot_name, s.address, s.city, s.level, s.tel
        LIMIT 1
    """
    cur.execute(sql, (spot_name,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return {
            "spot_name": spot_name,
            "address": "暂无地址信息",
            "ticket": "暂无门票信息",
            "open_time": "暂无开放时间信息",
            "city": None,
            "level": "暂无景区等级信息",
            "telephone": "暂无联系电话",
        }
    # 地址以scenic_spots.address精确地址为主（该列268行全有值）；location_val多为残缺描述，仅作兜底，不拼接
    address_val = (row[1] or "").strip() or (row[2] or "").strip() or "暂无地址信息"
    return {
        "spot_name": row[0],
        "address": address_val,
        "ticket": row[3] or "暂无门票信息",
        "open_time": row[4] or "暂无开放时间信息",
        "city": row[5],
        "level": row[6] or "暂无景区等级信息",
        "telephone": row[7] or "暂无联系电话",
    }


@safe_call(default_ret=0, log_ctx="query_scenic_count")
def query_scenic_count(city_name: str, target_level=None):
    city_name = _normalize_city_name(city_name)
    conn = sqlite3.connect(DB_PATH)
    try:
        if target_level:
            sql = """
                SELECT COUNT(DISTINCT spot_name)
                FROM scenic_spots
                WHERE city = ? AND level = ?
            """
            row = conn.execute(sql, (city_name, target_level)).fetchone()
        else:
            sql = """
                SELECT COUNT(DISTINCT spot_name)
                FROM scenic_spots
                WHERE city = ?
            """
            row = conn.execute(sql, (city_name,)).fetchone()
    finally:
        conn.close()
    return int(row[0] or 0)


@safe_call(default_ret=SAFE_EMPTY_LIST, log_ctx="query_city_scenic_list")
def query_city_scenic_list(city_name: str, target_level=None, ticket_free=False):
    city_name = _normalize_city_name(city_name)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # 【修复】scenic_spots/scenic_cleaned存在同名重复行，DISTINCT在ticket不同时去不掉，
    # 改GROUP BY按spot_name聚合，保证与COUNT(DISTINCT spot_name)数量一致（39=39而非41）
    # 【新增】ticket_free：清单/数量工具的字段值过滤能力（"列出广州门票0元的景区"）
    # 窄口径=基础门票免费；宽口径=含"不收取门票费用/免门票入园"等写法（空结果回流时用）
    free_filter = ""
    if ticket_free == "wide":
        free_filter = " AND (c.ticket_val LIKE '%免费%' OR c.ticket_val LIKE '%不收取%' OR c.ticket_is_free = 1)"
    elif ticket_free:
        free_filter = " AND (c.ticket_val LIKE '免费%' OR c.ticket_val LIKE '%：免费%')"
    if target_level is not None:
        sql = f"""
        SELECT s.spot_name, s.city, s.level, MAX(c.ticket_val) AS ticket_val
        FROM scenic_spots s
        LEFT JOIN scenic_cleaned c ON s.spot_name = c.spot_name
        WHERE s.city = ? AND s.level = ?{free_filter}
        GROUP BY s.spot_name, s.city, s.level
        """
        cur.execute(sql, (city_name, target_level))
    else:
        sql = f"""
        SELECT s.spot_name, s.city, s.level, MAX(c.ticket_val) AS ticket_val
        FROM scenic_spots s
        LEFT JOIN scenic_cleaned c ON s.spot_name = c.spot_name
        WHERE s.city = ?{free_filter}
        GROUP BY s.spot_name, s.city, s.level
        """
        cur.execute(sql, (city_name,))
    rows = cur.fetchall()
    conn.close()
    doc_list = []
    for spot_name, city, level, ticket_val in rows:
        if level is None:
            level_text = "暂无景区等级信息"
        else:
            level_text = level
        content_text = f"景区名称：{spot_name}，景区等级：{level_text}"
        doc_list.append({
            "spot_name": spot_name,
            "city": city,
            "level": level,
            "ticket": ticket_val or "暂无门票信息",
            "chunk_text": content_text
        })
    return doc_list

@safe_call(default_ret=(SAFE_EMPTY_LIST, SAFE_EMPTY_LIST), log_ctx="query_city_scenic_list_budget")
def query_city_scenic_list_budget(city_name: str, target_level=None, budget_max=None):
    """
    预算维度清单：复用全量清单后按票价值数值过滤（ticket_val是混杂文本，SQL无法做数值比较）。
    返回 (符合预算的景区列表, 票价数据缺失的景区列表)——缺失组如实说明，不冒充符合也不丢弃。
    """
    all_docs = query_city_scenic_list(city_name, target_level)
    limit = int(budget_max)
    matched, unknown = [], []
    for d in all_docs:
        tv = parse_ticket_max_num(d.get("ticket", ""))
        if tv is None:
            unknown.append(d)
        elif tv <= limit:
            matched.append(d)
    return matched, unknown


@safe_call(default_ret=SAFE_EMPTY_LIST, log_ctx="query_city_scenic_filtered")
def query_city_scenic_filtered(city_name: str, target_level=None, budget_max=None):
    """
    SQL硬过滤：城市、等级；
    budget_max==0（免费）是确定性预算值：用ticket_is_free列+“免费”标志直接过滤；
    其他预算值仍交LLM软筛选（ticket_val混杂文本，数值过滤易错）。
    """
    city_name = _normalize_city_name(city_name)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    base_sql = """
    SELECT s.spot_name, s.city, s.level, MAX(c.ticket_val) AS ticket_val, MAX(c.ticket_is_free) AS ticket_is_free
    FROM scenic_spots s
    LEFT JOIN scenic_cleaned c ON s.spot_name = c.spot_name
    WHERE s.city = ?
    """
    params = [city_name]
    if target_level:
        base_sql += " AND s.level = ?"
        params.append(target_level)
    try:
        if budget_max is not None and int(budget_max) == 0:
            base_sql += " AND (c.ticket_val LIKE '免费%' OR c.ticket_val LIKE '%：免费%')"
    except (TypeError, ValueError):
        pass
    base_sql += " GROUP BY s.spot_name, s.city, s.level"
    cur.execute(base_sql, params)
    rows = cur.fetchall()
    conn.close()
    out = []
    for spot_name, city, level, ticket_val, _is_free in rows:
        out.append({
            "spot_name": spot_name,
            "city": city,
            "level": level,
            "ticket": ticket_val or "暂无门票信息"
        })
    return out


def _retrieve_spot_knowledge_chunk(spot_name: str, user_query: str = "") -> str:
    try:
        spot_name, _ = scenic_alias_normalize(spot_name)
        q_vec = get_query_embedding(spot_name, emb_tokenizer, emb_model)
        vec_res = vector_search_numpy(q_vec, vector_array, meta_data, TOP_K_VECTOR)
        short_sp, _ = scenic_alias_normalize(spot_name)
        kw_res = keyword_search_db(short_sp, TOP_K_KEYWORD)
        merge_chunk = vec_res + kw_res
        unique_chunk = []
        seen = set()
        for ck_item in merge_chunk:
            ck_key = ck_item.get("spot_name", "") + "|||" + ck_item.get("chunk_text", "")
            if ck_key not in seen:
                seen.add(ck_key)
                unique_chunk.append(ck_item)
        spot_only_chunk = [c for c in unique_chunk if c.get("spot_name") == spot_name]
        if not spot_only_chunk:
            return "暂无知识库介绍资料"
        chunk_merge_text = "\n".join([c["chunk_text"] for c in spot_only_chunk])
        return chunk_merge_text if chunk_merge_text.strip() else "暂无知识库介绍资料"
    except Exception as e:
        logger.warning(f"_retrieve_spot_knowledge_chunk[{spot_name}]失败 err={str(e)}", exc_info=True)
        return "暂无知识库介绍资料"


def extract_recommend_filter_condition(user_query: str, session_ctx: dict, llm_cfg: dict):
    sys_prompt = """你是推荐条件提取助手，**禁止生成任何推荐回答，只输出JSON**。 用户输入城市景区推荐问句，请提取下面字段：
{
"city":"地级市名称或者null",
"scenic_level":"5A/4A/3A或者null",
"budget_max":门票预算上限数字，没有则null,
"crowd":"人群标签：亲子/老人/年轻人等，没有null",
"child_age":孩子年龄数字，没有null,
"style":"景色风格：自然风景/人文古迹等，没有null",
"traffic_require":"交通要求：交通方便等，没有null",
"exclude_spots":["用户明确说不要/不去/排除的景区标准名称数组，没有则null"]
}
规则：
1. 只提取用户明确提到的信息，不要脑补；
2. city只输出地级市，例如输入广州输出广州市，内部归一化会去掉市；不要输出广东省这类省份；
3. budget_max仅针对门票价格，不要把住宿交通算进去；
4. 用户说“不要X/X不去/X排除”时，X必须写入exclude_spots；排除的景区绝对不是推荐参考；
5. 严格```json ```输出，不要额外文字。
"""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"用户提问：{user_query}"}
    ]
    raw = generate_answer_by_api(messages, llm_cfg=llm_cfg, timeout=12)
    json_str = re.search(r"```json\s*\n(.*?)\n```", raw, re.DOTALL)
    if not json_str:
        logger.warning("推荐条件提取：未找到json块")
        return None
    try:
        cond = json.loads(json_str.group(1).strip())
        logger.info("【推荐LLM提取过滤条件】解析完成")
        return cond
    except json.JSONDecodeError as e:
        logger.warning(f"推荐条件JSON解析失败 err={str(e)}")
        return None


@safe_call(default_ret=None, log_ctx="extract_session_preference_update")
def extract_session_preference_update(user_query: str, session_ctx: dict, llm_cfg: dict):
    sys_prompt = """你是会话偏好提取助手，只输出JSON，禁止生成回答。
提取用户补充的游玩筛选条件：
{
"crowd":"人群标签，无则null",
"child_age":孩子年龄数字，无则null,
"budget_max":门票预算上限数字，无则null,
"style":"景色风格，无则null",
"traffic_require":"交通要求，无则null",
"extra_require":"其他筛选要求的原文（如排除某些景区、出行天数、特殊注意事项），无则null"
}
只提取用户明确说出的内容，禁止脑补，```json```输出。
"""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"用户本轮补充条件：{user_query}"}
    ]
    raw = generate_answer_by_api(messages, llm_cfg=llm_cfg, timeout=12)
    json_str = re.search(r"```json\s*\n(.*?)\n```", raw, re.DOTALL)
    if not json_str:
        logger.warning("会话补充偏好提取未找到json")
        return session_ctx["user_preference"]
    try:
        new_pref = json.loads(json_str.group(1).strip())
        old_pref = session_ctx.get("user_preference", {})
        merged_pref = {**old_pref, **new_pref}
        merged_pref = {k: v for k, v in merged_pref.items() if v is not None}
        session_ctx["user_preference"] = merged_pref
        logger.info("【会话临时偏好已更新，不持久化】")
        return merged_pref
    except Exception as e:
        logger.warning(f"会话偏好提取解析失败 err={str(e)}", exc_info=True)
        return session_ctx["user_preference"]


# =========【修改】重构 extract_memory_preference，不再使用旧longterm_memory全局实例 =========
@safe_call(default_ret=None, log_ctx="extract_memory_preference")
def extract_memory_preference(user_query: str, session_ctx: dict, curr_mem_mgr, llm_cfg: dict):

    prompt = f""" 用户输入：{user_query}
请提取用户游玩偏好，输出严格```json```格式，字段：
{{
"crowd":"",
"child_age":null,
"max_price":null,
"city":"",
"travel_mode":""
}}
crowd：人群标签：亲子/老人/年轻人等；
child_age：孩子年龄；
max_price：预算数字，没有填null；
city：目标城市；
travel_mode：出行方式；
没有对应值填空字符串或者null，不要编造信息。
"""
    messages = [{"role": "system", "content": "你是偏好提取助手，只提取用户明确提到的信息，禁止脑补。"},
                {"role": "user", "content": prompt}]
    raw = generate_answer_by_api(messages, llm_cfg=llm_cfg, timeout=12)

    json_str = re.search(r"```json\s*\n(.*?)\n```", raw, re.DOTALL).group(1).strip()
    pref = json.loads(json_str)
    valid_keys = ["crowd", "child_age", "max_price", "city", "travel_mode"]
    new_pref = {k: pref.get(k) for k in valid_keys}
    new_pref = {k: v for k, v in new_pref.items() if v not in ("", None)}
    session_ctx["user_preference"].update(new_pref)

    sync_data = {
        "hobby": new_pref.get("crowd"),
        "price": f"{new_pref['max_price']}元以内" if new_pref.get("max_price") else None,
        "time": None,
        "spot_type": None
    }
    changed = False
    if curr_mem_mgr is not None:
        changed = curr_mem_mgr.update_long_memory(sync_data)
        if changed:
            logger.info("【持久记忆更新】偏好已同步写入当前会话long_memory.json")
        else:
            logger.info("【持久记忆更新】偏好无变化，不写磁盘")
    else:
        logger.warning("[extract_memory_preference] curr_mem_mgr为None，跳过持久化保存")
    return session_ctx
def score_candidate_spot_list(sql_filtered_list, session_pref, user_query):
    """
    【本地粗打分，不调用LLM】
    输入：SQL查询出来原始景区列表、session会话偏好
    返回：按匹配分数降序排好的候选列表，返回最多TOP12
    打分规则：
        5A=5分；4A=3分
        预算符合max_price +4分
        知识库命中亲子/儿童/研学/科普关键词 +4分
    """
    TOP_K_AFTER_SCORE = 12
    keyword_list = ["亲子","儿童","研学","科普","动物园","游乐","青少年"]
    scored = []
    for idx, spot_item in enumerate(sql_filtered_list):
        score = 0
        # 1、等级打分
        lv = spot_item.get("level","")
        if lv == "5A":
            score +=5
        elif lv == "4A":
            score +=3
        # 2、预算打分（预算值可能是int或"50元"这类字符串，统一安全转换）
        ticket_str = spot_item.get("ticket","")
        ticket_max = parse_ticket_max_num(ticket_str)
        max_price_pref = session_pref.get("max_price") or session_pref.get("budget_max")
        budget_val = None
        if max_price_pref is not None:
            try:
                budget_val = int(str(max_price_pref).strip().removesuffix("元"))
            except ValueError:
                budget_val = None
        if ticket_max is not None and budget_val is not None and ticket_max <= budget_val:
            score +=4
        #3、知识库关键词匹配打分
        chunk_txt = _retrieve_spot_knowledge_chunk(spot_item["spot_name"], user_query)
        hit_key = any(k in chunk_txt for k in keyword_list)
        if hit_key:
            score +=4
        # 增加idx作为第二排序key，分数相同时用下标，避免dict对比报错
        scored.append( (-score, idx, spot_item) )
    #分数降序
    scored.sort()
    sorted_items = [i[2] for i in scored]
    return sorted_items[:TOP_K_AFTER_SCORE]


def reorder_for_llm_attention(spot_item_list):
    """
    针对Lost‑in‑the‑middle问题做U型重排
    输入：已经按分数降序排好的候选列表
    返回：重排之后列表：[第0名] + 中间部分 + [第1名]
    """
    if len(spot_item_list) <=2:
        return spot_item_list
    first = spot_item_list[0]
    second = spot_item_list[1]
    middle = spot_item_list[2:]
    return [first] + middle + [second]

def llm_filter_spot_name_list(spot_simple_list, user_pref, llm_cfg: dict, user_query: str = ""):
    """
    【两阶段第一阶段：仅筛选景区名称，输出JSON数组】
    spot_simple_list: [{"spot_name":"xxx","level":"4A","ticket":"20元"},...]
    user_pref: 会话偏好+本轮LLM提取条件合并dict，crowd、budget_max等
    user_query: 用户原始提问，偏好缺失时LLM依据它判断
    返回筛选之后景区名称list；失败返回原始全部
    """
    sys_prompt = """ 你是严格的景区筛选工具。
### 用户偏好 {}
### 用户原始提问 {}
### 筛选规则
1. 遍历全部输入景区，结合用户偏好和用户原始提问逐个判断是否保留；
2. 重点条件：亲子/人群适配、门票价格不超过偏好中的预算上限；门票"免费"即0元，预算为0时只保留免费景区；门票为"暂无门票信息"的景区，不能仅因价格缺失被排除，按其余条件判断；
3. 输出必须用json 代码块包裹一个纯数组，数组元素是从输入列表原样复制的景区名字符串；
4. 只有确实没有任何一个景区符合条件，才允许输出空数组：json\n[]\n；
5. 禁止写任何解释、说明、中文文字，只输出代码块。
### 输出示例（仅供格式参考，不要使用示例内容）
json
["广州动物园","广东科学中心"]
### 待筛选景区列表
{}
""".format(
        json.dumps(user_pref, ensure_ascii=False),
        user_query.strip() or "（无）",
        json.dumps(spot_simple_list, ensure_ascii=False, indent=2)
    )
    msg = [{"role":"system","content":sys_prompt}]
    msg.append({"role":"user","content":"执行筛选，严格输出带json包裹的 JSON 数组，不要多余文字。"})
    raw = generate_answer_by_api(msg, llm_cfg=llm_cfg, timeout=15)
    #优先匹配 markdown json 代码块
    match = re.search(r"json\s*\n(.*?)\n", raw, re.DOTALL)
    if match:
        json_text = match.group(1).strip()
    else:
        # 兜底：没有标记，直接尝试抓取字符串里面的 [] 裸 json
        logger.warning("【阶段 1】没有找到json 标记，尝试裸 JSON 解析 ")
        match_bare = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match_bare:
            json_text = match_bare.group(0).strip()
        else:
            logger.warning("【阶段 1】完全无法提取 json，降级全部候选")
            return [x["spot_name"] for x in spot_simple_list]
    try:
        name_arr = json.loads(json_text)
        if not isinstance(name_arr, list):
            logger.warning("【阶段 1 筛选结果不是 list，降级全部候选】")
            return [x["spot_name"] for x in spot_simple_list]
        logger.info(f"【两阶段第一阶段 LLM 筛选完成，符合条件景区列表：{name_arr}】")
        return name_arr
    except Exception as e:
        logger.warning(f"【阶段 1 筛选 JSON 解析失败 err={str(e)}，降级全部候选】")
        return [x["spot_name"] for x in spot_simple_list]



def _recommend_build_prompt_for_soft_filter(user_query, spot_detail_list, spot_chunk_dict, session_pref, history_digest: str = ""):
    batch_context = ""
    for d in spot_detail_list:
        spot_n = d["spot_name"]
        chunk_text = spot_chunk_dict.get(spot_n, "暂无知识库介绍资料")
        batch_context += f"""
==================== 【{spot_n}】
城市:{d['city']}
等级:{d['level']}
门票:{d['ticket']}
开放时间:{d['open_time']}
地址:{d['address']}
联系电话:{d['telephone']}
【景区知识库介绍原文】
{chunk_text}
====================
"""
    pref_text = json.dumps(session_pref, ensure_ascii=False, indent=2)
    sys_content = f"""
# 任务说明
下面是经过SQL+本地粗打分筛选之后得到的地级市景区候选集合。
当前会话用户筛选偏好：
{pref_text}
## 最近对话参考（仅用于衔接上下文，不是资料来源，禁止把其中的内容当作景区信息引用）
{history_digest or "（无）"}
## 硬性执行规则
1. **只允许使用下面给出的候选景区信息，禁止编造任何不在列表内的景区，禁止编造属性；**
2. SQL已经完成城市、等级硬过滤；预算、亲子人群、孩子年龄、景色风格、交通便利全部由你基于文本做软条件筛选；
3. 根据用户原始提问以及会话累积偏好做筛选、排序，给出推荐理由；理由必须严格参考【景区知识库介绍原文】。
4. 每个景区的介绍控制在2-3句话（核心亮点+适配理由即可），禁止长篇展开历史沿革、面积数据等背景信息。
## ⚠️【最重要强制约束，必须严格遵守】
第一步：从头到尾完整遍历**全部给到的每一个候选景区**，逐个判断该景区是否满足用户偏好条件；
第二步：**所有满足条件的景区，全部都要输出，绝对不能只输出1‑2个就结束回答；**
第三步：区分两组输出：
- 【首选推荐】：高度匹配用户全部偏好的景区；
- 【可考虑，但需确认】：部分条件达标，或者部分信息缺失的景区；
即使匹配度一般，只要大体符合条件，也必须放到【可考虑，但需确认】分组，不允许直接丢弃。
第四步：每一个输出景区都要写简短适配理由。
## 用户原始提问
{user_query}
## 候选景区资料
{batch_context}
""" + OUTPUT_FORMAT_RULE
    messages = [
        {"role": "system", "content": sys_content},
    ]
    messages.append({"role": "user", "content": f"请执行上面任务，完成景区筛选与输出，不要省略任何符合条件的景区。\n用户需求：{user_query}"})
    return messages


def _history_digest(history: list, max_msgs: int = 6, max_chars: int = 200) -> str:
    """把最近几轮对话压成摘要文本，供生成提示词衔接上下文（仅参考，不是资料来源）"""
    lines = []
    for m in (history or [])[-max_msgs:]:
        if not isinstance(m, dict):
            continue
        role = "用户" if m.get("role") == "user" else "客服"
        c = (m.get("content") or "").strip().replace("\n", " ")
        if c:
            lines.append(f"{role}：{c[:max_chars]}")
    return "\n".join(lines) or "（无）"


def build_chat_messages(user_query, rerank_docs, history, session_ctx):
    """
    构造LLM请求messages数组
    :param user_query: 用户当前提问
    :param rerank_docs: 重排后的检索文档片段
    :param history: 合并后的对话历史 merged_history_for_llm，格式[{"role":"xxx","content":"xxx"},...]
    :param session_ctx: 会话上下文
    :return: messages列表，用于generate_answer_by_api
    """
    # 1. 系统提示词：基础规则
    system_base = {
        "role": "system",
        "content": "基于下面的景区结构化信息，回答用户筛选问题，只使用给出的信息，不要编造内容。如果资料不存在该信息，回复暂无相关信息；优先列出符合筛选条件的景区名称，支持按等级、地区、门票、开放时间、特色项目筛选，禁止脑补景区不存在属性。" + OUTPUT_FORMAT_RULE
    }
    # 2. 拼接检索出来的RAG文档上下文，作为第二条system消息
    batch_context_text = ""
    for doc in rerank_docs:
        spot_name = doc.get("spot_name", "")
        chunk_text = doc.get("chunk_text", "")
        batch_context_text += f"【景区：{spot_name}】\n{chunk_text}\n\n"
    system_context = {
        "role": "system",
        "content": batch_context_text
    }
    # 3. 组装消息：系统基础指令 + 检索上下文 + 历史对话 + 当前用户提问
    messages = []
    messages.append(system_base)
    messages.append(system_context)
    # 追加历史对话（过滤空消息）
    if isinstance(history, list):
        for msg in history:
            if msg.get("role") and msg.get("content"):
                messages.append(msg)
    # 当前用户query
    messages.append({"role": "user", "content": user_query})
    return messages


def hybrid_query(
    user_query: str,
    chat_history: list,
    session_ctx: dict,
    session_id: str,
    total_context_token: int,
    llm_cfg: dict,
    return_metadata: bool = False,
    is_inner_sub_task: bool = False,
    curr_mem_mgr = None,
    pre_judge: dict | None = None
):

    """
    session_ctx必须初始化字段：
    session_ctx["user_preference"] = {}  #会话临时偏好，内存，会话销毁清空
    session_ctx["last_intent_type"] = "" #记录上一轮意图，用于推荐会话补充条件继承上下文
    is_inner_sub_task: True =多景区拆分出来的内部子任务，跳过磁盘短期记忆读取，子任务会话副本隔离
    【重要改动】
    入参新增 curr_mem_mgr: 传入当前会话 SessionMemoryManager 实例；
    本函数不再读取旧全局longterm_memory；持久记忆读写统一交给上层agent_reflection_loop；
    extract_memory_preference 需要外部传入 curr_mem_mgr 完成会话隔离持久化。
    """
    #初始化会话字段（防止新会话缺失key）
    if "user_preference" not in session_ctx:
        session_ctx["user_preference"] = {}
    if "last_intent_type" not in session_ctx:
        session_ctx["last_intent_type"] = ""

    # =========子任务开关：跳过磁盘short2读取，长期记忆直接从内存副本拿，避免重复IO =========
    if not is_inner_sub_task:
        recall_short_items = short2_memory.recall_by_session(session_id, total_context_token)
        # ========= 删除原来解析旧全局longterm_memory的整套代码 =========
        long_recall_items = []
    else:
        logger.info("[子任务模式启动] 禁用B方案多景区拆分")
        recall_short_items = []
        long_recall_items = []

    route_info = classify_query(user_query, session_ctx)
    route = route_info["route"]
    entities = route_info["entities"]
    base_entities = entities.copy()
    city_name = entities.get("city")
    target_level = entities.get("level")
    allow_cross_city = entities.get("allow_cross_city", False)
    logger.info(
        "路由判定 route=%s confidence=%.2f reason=%s entities=%s",
        route,
        route_info["confidence"],
        route_info["reason"],
        entities,
    )
    # 【筛选假设】识别层的概率性判断显式留痕：这些是"假设"而非事实，空结果时回流放宽重查
    if entities.get("ticket_free") or entities.get("budget_max") is not None or entities.get("exclude_spot_list"):
        logger.info(
            "【筛选假设】免费=%s 预算≤%s 排除=%s 等级=%s（假设来自关键词/LLM识别，非数据库事实）",
            entities.get("ticket_free"), entities.get("budget_max"),
            entities.get("exclude_spot_list"), entities.get("level"),
        )
    # 【修复】规划器已识别出景区而内部词典漏识别时，以注入实体为准：
    # clarify不得一票否决（星湖/古龙峡类简称——解析链能解析，词典形态表没见过）
    if pre_judge and pre_judge.get("spot_name") and not entities.get("spot_name"):
        entities["spot_name"] = pre_judge.get("spot_name")
        if pre_judge.get("city") and not entities.get("city"):
            entities["city"] = pre_judge.get("city")
            session_ctx["city"] = entities["city"]
        if pre_judge.get("scenic_level") and not entities.get("level"):
            entities["level"] = pre_judge.get("scenic_level")
        _intent_route = {"spot_detail": ROUTE_RAG, "rag_required": ROUTE_RAG,
                         "city_scenic_list": ROUTE_SCENIC_LIST, "scenic_count": ROUTE_SCENIC_COUNT,
                         "recommendation": ROUTE_RECOMMENDATION}
        _new_route = _intent_route.get(pre_judge.get("intent_type"), ROUTE_RAG)
        if route == ROUTE_CLARIFY:
            route = route_info["route"] = _new_route
            route_info["entities"] = entities
            logger.info(f"【clarify拦截】规划器已识别景区[{entities['spot_name']}]，按注入实体继续路由 route={route}")
    #推荐会话补充条件：route已经被classify_query改为ROUTE_RECOMMENDATION，执行会话偏好提取，仅更新内存，不持久化
    if route == ROUTE_RECOMMENDATION and session_ctx.get("last_intent_type") == ROUTE_RECOMMENDATION and city_name:
        logger.info("【推荐会话补充筛选条件，提取会话临时偏好，不写入持久longterm_memory】")
        extract_session_preference_update(user_query, session_ctx, llm_cfg=llm_cfg)


    merged_history_for_llm = chat_history.copy()
    merged_history_for_llm.extend(recall_short_items)
    merged_history_for_llm.extend(long_recall_items)

    if route in (ROUTE_GREETING, ROUTE_AGENT_PROFILE):
        old_focus_mode = session_ctx["focus_mode"]
        old_focus_round = session_ctx["focus_keep_round"]
        old_candidate_list = session_ctx["candidate_spot_list"]
        old_latest_subject = session_ctx["latest_subject"]
        if route == ROUTE_GREETING:
            ans = GREETING_ANSWER
        else:
            ans = AGENT_PROFILE_ANSWER
        session_ctx["focus_mode"] = old_focus_mode
        session_ctx["focus_keep_round"] = old_focus_round
        session_ctx["candidate_spot_list"] = old_candidate_list
        session_ctx["latest_subject"] = old_latest_subject
        #保存本轮意图
        session_ctx["last_intent_type"] = route
        return _query_result(ans, [], return_metadata, route, route_info["answer_type"])

    if session_ctx.get("focus_keep_round", 0) > 0 and route not in (ROUTE_GREETING, ROUTE_AGENT_PROFILE):
        session_ctx["focus_keep_round"] -= 1
        if session_ctx["focus_keep_round"] <= 0:
            session_ctx["focus_mode"] = False
            session_ctx["candidate_spot_list"] = []
            logger.info("[focus_mode]焦点轮次耗尽，关闭焦点模式，清空候选列表")

    user_spot = entities.get("spot_name")
    if user_spot:
        session_ctx["latest_subject"] = user_spot
        session_ctx["candidate_spot_list"] = []

    if route == ROUTE_SCENIC_LIST and city_name:
        sql_ctx = query_city_scenic_list(city_name, target_level)
        cand_spots = [x["spot_name"] for x in sql_ctx]
        _update_session_subject_by_spots(session_ctx, cand_spots)

    if entities.get("has_pronoun") and not entities.get("spot_name") and not entities.get("city"):
        answer = "没有找到上文提到的景区，请告诉我具体景区或者城市名称。"
        result = _query_result(answer, [], return_metadata, "clarify", "clarification")
        session_ctx["last_intent_type"] = ROUTE_CLARIFY
        return _attach_route_metadata(result, return_metadata, route_info)

    # 命中记忆触发词：记住我，传入curr_mem_mgr做会话隔离持久化
    # 不接收返回值：函数内部原地修改session_ctx；其@safe_call兜底返回None，赋值回去会污染session_ctx
    if entities.get("trigger_memory_instruction"):
        extract_memory_preference(user_query, session_ctx, curr_mem_mgr, llm_cfg)

    need_llm_judge = False
    judge_res = None
    if route == ROUTE_RAG or route == ROUTE_RECOMMENDATION:
        if entities.get("city") or entities.get("spot_name") or entities.get("trigger_memory_instruction"):
            need_llm_judge = True

    sub_intent_results = []
    ALL_TRIGGER_WORDS = PRONOUNS + CONTEXT_CONTINUE_WORDS
    if need_llm_judge:
        try:
            # pre_judge 由问句规划Agent注入时直接使用，跳过内部Judge调用（避免子问题重复消耗LLM）
            judge_res = pre_judge if pre_judge is not None else call_llm_route_judge(user_query, session_ctx, llm_cfg=llm_cfg, base_entities=base_entities)

            intent_list = judge_res.get("intent_list", [judge_res["intent_type"]])[:3]
            spot_name_list_raw = judge_res.get("spot_name_list", [])
            logger.info(f"【LLM‑Judge 子意图总列表】intent_list={intent_list}, spot_name_list={spot_name_list_raw}")
            MAX_MULTI_SPOT = 3
            # ======================【B方案：拆分为干净独立子问题】======================
            # 【核心修改】is_inner_sub_task=True时，直接禁用多景区拆分，跳过B方案
            if (not is_inner_sub_task and not judge_res.get("_comparison")
                    and isinstance(spot_name_list_raw, list) and len(spot_name_list_raw) >= 2):
                # 1. 全局别名归一化，全部转为数据库标准全名
                norm_spot_list = []
                session_alias = session_ctx.get("session_alias_map", {})
                for raw_name in spot_name_list_raw:
                    std_name, mapped_flag = scenic_alias_normalize(raw_name, session_alias)
                    norm_spot_list.append(std_name)
                    logger.info(f"【多景区别名归一化】原始名称={raw_name} → 标准名称={std_name}, 是否映射={mapped_flag}")
                # 超限不再整体拒绝：处理前MAX_MULTI_SPOT个，剩余景区在合并回答中提示单独提问
                over_limit_spots = norm_spot_list[MAX_MULTI_SPOT:]
                if over_limit_spots:
                    logger.info(
                        f"【多景区超限】共{len(norm_spot_list)}个，本次处理前{MAX_MULTI_SPOT}个，剩余{len(over_limit_spots)}个提示单独提问")
                multi_spot_task_list = norm_spot_list[:MAX_MULTI_SPOT]
                logger.info(f"【B方案‑拆分子任务】原始列表={spot_name_list_raw},归一化任务列表={multi_spot_task_list}")
                # 2.提取公共约束：移除全部景区实体名称，保留用户查询条件
                common_requirement = extract_common_requirement(user_query, spot_name_list_raw)
                logger.info(f"【多景区子任务公共约束】：{common_requirement}")
                sub_answers = []
                sub_spots_all = []
                sub_docs_all = []
                # 3.循环生成【干净独立子query】公共约束 + 当前唯一标准景区名称，句子不再混杂其它景区，避免实体污染
                for idx, one_spot in enumerate(multi_spot_task_list):
                    logger.info(f"【B‑子任务】序号{idx + 1}/{len(multi_spot_task_list)}，待处理景区：{one_spot}")
                    try:
                        sub_session_ctx = copy.deepcopy(session_ctx)
                        sub_query_text = f"{common_requirement}【{one_spot}】"
                        sub_total_token = total_context_token
                        logger.info("【子任务启动】序号=%d", idx + 1)
                        sub_ans, sub_docs, sub_meta = hybrid_query(
                            user_query=sub_query_text,
                            chat_history=[],
                            session_ctx=sub_session_ctx,
                            llm_cfg=llm_cfg,
                            session_id=session_id,
                            total_context_token=sub_total_token,
                            return_metadata=True,
                            is_inner_sub_task=True,
                            curr_mem_mgr=curr_mem_mgr
                        )
                        sub_answers.append(sub_ans)
                        # 【修复】子任务的知识库chunk必须向上传播进主流程资料：
                        # 否则溯源核验拿不到子任务的资料，会把真实有据的内容判成"无出处"并删除
                        sub_docs_all.extend(sub_docs or [])
                        sub_spots_all.extend(sub_meta.get("spots", []))
                        logger.info(f"【子任务执行成功】景区:{one_spot}")
                    except LLMServiceError:
                        raise
                    except Exception as sub_task_err:
                        logger.error("【多景区子任务失败】type=%s", type(sub_task_err).__name__)
                        fail_text = f"\n【{one_spot}】：系统暂时获取该景区信息失败。"
                        sub_answers.append(fail_text)
                logger.info(f"【B方案循环结束】总共待处理{len(multi_spot_task_list)}，实际收集回答数量：{len(sub_answers)}")
                # 4.LLM合并全部子回答
                merged_ans = _llm_merge_multi_result(user_query, sub_answers, llm_cfg=llm_cfg)
                if over_limit_spots:
                    merged_ans += (
                        f"\n\n另外，单次最多同时处理{MAX_MULTI_SPOT}个景区，"
                        f"以下景区本次未查询：{'、'.join(over_limit_spots)}。请分开提问，我再为您逐一查询。"
                    )

                sub_intent_results.append({"ans": merged_ans, "spots": sub_spots_all})
                final_spots = []
                for item in sub_intent_results:
                    final_spots.extend(item["spots"])
                # 全部子任务结束才更新原始主会话上下文，子任务副本直接丢弃
                _update_session_subject_by_spots(session_ctx, final_spots)
                result = _query_result(merged_ans, sub_docs_all, return_metadata, "multi_intent", "multi_intent",
                                       spots=final_spots)
                session_ctx["last_intent_type"] = ROUTE_RAG
                return _attach_route_metadata(result, return_metadata, route_info)
            # ======================B方案多景区分支结束，下面保留原有单意图处理逻辑 ======================
            for intent in intent_list:
                logger.info(f"【LLM‑Judge 处理子意图】当前intent={intent}")
                sub_judge = judge_res.copy()
                sub_judge["intent_type"] = intent
                # 【修复·加强】推荐意图一律不回填参考主体：ref_spot仅用于"像X一样"类比，
                # 回填会把排除对象/顺带提及的景区错当类比参考（"不要白云山"→类比白云山）。
                # 类比场景由Judge/规划器自行填写spot_name，本地只兜底city。
                if intent == "recommendation":
                    if base_entities.get("spot_name"):
                        logger.info(f"【实体回填拦截】推荐意图不回填参考主体（本地识别到{base_entities['spot_name']}，仅作排除/顺带提及处理）")
                elif not sub_judge.get("spot_name") and base_entities.get("spot_name"):
                    sub_judge["spot_name"] = base_entities["spot_name"]
                if not sub_judge.get("city") and base_entities.get("city"):
                    sub_judge["city"] = base_entities["city"]
                has_pronoun_sub = any(p in user_query for p in ALL_TRIGGER_WORDS)
                sub_spot = sub_judge.get("spot_name", "")
                sub_city = sub_judge.get("city", "")
                if has_pronoun_sub and (not sub_spot) and (not sub_city):
                    logger.warning("LLM‑Judge子意图命中纯代词，二次拦截，跳过该子意图")
                    continue
                try:
                    if intent == "spot_detail":
                        sp = sub_judge["spot_name"]
                        tf = sub_judge["target_fields"]
                        spot_detail = get_single_spot_detail(sp)
                        spot_mapped_flag = sub_judge.get("_spot_mapped_flag", False)
                        origin_input_spot = sub_judge.get("_origin_spot_input", "")
                        sub_note = ""
                        if spot_mapped_flag and origin_input_spot and origin_input_spot != sp:
                            # 【修复】"发生过归一"≠"无独立条目"（圆明新园→珠海市圆明新园是正常简写归一）：
                            # 只有原名字确实不在主表时才提示，避免给已收录景区扣上"无独立资料"的帽子
                            _probe = None
                            try:
                                _conn_n = sqlite3.connect(DB_PATH)
                                _probe = _conn_n.execute(
                                    "SELECT 1 FROM scenic_spots WHERE spot_name LIKE ? LIMIT 1",
                                    (f"%{origin_input_spot}%",)).fetchone()
                                _conn_n.close()
                            except Exception:
                                _probe = None
                            if not _probe:
                                sub_note = f"\n> 提示：数据库没有【{origin_input_spot}】独立资料，返回所属度假区信息供参考。"
                        detail_lines = {
                            "ticket": f"门票信息：{spot_detail['ticket']}",
                            "open_time": f"开放时间：{spot_detail['open_time']}",
                            "address": f"地址：{spot_detail['address']}",
                            "level": f"景区等级：{spot_detail['level']}",
                            "telephone": f"联系电话：{spot_detail['telephone']}",
                        }
                        ans_part = "\n".join([f"【{spot_detail['spot_name']}】"] + [detail_lines[f] for f in tf])
                        ans_part += sub_note
                        intent_docs = []
                        has_rag_demand = any(k in user_query for k in RAG_KEYWORDS)
                        if has_rag_demand:
                            q_vec = get_query_embedding(sp, emb_tokenizer, emb_model)
                            vec_res = vector_search_numpy(q_vec, vector_array, meta_data, TOP_K_VECTOR)
                            short_spot_name, _ = scenic_alias_normalize(sp)
                            kw_res = keyword_search_db(short_spot_name, TOP_K_KEYWORD)
                            merge = vec_res + kw_res
                            unique_chunk = []
                            seen = set()
                            for item in merge:
                                ck = item.get("spot_name", "") + "|||" + item.get("chunk_text", "")
                                if ck not in seen:
                                    seen.add(ck)
                                    unique_chunk.append(item)
                            filtered = post_filter_docs(unique_chunk, session_ctx.get("city"), session_ctx["allow_cross_city"], [sp])
                            rerank_chunk = rerank(user_query, filtered, RERANK_TOP)
                            msg = build_chat_messages(user_query, rerank_chunk, merged_history_for_llm, session_ctx)
                            rag_part = generate_answer_by_api(msg, llm_cfg, user_query)

                            ans_part = ans_part + "\n\n====游玩参考介绍====\n" + rag_part
                            intent_docs = rerank_chunk
                        sub_intent_results.append({"ans": ans_part, "spots": [spot_detail], "docs": intent_docs})
                    elif intent == "city_scenic_list":
                        c = sub_judge["city"]
                        lv = sub_judge["scenic_level"]
                        sql_context = query_city_scenic_list(c, lv)
                        if not sql_context:
                            lv_text = f"{lv}级" if lv else "A级"
                            ans_part = f"{c}{lv_text}旅游景区共有0个。"
                        else:
                            lv_text = f"{lv}级" if lv else "A级"
                            scenic_lines = [f"{index}. {item['spot_name']}（{item['level'] or '等级无'}）" for index, item in enumerate(sql_context, start=1)]
                            ans_part = "\n".join([f"{c}{lv_text}旅游景区共有{len(sql_context)}个："] + scenic_lines)
                            sub_intent_results.append({"ans": ans_part, "spots": sql_context, "docs": sql_context})
                    elif intent == "scenic_count":
                        c = sub_judge["city"]
                        lv = sub_judge["scenic_level"]
                        total = query_scenic_count(c, lv)
                        level_text = f"{lv}级" if lv else "A级"
                        if total <= 0:
                            ans_part = f"{c}暂未查询到{level_text}旅游景区记录。"
                        else:
                            ans_part = f"{c}一共有{total}个{level_text}旅游景区。"
                        sub_intent_results.append({"ans": ans_part, "spots": [], "docs": []})
                    elif intent == "recommendation":
                        rec_city = sub_judge.get("city", "")
                        ref_spot = sub_judge.get("spot_name", "")
                        if not rec_city and not ref_spot:
                            logger.warning("recommendation子意图：缺少city或者参考spot，条件不足，跳过")
                            continue
                        rec_spots_detail = []
                        rec_ans = ""
                        if rec_city and not ref_spot:
                            logger.info(f"【推荐模式】城市偏好推荐 city={rec_city}")
                            filter_cond = extract_recommend_filter_condition(user_query, session_ctx, llm_cfg=llm_cfg)

                            sql_filtered_list = []
                            if filter_cond is not None:
                                cond_city = filter_cond.get("city") or rec_city
                                cond_level = filter_cond.get("scenic_level")
                                logger.info(f"【SQL硬过滤参数】city={cond_city}, level={cond_level}；budget_max(预算)交由LLM软筛选")
                                sql_filtered_list = query_city_scenic_filtered(cond_city, cond_level, budget_max=(filter_cond or {}).get('budget_max'))
                                logger.info(f"【SQL硬过滤返回候选数量】{len(sql_filtered_list)}")
                            if filter_cond is None or len(sql_filtered_list) == 0:
                                logger.warning("城市推荐：LLM条件提取失败或者SQL候选为空，启用城市全量兜底查询")
                                sql_filtered_list = query_city_scenic_list(rec_city, filter_cond.get("scenic_level") if filter_cond else None)
                            # 【新增】排除类推荐：用户明确不要的景区从候选中确定性剔除（Judge排除项+条件提取排除项）
                            exclude_set = set()
                            for name in (sub_judge.get("exclude_spot_list") or []):
                                nm = resolve_spot_mention(str(name))
                                if nm:
                                    exclude_set.add(nm)
                            if filter_cond:
                                for name in (filter_cond.get("exclude_spots") or []):
                                    nm = resolve_spot_mention(str(name))
                                    if nm:
                                        exclude_set.add(nm)
                            if exclude_set:
                                before_cnt = len(sql_filtered_list)
                                sql_filtered_list = [s for s in sql_filtered_list if s["spot_name"] not in exclude_set]
                                logger.info(f"【推荐排除】已排除{sorted(exclude_set)}，候选{before_cnt}→{len(sql_filtered_list)}")
                            # 【新增】把本轮提取的条件持久进会话偏好，后续"复述要求/按之前条件"类提问可继承
                            if filter_cond:
                                session_ctx["user_preference"] = {
                                    **session_ctx["user_preference"],
                                    **{k: v for k, v in filter_cond.items() if v not in (None, "")},
                                }
                            # 合并会话偏好与首轮LLM提取条件(budget_max/crowd等)，保证首轮推荐也带上预算/人群约束
                            merged_pref = {**session_ctx["user_preference"], **(filter_cond or {})}
                            # =========本地粗打分排序，过滤低分无关景区，只保留TOP12高分候选送入LLM软筛选=========
                            top_score_candidates = score_candidate_spot_list(sql_filtered_list,
                                                                             merged_pref, user_query)
                            # U‑shuffle重排，规避lost‑in‑the‑middle
                            top_score_candidates = reorder_for_llm_attention(top_score_candidates)
                            logger.info(
                                f"【推荐本地粗打分完成】原始候选{len(sql_filtered_list)}，打分+重排后送入LLM候选数量：{len(top_score_candidates)}")
                            # --------两阶段第一步：只传简易信息，让LLM输出符合条件的景区名称JSON数组---------
                            spot_simple_input = []
                            for s in top_score_candidates:
                                spot_simple_input.append({
                                    "spot_name": s["spot_name"],
                                    "level": s["level"],
                                    "ticket": s["ticket"]
                                })
                            filter_name_list = llm_filter_spot_name_list(spot_simple_input, merged_pref, llm_cfg=llm_cfg, user_query=user_query)

                            # 根据返回的景区名称过滤，只保留筛选命中的
                            filtered_top_candidates = [x for x in top_score_candidates if
                                                       x["spot_name"] in set(filter_name_list)]
                            if not filtered_top_candidates and top_score_candidates:
                                logger.warning("【两阶段第一阶段筛选为空，降级使用全部粗打分候选，交由第二阶段软筛选分组】")
                                filtered_top_candidates = list(top_score_candidates)
                            logger.info(f"【两阶段过滤后剩下待生成文案景区数量：{len(filtered_top_candidates)}】")
                            # 【提速】最终文案候选上限6个：产出体量决定核验/合并的处理耗时
                            filtered_top_candidates = filtered_top_candidates[:6]
                            spot_detail_list = []
                            spot_chunk_dict = dict()
                            for s_item in filtered_top_candidates:
                                spot_n = s_item["spot_name"]
                                d = get_single_spot_detail(spot_n)
                                spot_detail_list.append(d)
                                chunk_txt = _retrieve_spot_knowledge_chunk(spot_n, user_query)
                                spot_chunk_dict[spot_n] = chunk_txt
                            if len(spot_detail_list) == 0:
                                rec_ans = f"未查询到{rec_city}的A级景区，无法为你做推荐。"
                                sub_intent_results.append({"ans": rec_ans, "spots": rec_spots_detail})
                                continue
                            msg = _recommend_build_prompt_for_soft_filter(user_query, spot_detail_list, spot_chunk_dict,
                                                                          session_ctx["user_preference"],
                                                                          history_digest=_history_digest(merged_history_for_llm))
                            rec_ans = generate_answer_by_api(msg, llm_cfg=llm_cfg)

                            rec_spots_detail = spot_detail_list
                            sub_intent_results.append({"ans": rec_ans, "spots": rec_spots_detail})
                        elif ref_spot:
                            logger.info(f"【推荐模式】类比相似推荐 ref_spot={ref_spot}")
                            q_vec = get_query_embedding(ref_spot, emb_tokenizer, emb_model)
                            vec_res = vector_search_numpy(q_vec, vector_array, meta_data, TOP_K_VECTOR * 2)
                            sim_raw = [x for x in vec_res if x.get("spot_name", "") != ref_spot][:8]
                            for sim_item in sim_raw:
                                d = get_single_spot_detail(sim_item["spot_name"])
                                rec_spots_detail.append(d)
                            if len(rec_spots_detail) == 0:
                                rec_ans = f"找不到和【{ref_spot}】风格相近的景区，无法类比推荐。"
                                sub_intent_results.append({"ans": rec_ans, "spots": rec_spots_detail})
                                continue
                            batch_context = ""
                            for d in rec_spots_detail:
                                chunk_text = _retrieve_spot_knowledge_chunk(d["spot_name"], user_query)
                                batch_context += f"""
【{d['spot_name']}】
城市:{d['city']}
等级:{d['level']}
门票:{d['ticket']}
开放时间:{d['open_time']}
地址:{d['address']}
联系电话:{d['telephone']}
【知识库介绍】
{chunk_text}
"""
                            rec_msg = [
                                {"role": "system", "content": f"参考样例景区：{ref_spot}。下面是检索得到的一批风格相似景区真实资料，请严格基于资料类比推荐，禁止编造。\n【最近对话参考（仅衔接上下文，非资料来源）】\n{_history_digest(merged_history_for_llm)}\n{batch_context}" + OUTPUT_FORMAT_RULE},
                                {"role": "user", "content": user_query}
                            ]
                            rec_ans = generate_answer_by_api(rec_msg, llm_cfg=llm_cfg)

                            sub_intent_results.append({"ans": rec_ans, "spots": rec_spots_detail})
                    elif intent == "rag_required":
                        sp = sub_judge.get("spot_name", "")
                        # 【新增】对比类多景区：白名单覆盖全部对比对象，嵌入用完整问句（含双方景区名）
                        multi_list = [s for s in (sub_judge.get("spot_name_list") or []) if s][:4]
                        if not sp and not multi_list:
                            logger.warning("rag_required子意图缺少spot_name，跳过该子意图")
                            continue
                        if len(multi_list) >= 2:
                            q_vec = get_query_embedding(user_query, emb_tokenizer, emb_model)
                            # 【修复】关键词用核心名检索：全称LIKE（%韶关市丹霞山景区%）在正文里匹配不到
                            try:
                                from route_classifier import _strip_spot_affixes
                                _cities, _, _ = load_entities()
                                core_names = [_strip_spot_affixes(w, list(_cities)) or w for w in multi_list]
                            except Exception:
                                core_names = multi_list
                            kw_res = []
                            for w in core_names:
                                kw_res.extend(keyword_search_db(w, TOP_K_KEYWORD))
                            whitelist = multi_list
                        else:
                            q_vec = get_query_embedding(sp, emb_tokenizer, emb_model)
                            short_spot_name, _ = scenic_alias_normalize(sp)
                            kw_res = keyword_search_db(short_spot_name, TOP_K_KEYWORD)
                            whitelist = [sp]
                        vec_res = vector_search_numpy(q_vec, vector_array, meta_data, TOP_K_VECTOR)
                        merge = vec_res + kw_res
                        unique_chunk = []
                        seen = set()
                        for item in merge:
                            ck = item.get("spot_name", "") + "|||" + item.get("chunk_text", "")
                            if ck not in seen:
                                seen.add(ck)
                                unique_chunk.append(item)
                        filtered = post_filter_docs(unique_chunk, session_ctx.get("city"), session_ctx["allow_cross_city"], whitelist)
                        if len(multi_list) >= 2:
                            # 【修复】对比类配额重排：每景区各取前4条，防止单边挤占TOP5导致另一边资料缺失
                            rerank_chunk = []
                            for w in whitelist:
                                w_docs = [d for d in filtered if d.get("spot_name") == w]
                                rerank_chunk.extend(rerank(user_query, w_docs, 4))
                        else:
                            rerank_chunk = rerank(user_query, filtered, RERANK_TOP)
                        msg = build_chat_messages(user_query, rerank_chunk, merged_history_for_llm, session_ctx)
                        rag_ans = (generate_answer_by_api(msg, llm_cfg, user_query, max_tokens=800) if len(multi_list) >= 2 else generate_answer_by_api(msg, llm_cfg, user_query))

                        sub_intent_results.append({"ans": rag_ans, "spots": _aggregate_spot_from_docs(rerank_chunk), "docs": rerank_chunk})
                    else:
                        logger.warning(f"未知子意图 {intent}，直接跳过")
                        continue
                except LLMServiceError:
                    raise
                except Exception as sub_e:
                    logger.warning(f"【子意图执行失败】intent={intent}, err={str(sub_e)}", exc_info=True)
                    continue
        except LLMServiceError:
            raise
        except Exception as e:
            logger.warning(f"LLM judge调用失败，降级普通RAG，err={str(e)}", exc_info=True)
    if len(sub_intent_results) > 0:
        final_ans = "\n\n".join([x["ans"] for x in sub_intent_results])
        merge_spots = []
        merge_docs = []
        for x in sub_intent_results:
            merge_spots.extend(x["spots"])
            merge_docs.extend(x.get("docs") or [])
        _update_session_subject_by_spots(session_ctx, merge_spots)
        result = _query_result(final_ans, merge_docs, return_metadata, "multi_intent", "multi_intent", spots=merge_spots)
        # 按实际命中的意图类型记录：推荐意图后，后续追问才能走classify_query的城市继承分支
        if need_llm_judge and "recommendation" in intent_list:
            session_ctx["last_intent_type"] = ROUTE_RECOMMENDATION
        else:
            session_ctx["last_intent_type"] = ROUTE_RAG
        return _attach_route_metadata(result, return_metadata, route_info)

    if route == ROUTE_CLARIFY:
        hint = route_info.get("answer_hint")
        if hint:
            answer = hint
        elif entities.get("invalid_level"):
            answer = f"目前系统支持3A/4A/5A级景区，不支持{entities['invalid_level']}。"
        else:
            answer = "请提供广东省具体地级市（例如广州、韶关）或者景区名称，我再为您处理。"
        result = _query_result(answer, [], return_metadata, "clarify", "clarification")
        session_ctx["last_intent_type"] = ROUTE_CLARIFY
        return _attach_route_metadata(result, return_metadata, route_info)

    if route == ROUTE_SCENIC_LIST or route == ROUTE_SCENIC_COUNT:
        # 【清单续传】"继续/剩下的景区呢/下一页"：上一轮分页清单未展示完时按断点出下一批，不再重查重答
        if entities.get("list_continue") and isinstance(session_ctx.get("list_pages"), dict):
            lp = session_ctx["list_pages"]
            lp_city, lp_level = lp.get("city"), lp.get("level")
            full_list = query_city_scenic_list(lp_city, lp_level)
            offset = int(lp.get("offset") or 0)
            batch = full_list[offset:offset + 15]
            if not batch:
                session_ctx.pop("list_pages", None)
                answer = f"{lp_city}的A级旅游景区清单已全部展示完毕。"
                result = _query_result(answer, [], return_metadata, "scenic_list", "scenic_list")
                session_ctx["last_intent_type"] = ROUTE_SCENIC_LIST
                return _attach_route_metadata(result, return_metadata, route_info)
            remaining_after = full_list[offset + 15:]
            start_no = offset + 1
            batch_no = offset // 15 + 1
            scenic_lines = [
                f"{start_no + i}. {d['spot_name']}（{d.get('level') or '等级无'}）门票：{d.get('ticket') or '暂无'}"
                for i, d in enumerate(batch)
            ]
            answer = (f"{lp_city}A级旅游景区（第{batch_no}批，第{start_no}-{start_no + len(batch) - 1}家，"
                      f"共{len(full_list)}家）：\n" + "\n".join(scenic_lines))
            if remaining_after:
                session_ctx["list_pages"] = {"city": lp_city, "level": lp_level,
                                             "offset": offset + 15, "total": len(full_list)}
                answer += f"\n以上为第{batch_no}批，还剩{len(remaining_after)}家，如需继续查看请说“继续”。"
            else:
                session_ctx.pop("list_pages", None)
                answer += f"\n以上为最后一批，{lp_city}A级景区已全部展示完毕（共{len(full_list)}家）。"
            result = _query_result(answer, batch, return_metadata, "scenic_list", "scenic_list", batch)
            session_ctx["last_intent_type"] = ROUTE_SCENIC_LIST
            return _attach_route_metadata(result, return_metadata, route_info)

        # 【新增】多等级分别统计/列出："广州4A和5A景区分别有多少个"只答一个等级的问题
        level_list = [lv for lv in dict.fromkeys(entities.get("level_list") or []) if lv in ("3A", "4A", "5A")]
        # 【新增】免费/预算语义 → 清单工具按门票维度过滤
        ticket_free = bool(entities.get("ticket_free"))
        budget_max = entities.get("budget_max")
        level_text = f"{target_level}级" if target_level else "A级"
        filter_desc = ""

        # 第一轮：带筛选条件查询
        if budget_max is not None:
            filter_desc = f"门票不超过{budget_max}元"
            scope_text = f"{city_name}{filter_desc}的{level_text}旅游景区"
            sql_context, _unknown = query_city_scenic_list_budget(city_name, target_level, budget_max)
        elif ticket_free:
            filter_desc = "门票免费（0元）"
            scope_text = f"{city_name}{filter_desc}的{level_text}旅游景区"
            sql_context = query_city_scenic_list(city_name, target_level, ticket_free=True)
        else:
            scope_text = f"{city_name}{level_text}旅游景区"
            sql_context = query_city_scenic_list(city_name, target_level)

        # 【空结果回流】筛选条件未命中 → 放弃筛选条件，查全量+票价供参考
        filter_fallback = False
        if not sql_context and filter_desc:
            filter_fallback = True
            scope_text = f"{city_name}{level_text}旅游景区"
            sql_context = query_city_scenic_list(city_name, target_level)

        unknown_note = ""
        if filter_fallback:
            no_match_note = f"未查到{filter_desc}的景区（相关筛选条件数据库未收录）。以下为{scope_text}全部景区及票价供参考："
        else:
            no_match_note = f"没有查询到{scope_text}记录。请确认筛选条件或更换城市。"

        # count 分支：只报数量
        if route == ROUTE_SCENIC_COUNT:
            total = len(sql_context)
            if total <= 0:
                answer = no_match_note
            elif filter_fallback:
                answer = f"未查到{filter_desc}的景区。{scope_text}一共有{total}个（票价供参考）。"
            else:
                answer = f"{scope_text}一共有{total}个。"
            result = _query_result(answer, [], return_metadata, "scenic_count", "scenic_count")
            session_ctx["last_intent_type"] = ROUTE_SCENIC_COUNT
            return _attach_route_metadata(result, return_metadata, route_info)

        _update_session_subject_by_spots(session_ctx, sql_context)

        # 【分页】超过15个时只展示前15个，余量存入session_ctx供续传
        total = len(sql_context)
        shown_batch = sql_context[:15]
        remaining_batch = sql_context[15:]

        if remaining_batch:
            session_ctx["list_pages"] = {
                "city": city_name, "level": target_level,
                "offset": 15, "total": total,
            }

        scenic_lines = [
            f"{i}. {d['spot_name']}（{d.get('level') or '等级无'}）门票：{d.get('ticket') or '暂无'}"
            for i, d in enumerate(shown_batch, start=1)
        ]

        if not shown_batch:
            answer = no_match_note
        elif filter_fallback:
            answer = no_match_note + "\n".join([""] + scenic_lines)
            if remaining_batch:
                answer += f"\n以上为前15家（共{total}家），如需查看其余景区请继续询问。"
        else:
            header = f"{scope_text}一共有{total}个："
            answer = "\n".join([header] + scenic_lines)
            if remaining_batch:
                answer += f"\n以上为前15家（共{total}家），如需查看其余景区请继续询问。"

        result = _query_result(
            answer,
            shown_batch,
            return_metadata,
            "scenic_list",
            "scenic_list",
            shown_batch,
        )
        session_ctx["last_intent_type"] = ROUTE_SCENIC_LIST
        return _attach_route_metadata(result, return_metadata, route_info)

    if route == ROUTE_RAG:
        # 标准RAG单景区检索链路
        user_spot = entities.get("spot_name")
        # 【修复】有景区实体时以实体为检索锚点：省略式追问（"周末呢?"）原句无语义，
        # 用原句嵌入会检索出无关chunk再被白名单滤空，最终LLM只能凭历史乱接主体
        if user_spot:
            anchor_spot, _ = scenic_alias_normalize(user_spot)
            q_vec = get_query_embedding(anchor_spot, emb_tokenizer, emb_model)
            vec_res = vector_search_numpy(q_vec, vector_array, meta_data, TOP_K_VECTOR)
            kw_res = keyword_search_db(anchor_spot, TOP_K_KEYWORD)
        else:
            q_vec = get_query_embedding(user_query, emb_tokenizer, emb_model)
            vec_res = vector_search_numpy(q_vec, vector_array, meta_data, TOP_K_VECTOR)
            kw_res = keyword_search_db(user_query, TOP_K_KEYWORD)
        merge = vec_res + kw_res
        unique_chunk = []
        seen = set()
        for item in merge:
            ck = item.get("spot_name", "") + "|||" + item.get("chunk_text", "")
            if ck not in seen:
                seen.add(ck)
                unique_chunk.append(item)
        filtered = post_filter_docs(unique_chunk, city_name, allow_cross_city, [user_spot] if user_spot else [])
        rerank_chunk = rerank(user_query, filtered, RERANK_TOP)
        msg = build_chat_messages(user_query, rerank_chunk, merged_history_for_llm, session_ctx)
        answer = generate_answer_by_api(msg, llm_cfg, user_query)

        spots = _aggregate_spot_from_docs(rerank_chunk)
        _update_session_subject_by_spots(session_ctx, spots)
        result = _query_result(answer, rerank_chunk, return_metadata, route, "rag_answer", spots=spots)
        session_ctx["last_intent_type"] = ROUTE_RAG
        return _attach_route_metadata(result, return_metadata, route_info)

    if route == ROUTE_RECOMMENDATION:
        # 兜底推荐分支（LLM‑Judge未触发时，直接走推荐逻辑）
        rec_city = city_name
        ref_spot = entities.get("spot_name")
        rec_spots_detail = []
        rec_ans = ""
        if rec_city and not ref_spot:
            logger.info(f"【兜底推荐模式】城市偏好推荐 city={rec_city}")
            filter_cond = extract_recommend_filter_condition(user_query, session_ctx, llm_cfg=llm_cfg)

            sql_filtered_list = []
            if filter_cond is not None:
                cond_city = filter_cond.get("city") or rec_city
                cond_level = filter_cond.get("scenic_level")
                logger.info(f"【兜底SQL硬过滤参数】city={cond_city}, level={cond_level}；budget_max交由LLM软筛选")
                sql_filtered_list = query_city_scenic_filtered(cond_city, cond_level, budget_max=(filter_cond or {}).get('budget_max'))
            if filter_cond is None or len(sql_filtered_list) == 0:
                logger.warning("兜底城市推荐：条件提取失败或者SQL候选为空，使用城市全量景区")
                sql_filtered_list = query_city_scenic_list(rec_city,
                                                           filter_cond.get("scenic_level") if filter_cond else None)
            # 【新增】兜底分支排除类推荐：确定性剔除用户不要的景区
            exclude_set = set()
            if filter_cond:
                for name in (filter_cond.get("exclude_spots") or []):
                    nm = resolve_spot_mention(str(name))
                    if nm:
                        exclude_set.add(nm)
            if exclude_set:
                before_cnt = len(sql_filtered_list)
                sql_filtered_list = [s for s in sql_filtered_list if s["spot_name"] not in exclude_set]
                logger.info(f"【兜底推荐排除】已排除{sorted(exclude_set)}，候选{before_cnt}→{len(sql_filtered_list)}")
            # 【新增】兜底分支同样把本轮条件持久进会话偏好
            if filter_cond:
                session_ctx["user_preference"] = {
                    **session_ctx["user_preference"],
                    **{k: v for k, v in filter_cond.items() if v not in (None, "")},
                }
            # =========兜底分支同样执行本地粗打分+U型重排（合并filter_cond保证首轮预算约束生效）=========
            merged_pref = {**session_ctx["user_preference"], **(filter_cond or {})}
            top_score_candidates = score_candidate_spot_list(sql_filtered_list, merged_pref, user_query)
            top_score_candidates = reorder_for_llm_attention(top_score_candidates)
            logger.info(f"【兜底推荐本地粗打分完成】原始候选{len(sql_filtered_list)}，打分+重排后送入LLM候选数量：{len(top_score_candidates)}")
            # 【提速】最终文案候选上限6个
            top_score_candidates = top_score_candidates[:6]
            spot_detail_list = []
            spot_chunk_dict = dict()
            for s_item in top_score_candidates:
                spot_n = s_item["spot_name"]
                d = get_single_spot_detail(spot_n)
                spot_detail_list.append(d)
                chunk_txt = _retrieve_spot_knowledge_chunk(spot_n, user_query)
                spot_chunk_dict[spot_n] = chunk_txt
            if len(spot_detail_list) == 0:
                rec_ans = f"未查询到{rec_city}的A级景区，无法为你做推荐。"
                result = _query_result(rec_ans, [], return_metadata, route, "recommendation", spots=rec_spots_detail)
                session_ctx["last_intent_type"] = ROUTE_RECOMMENDATION
                return _attach_route_metadata(result, return_metadata, route_info)
            msg = _recommend_build_prompt_for_soft_filter(user_query, spot_detail_list, spot_chunk_dict,
                                                          session_ctx["user_preference"],
                                                          history_digest=_history_digest(merged_history_for_llm))
            rec_ans = generate_answer_by_api(msg, llm_cfg=llm_cfg)

            rec_spots_detail = spot_detail_list
        elif ref_spot:
            logger.info(f"【兜底推荐模式】类比相似推荐 ref_spot={ref_spot}")
            q_vec = get_query_embedding(ref_spot, emb_tokenizer, emb_model)
            vec_res = vector_search_numpy(q_vec, vector_array, meta_data, TOP_K_VECTOR * 2)
            sim_raw = [x for x in vec_res if x.get("spot_name", "") != ref_spot][:8]
            for sim_item in sim_raw:
                d = get_single_spot_detail(sim_item["spot_name"])
                rec_spots_detail.append(d)
            if len(rec_spots_detail) == 0:
                rec_ans = f"找不到和【{ref_spot}】风格相近的景区，无法类比推荐。"
                result = _query_result(rec_ans, [], return_metadata, route, "recommendation", spots=rec_spots_detail)
                session_ctx["last_intent_type"] = ROUTE_RECOMMENDATION
                return _attach_route_metadata(result, return_metadata, route_info)
            batch_context = ""
            for d in rec_spots_detail:
                chunk_text = _retrieve_spot_knowledge_chunk(d["spot_name"], user_query)
                batch_context += f"""
    【{d['spot_name']}】
    城市:{d['city']}
    等级:{d['level']}
    门票:{d['ticket']}
    开放时间:{d['open_time']}
    地址:{d['address']}
    联系电话:{d['telephone']}
    【知识库介绍】
{chunk_text}
"""
            rec_msg = [
                {"role": "system",
                 "content": f"参考样例景区：{ref_spot}。下面是检索得到的一批风格相似景区真实资料，请严格基于资料类比推荐，禁止编造。\n【最近对话参考（仅衔接上下文，非资料来源）】\n{_history_digest(merged_history_for_llm)}\n{batch_context}" + OUTPUT_FORMAT_RULE},
                {"role": "user", "content": user_query}
            ]
            rec_ans = generate_answer_by_api(rec_msg, llm_cfg=llm_cfg)

        _update_session_subject_by_spots(session_ctx, rec_spots_detail)
        result = _query_result(rec_ans, [], return_metadata, route, "recommendation", spots=rec_spots_detail)
        session_ctx["last_intent_type"] = ROUTE_RECOMMENDATION
        return _attach_route_metadata(result, return_metadata, route_info)

    # 全局兜底分支：路由匹配全部失败
    logger.warning(f"未知路由分支 route={route}，进入兜底RAG")
    q_vec = get_query_embedding(user_query, emb_tokenizer, emb_model)
    vec_res = vector_search_numpy(q_vec, vector_array, meta_data, TOP_K_VECTOR)
    kw_res = keyword_search_db(user_query, TOP_K_KEYWORD)
    merge = vec_res + kw_res
    unique_chunk = []
    seen = set()
    for item in merge:
        ck = item.get("spot_name", "") + "|||" + item.get("chunk_text", "")
        if ck not in seen:
            seen.add(ck)
            unique_chunk.append(item)
    filtered = post_filter_docs(unique_chunk, city_name, allow_cross_city, [])
    rerank_chunk = rerank(user_query, filtered, RERANK_TOP)
    msg = build_chat_messages(user_query, rerank_chunk, merged_history_for_llm, session_ctx)
    answer = generate_answer_by_api(msg, llm_cfg, user_query)
    spots = _aggregate_spot_from_docs(rerank_chunk)
    _update_session_subject_by_spots(session_ctx, spots)
    result = _query_result(answer, rerank_chunk, return_metadata, "fallback", "fallback_rag", spots=spots)
    return _attach_route_metadata(result, return_metadata, route_info)
