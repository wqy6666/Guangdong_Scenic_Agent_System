# -*- coding: utf-8 -*-
""" 结构化查询接口：专门处理景区星级、城市清单类查询 """
import sqlite3
import pandas as pd
import json
import logging
from config import DB_PATH, DEEPSEEK_API_KEY
from llm_answer import generate_answer_by_api
from llm_answer import build_chat_messages, generate_answer_by_api

logger = logging.getLogger(__name__)


def get_scenic_by_city_level(city_name: str, level: str = None):
    """
    根据城市+星级查询景区清单（自动去重）
    :param city_name:城市名
    :param level: 4A / 5A，不传查全部
    :return:总数，景区名称列表
    """
    conn = sqlite3.connect(DB_PATH)
    if level:
        # 修复：数据表字段为 level，不是 star_level
        sql = """
        SELECT DISTINCT spot_name FROM scenic_spots
        WHERE city = ? AND level = ?
        """
        df = pd.read_sql(sql, conn, params=[city_name, level])
    else:
        sql = """
        SELECT DISTINCT spot_name FROM scenic_spots
        WHERE city = ?
        """
        df = pd.read_sql(sql, conn, params=[city_name])
    conn.close()
    name_list = df["spot_name"].tolist()
    total = len(name_list)
    return total, name_list


def get_scenic_detail_by_city(city_name: str):
    """获取一个城市全部景区完整结构化信息，供给RAG生成答案"""
    conn = sqlite3.connect(DB_PATH)
    sql = """
    SELECT spot_name,location_val,ticket_val,open_time_val,intro_val,city
    FROM scenic_cleaned WHERE city = ?
    """
    df = pd.read_sql(sql, conn, params=[city_name])
    conn.close()
    detail_list = df.to_dict("records")
    return detail_list


def get_single_spot_detail(spot_name: str):
    """
    查询单个景区门票、开放时间、地址详情
    :param spot_name:景区完整名称
    :return:字典，空字段自动填充兜底文案
    """
    conn = sqlite3.connect(DB_PATH)
    sql = """
    SELECT spot_name,location_val,ticket_val,open_time_val
    FROM scenic_cleaned WHERE spot_name = ?
    """
    df = pd.read_sql(sql, conn, params=[spot_name])
    conn.close()
    if df.empty:
        return None
    row = df.iloc[0]
    detail_info = {
        "spot_name": row["spot_name"],
        "location": row["location_val"] if row["location_val"] else "暂无地址信息",
        "ticket": row["ticket_val"] if row["ticket_val"] else "暂无门票相关信息",
        "open_time": row["open_time_val"] if row["open_time_val"] else "暂无开放时间信息"
    }
    return detail_info


def rule_judge_sql_permission(user_query: str):
    """
    第一层本地边界裁判
    返回标记：
        fast_sql:高置信纯名单，直接放行高速SQL
        need_llm_check:命中模糊关键词，交给LLM二次裁决
        forbid_sql:命中加工黑名单，禁止SQL，直接RAG
    """
    high_conf_trigger = {"一共有", "多少", "列出", "名单"}
    fuzzy_trigger = {"哪些", "有哪些"}
    process_black_words = {"推荐", "预算", "攻略", "游玩", "路线", "挑选",
                           "选一个", "对比", "亲子", "打卡", "规划", "去哪", "去什么"}

    hit_high = any(word in user_query for word in high_conf_trigger)
    hit_fuzzy = any(word in user_query for word in fuzzy_trigger)
    hit_black = any(word in user_query for word in process_black_words)

    if hit_black:
        return "forbid_sql", "本地规则判定：含有推荐加工词，进入双召回RAG链路"
    if hit_high:
        return "fast_sql", "本地规则判定：高置信纯名单查询，放行高速SQL"
    if hit_fuzzy:
        return "need_llm_check", "本地规则判定：命中模糊清单词，交由LLM意图裁判"

    return "forbid_sql", "本地规则判定：未命中清单关键词，进入双召回RAG链路"


def parse_intent_by_llm(user_query: str, history: list):
    messages = build_chat_messages(user_query, [], history)
    #关闭流式，意图解析不需要打印到控制台
    resp = generate_answer_by_api(messages, stream=False)
    import json
    try:
        intent_info = json.loads(resp)
    except Exception as e:
        logger.warning(f"意图JSON解析失败，使用兜底参数:{e}")
        intent_info = {"query_type": "general", "city": None, "max_price": None, "spot_level": None,
                       "spot_name": None}

    # ==========省份‑城市字段协议截断校验============
    PROVINCE_BLACK_LIST = {"广东", "广东省"}
    city_raw = intent_info.get("city", "")
    if city_raw in PROVINCE_BLACK_LIST:
        logger.warning(f"检测到省份关键词【{city_raw}】，不属于城市字段，参数清空")
        intent_info["city"] = None
    #===============================================
    return intent_info
