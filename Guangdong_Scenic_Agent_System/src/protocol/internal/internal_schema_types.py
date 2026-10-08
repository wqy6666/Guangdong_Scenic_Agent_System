# -*- coding: utf-8 -*-
"""
对内协议工具模块 internal_schema_types.py
职责：
1、正则从LLM返回文本提取 ```json ... ``` 块
2、做轻量JSON格式校验
3、业务层二次校验：仅校验实体白名单，移除置信阈值校验（调试版本）
4、实体合并规则：本地base_entities优先级高于LLM‑Judge输出实体
5、异常兜底：全部异常返回 rag_required 结构
"""
import sys
from pathlib import Path
CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import json
import re
from logger import logger
from route_classifier import load_entities, scenic_alias_normalize, resolve_spot_mention

CURRENT_DIR = Path(__file__).parent

SQL_ALLOW_THRESHOLD_LIST = 95
SQL_ALLOW_THRESHOLD_SPOT_DETAIL = 85
ALLOW_TARGET_FIELDS = {"ticket", "open_time", "address", "telephone", "level"}

SAFE_FALLBACK_RESULT = {
    "intent_type": "rag_required",
    "spot_name": "",
    "spot_name_list": [],
    "city": "",
    "scenic_level": "",
    "target_fields": [],
    "can_use_sql": False,
    "confidence_score": 0,
    "reason": "LLM裁判解析或者校验失败，降级RAG",
    "allow_cross_city": False,
    # 【修复】降级即rag_required，intent_list必须与之一致；为空会让下游跳过全部子意图分支
    "intent_list": ["rag_required"]
}


def extract_json_from_llm_text(raw_llm_text: str) -> str | None:
    pattern = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)
    match = pattern.search(raw_llm_text)
    if match:
        return match.group(1).strip()
    return None


def validate_llm_judge_format(judge_data: dict) -> tuple[bool, str]:
    if not isinstance(judge_data, dict):
        return False, "judge_data 不是字典"

    required_top_keys = [
        "intent_type",
        "spot_name",
        "spot_name_list",
        "city",
        "scenic_level",
        "target_fields",
        "confidence_score",
        "allow_cross_city",
    ]
    for key in required_top_keys:
        if key not in judge_data:
            return False, f"缺少顶层字段：{key}"

    sl = judge_data.get("scenic_level")
    if sl is not None:
        if isinstance(sl, (int, float)):
            sl = f"{int(sl)}A"
            judge_data["scenic_level"] = sl
        judge_data["scenic_level"] = str(judge_data["scenic_level"])

    if not isinstance(judge_data.get("intent_type"), str):
        return False, "intent_type 必须是字符串"
    if not isinstance(judge_data.get("spot_name"), str):
        return False, "spot_name 必须是字符串"
    if not isinstance(judge_data.get("spot_name_list"), list):
        return False, "spot_name_list 必须是数组"
    if not isinstance(judge_data.get("city"), str):
        return False, "city 必须是字符串"
    if not isinstance(judge_data.get("scenic_level"), str):
        return False, "scenic_level 必须是字符串"
    if not isinstance(judge_data.get("target_fields"), list):
        return False, "target_fields 必须是数组"
    if not isinstance(judge_data.get("confidence_score"), (int, float)):
        return False, "confidence_score 必须是数字"
    if not isinstance(judge_data.get("allow_cross_city"), bool):
        return False, "allow_cross_city 必须是布尔值"

    if "intent_list" not in judge_data:
        judge_data["intent_list"] = [judge_data["intent_type"]]

    judge_data["target_fields"] = [f for f in judge_data["target_fields"] if f in ALLOW_TARGET_FIELDS]
    return True, ""


def business_entity_check_single_intent(intent: str, judge_data: dict) -> tuple[bool, str]:
    """
    【调试版】移除置信分数校验！
    仅校验实体是否存在于词典，不再判断confidence_score
    """
    cities, spots_tuple, _ = load_entities()
    spot_name = judge_data.get("spot_name", "")
    city = judge_data.get("city", "")
    spot_name_list = [s[0] for s in spots_tuple]

    if intent == "spot_detail":
        if spot_name and spot_name not in spot_name_list:
            return False, f"实体校验失败，景区[{spot_name}]不在实体词典"
        return True, ""
    elif intent == "city_scenic_list":
        if city and city not in cities:
            return False, f"实体校验失败，城市[{city}]不在城市白名单"
        return True, ""
    return True, ""


def merge_base_entities(judge_result: dict, base_entities: dict) -> dict:
    out = judge_result.copy()
    base_spot = base_entities.get("spot_name")
    base_city = base_entities.get("city")
    base_level = base_entities.get("level")

    # 排除清单先归一化，供回填拦截判断
    exc_raw = out.get("exclude_spot_list", [])
    if isinstance(exc_raw, list):
        norm_exc = []
        for name in exc_raw:
            nm = resolve_spot_mention(str(name))
            if nm:
                norm_exc.append(nm)
        out["exclude_spot_list"] = norm_exc
    else:
        out["exclude_spot_list"] = []
    exclude_names = set(out["exclude_spot_list"])

    if base_spot is not None and base_spot.strip() != "":
        # 【修复】排除类推荐：被用户排除的景区不得经本地回填变成推荐参考主体
        if base_spot in exclude_names:
            logger.info(f"【实体合并拦截】{base_spot}在排除清单中，不回填为spot_name")
        elif out.get("intent_type") == "recommendation":
            # 推荐意图一律不回填参考主体：ref_spot仅用于"像X一样"类比，回填会把排除对象错当参考
            logger.info(f"【实体合并拦截】推荐意图不回填参考主体（本地识别到{base_spot}）")
        elif out.get("spot_name", "") != base_spot:
            logger.warning(f"【实体合并】LLM‑Judge spot_name={out.get('spot_name')}，覆盖使用本地base_spot={base_spot}")
            out["spot_name"] = base_spot

    if base_city is not None and base_city.strip() != "":
        if out.get("city", "") != base_city:
            logger.warning(f"【实体合并】LLM‑Judge city={out.get('city')}，覆盖使用本地base_city={base_city}")
            out["city"] = base_city

    if base_level is not None and base_level.strip() != "":
        if out.get("scenic_level", "") != base_level:
            logger.warning(f"【实体合并】LLM‑Judge scenic_level={out.get('scenic_level')}，覆盖使用本地base_level={base_level}")
            out["scenic_level"] = base_level

    # =====================【修复】对spot_name_list数组里面每一项执行本地实体词典归一化 =====================
    spot_list = out.get("spot_name_list", [])
    if isinstance(spot_list, list):
        new_spot_list = []
        for name in spot_list:
            norm_name, _ = scenic_alias_normalize(name)
            new_spot_list.append(norm_name)
        out["spot_name_list"] = new_spot_list
        logger.info(f"【merge_base_entities】spot_name_list数组归一化完成：原始={spot_list} → 归一化后={new_spot_list}")

    return out


def _fallback_with_base(base_entities: dict | None) -> dict:
    """
    降级兜底时保留本地已回填的实体：
    省略式追问（"周末呢?"）经代词回填得到景区实体，若Judge输出异常走降级，
    不能把已回填的实体一并丢掉，否则下游检索失去锚点，LLM会凭历史乱接主体。
    """
    fb = SAFE_FALLBACK_RESULT.copy()
    if base_entities:
        base_spot = (base_entities.get("spot_name") or "").strip()
        base_city = (base_entities.get("city") or "").strip()
        if base_spot:
            fb["spot_name"] = base_spot
            fb["spot_name_list"] = [base_spot]
        if base_city:
            fb["city"] = base_city
    return fb


def parse_llm_judge_response(raw_llm_output: str, base_entities: dict | None = None, session_alias_map: dict | None = None) -> dict:
    """
    :param raw_llm_output: LLM原始输出文本
    :param base_entities: 本地抽取实体
    :param session_alias_map: 会话临时别名映射，用于别名归一化
    """
    json_str = extract_json_from_llm_text(raw_llm_output)
    if json_str is None:
        logger.warning("LLM裁判：无法提取 ```json``` 代码块")
        return _fallback_with_base(base_entities)

    try:
        judge_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"LLM裁判 json.loads 解析异常: {str(e)}")
        return _fallback_with_base(base_entities)

    format_ok, format_msg = validate_llm_judge_format(judge_dict)
    if not format_ok:
        logger.warning(f"LLM裁判格式校验失败: {format_msg}")
        return _fallback_with_base(base_entities)

    origin_spot_raw = judge_dict.get("spot_name", "")
    norm_spot, spot_is_mapped = scenic_alias_normalize(origin_spot_raw, session_alias_map)
    if norm_spot:
        resolved = resolve_spot_mention(norm_spot)
        if resolved:
            norm_spot, spot_is_mapped = resolved, True
    judge_dict["spot_name"] = norm_spot
    judge_dict["_origin_spot_input"] = origin_spot_raw
    judge_dict["_spot_mapped_flag"] = spot_is_mapped

    # =========【修复】spot_name_list 传入session_alias_map做别名归一化 =========
    origin_list_raw = judge_dict.get("spot_name_list", [])
    norm_list = []
    list_mapped_any = False
    for s in origin_list_raw:
        ns, mapped = scenic_alias_normalize(s, session_alias_map)
        if ns:
            r2 = resolve_spot_mention(ns)
            if r2:
                ns, mapped = r2, True
        norm_list.append(ns)
        if mapped:
            list_mapped_any = True
    judge_dict["spot_name_list"] = norm_list
    judge_dict["_origin_spot_list_input"] = origin_list_raw
    judge_dict["_spot_list_mapped_flag"] = list_mapped_any

    if base_entities is not None:
        judge_dict = merge_base_entities(judge_dict, base_entities)

    raw_intent_list = judge_dict.get("intent_list", [judge_dict["intent_type"]])
    valid_intent_list = []
    for intent in raw_intent_list:
        ok, msg = business_entity_check_single_intent(intent, judge_dict)
        if ok:
            valid_intent_list.append(intent)
            logger.info(f"【子意图过滤】intent={intent} 实体校验通过，保留子意图")
        else:
            logger.warning(f"【子意图过滤】intent={intent} 校验不通过，原因:{msg}，丢弃该子意图")

    if len(valid_intent_list) == 0:
        logger.warning("全部子意图校验失败，降级rag_required")
        return _fallback_with_base(base_entities)

    judge_dict["intent_list"] = valid_intent_list
    judge_dict["intent_type"] = valid_intent_list[0]
    judge_dict["intent_list"] = judge_dict["intent_list"][:3]
    return judge_dict


if __name__ == "__main__":
    print("该文件是库模块，请运行main_agent.py，不要直接执行本脚本！")
