# -*- coding: utf-8 -*-
""" agent_reflection_loop.py 【V3.0 回答核验闭环 + 问句分析与补全Agent】
改动说明：
1.【第二部分】反思循环改造为"回答核验闭环"：
   - 判决对象从"资料摘要"改为"子问题清单 × 草稿回答全文"（不再压缩摘要）；
   - 核验两个维度：覆盖完整（问了的都答了吗）+ 内容可溯源（关键事实在资料里有无出处）；
   - 未通过则按核验器给的动作定向补检索（rag/sql），资料无从补充则如实声明；
   - 最终回答生成后增加"后置核验"，缺哪段补挂对应资料原文/更正提示（纯拼接，零额外LLM）；
2.【第一部分】接入问句分析与补全Agent（plan_sub_questions）：
   - 歧义信号触发（多城市/多等级/复合连接词/无实体澄清类）才调用，普通单问题走快速通道；
   - 子问题 decision=answer/clarify/skip 由Agent决定"回答多少、跳过多少"；
   - Agent输出经词典校验后转成 pre_judge 注入 hybrid_query，跳过其内部Judge避免重复调用；
3. make_info_summary / reflection_judge 退役，由 verify_answer_quality 取代。
"""
import re
import json
import copy
import jieba
from typing import List, Dict, Any, Tuple
from logger import logger
from service_errors import LLMServiceError
from llm_answer import generate_answer_by_api, build_chat_messages, OUTPUT_FORMAT_RULE
from service_errors import LLMOutputError
from route_classifier import (
    classify_query,
    scenic_alias_normalize,
    resolve_spot_mention,
    spot_content_exists,
    load_entities,
    AGENT_PROFILE_ANSWER,
    AGENT_PROFILE_KEYWORDS,
    ROUTE_CLARIFY,
    ROUTE_RAG,
    ROUTE_RECOMMENDATION,
    ROUTE_SCENIC_COUNT,
    ROUTE_SCENIC_LIST,
)
from llm_judge_call import plan_sub_questions

def post_filter_docs(doc_list: list, target_city: str | None, allow_cross_city: bool, spot_white_list: list[str]):
    if not doc_list:
        return doc_list
    same_city = []
    cross_city = []
    spot_match = []
    other = []
    for d in doc_list:
        doc_city = d.get("city")
        doc_spot = d.get("spot_name", "")
        if spot_white_list and doc_spot in spot_white_list:
            spot_match.append(d)
        elif target_city and doc_city == target_city:
            same_city.append(d)
        else:
            cross_city.append(d)
    if allow_cross_city:
        return spot_match + same_city + cross_city
    else:
        return spot_match + same_city

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
    top_spot, top_count = sorted_spots[0]
    if len(sorted_spots) == 1:
        return [top_spot]
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

from rag_hybrid_query import (
    get_query_embedding,
    vector_search_numpy,
    keyword_search_db,
    rerank,
    get_single_spot_detail,
    vector_array,
    meta_data,
    emb_tokenizer,
    emb_model,
    TOP_K_VECTOR,
    TOP_K_KEYWORD,
    RERANK_TOP
)

from memory_utils import detect_lookup_reference, process_retrieve_hit_items, combine_memory_fragments

MAX_AGENT_REFLECT_ITER = 3
REFLECT_ACTION_DONE = "done"
REFLECT_ACTION_RAG = "rag"
REFLECT_ACTION_SQL = "sql"

# =========【第一部分】问句分析与补全Agent：信号触发的开关与探测 =========
PLANNER_ENABLED = True  # 总开关：False则完全回到快速通道老路径
PLANNER_MAX_SUB_QUESTIONS = 4
_PLANNER_COMPOUND_PATTERN = re.compile(r"(再|另外|顺便|然后|还有|以及|分别|同时)")
_PLANNER_INTENT_ROUTE_MAP = {
    "spot_detail": ROUTE_RAG,
    "rag_required": ROUTE_RAG,
    "recommendation": ROUTE_RECOMMENDATION,
    "city_scenic_list": ROUTE_SCENIC_LIST,
    "scenic_count": ROUTE_SCENIC_COUNT,
}


def _detect_multi_city(user_query: str) -> bool:
    """用户输入中出现>=2个不同城市名（含'市'后缀形式）"""
    try:
        cities, _, _ = load_entities()
    except Exception:
        return False
    hit = 0
    for c in cities:
        if c and (c in user_query or (c + "市") in user_query):
            hit += 1
            if hit >= 2:
                return True
    return False


def _detect_multi_spot(user_query: str) -> bool:
    """用户输入中命中>=2个不同词典景区（含别名/简写形式），如"白云山只查门票，丹霞山只介绍地貌" """
    try:
        from route_classifier import _spot_forms
        _, spots_tuple, _ = load_entities()
    except Exception:
        return False
    city_list = [_city for _, _city in spots_tuple]
    hit_names = set()
    for candidate, _city in spots_tuple:
        for form in _spot_forms(candidate, city_list):
            if form and form in user_query:
                hit_names.add(candidate)
                break
        if len(hit_names) >= 2:
            return True
    return False


def _needs_query_planning(user_query: str, route_info: dict, is_lookup_meta: bool) -> bool:
    """
    歧义信号触发制：命中任一信号才调用问句规划Agent。
    回看/复述类元提问不走Agent（由记忆轮+合并链路处理，避免被误判skip）。
    """
    if not PLANNER_ENABLED or is_lookup_meta:
        return False
    entities = route_info.get("entities", {})
    # 信号0：多景区（"白云山只查门票，丹霞山只介绍地貌"）——逐景区约束交规划器生成干净子问题，
    # 避免B方案的公共约束提取把各景区约束混在一起污染子任务
    if _detect_multi_spot(user_query):
        return True
    # 信号1：多等级（"广州4A和5A景区分别有多少个"）
    if len(entities.get("level_list") or []) >= 2:
        return True
    # 信号2：多城市（"广州和深圳分别有哪些5A"）
    if _detect_multi_city(user_query):
        return True
    # 信号3：复合连接词（"告诉我丹霞山门票，再列出广州的5A景区"）
    if _PLANNER_COMPOUND_PATTERN.search(user_query):
        return True
    # 信号4：无实体澄清类（元提问/模糊问句，交给Agent决策答/反问/跳过）
    if route_info.get("route") == ROUTE_CLARIFY:
        return True
    return False


def _normalize_planner_spot(name: str) -> str:
    std, _ = scenic_alias_normalize(name or "")
    return std


def _build_pre_judge(sub: dict) -> Tuple[bool, Dict[str, Any] | None, str]:
    """
    把规划器子问题转换成 hybrid_query 可注入的 judge_res 结构，并做实体词典校验。
    校验失败返回 (False, None, 用户可读的拒绝说明)。
    """
    intent = (sub.get("intent") or "").strip()
    if intent not in _PLANNER_INTENT_ROUTE_MAP:
        intent = "rag_required"  # 未知意图按知识库问答兜底
    spot_raw = (sub.get("spot_name") or "").strip()
    city_raw = (sub.get("city") or "").strip()
    # 【修复】规划器输出的景区名可能是非标准写法（"丹霞山风景名胜区"），
    # 走解析链归一到词典标准名，而不是精确匹配失败就整体拒绝
    # 【新增】对比类支持多景区：spot_name可能被拼接（"A、B"），或spot_name_list携带多个景区
    mention_list = []
    if spot_raw:
        mention_list.extend([n.strip() for n in re.split(r"[、，,和与]", spot_raw) if n.strip()])
    for n in (sub.get("spot_name_list") or []):
        n = str(n).strip()
        if n and n not in mention_list:
            mention_list.append(n)
    resolved_list = []
    for m in mention_list:
        r = resolve_spot_mention(m)
        if r and r not in resolved_list:
            resolved_list.append(r)
    spot_resolved = resolved_list[0] if resolved_list else None
    spot = spot_resolved or _normalize_planner_spot(spot_raw)

    try:
        cities, spots_tuple, _ = load_entities()
    except Exception:
        cities, spots_tuple = (), ()
    spot_names = {s[0] for s in spots_tuple}

    if intent in ("spot_detail", "rag_required") and spot_raw and spot_resolved is None:
        return False, None, f"未在景区数据库中识别到「{spot_raw}」，请确认景区名称后重新提问。"
    if intent in ("city_scenic_list", "scenic_count", "recommendation") and city_raw:
        city_norm = city_raw.removesuffix("市")
        if city_norm not in cities and city_raw not in cities:
            return False, None, f"暂不支持「{city_raw}」的查询，目前支持广东省内地级市。"

    judge_res = {
        "intent_type": intent,
        "intent_list": [intent],
        "spot_name": spot,
        "spot_name_list": (resolved_list if len(resolved_list) >= 2 else ([spot] if spot else [])),
        "city": city_raw,
        "scenic_level": (sub.get("scenic_level") or "").strip(),
        "target_fields": [f for f in (sub.get("target_fields") or [])
                          if f in ("ticket", "open_time", "address", "telephone", "level")],
        # 规划器表达的排除对象经解析链归一后透传，供推荐分支确定性剔除
        "exclude_spot_list": [r for r in (resolve_spot_mention(str(n)) for n in (sub.get("exclude_spot_list") or [])) if r],
        "confidence_score": 90,
        "allow_cross_city": False,
        "can_use_sql": intent in ("spot_detail", "city_scenic_list", "scenic_count"),
        "reason": "query_planner注入",
        "_origin_spot_input": spot_raw,
        "_spot_mapped_flag": spot != spot_raw,
        # 规划器刻意把多个景区放进同一子问题=联合作答类（对比/比较），禁止下游按景区再拆分
        "_comparison": len(resolved_list) >= 2,
    }
    return True, judge_res, ""


def _spot_dict_to_material_text(d: dict) -> str:
    """把工具返回的景区结构化字典格式化为资料原文，供核验器做溯源比对"""
    if d.get("chunk_text"):
        return str(d["chunk_text"])
    lines = []
    if d.get("spot_name"):
        lines.append(f"景区名称：{d['spot_name']}")
    if d.get("city"):
        lines.append(f"城市：{d['city']}")
    if d.get("level"):
        lines.append(f"景区等级：{d['level']}")
    if d.get("ticket"):
        lines.append(f"门票信息：{d['ticket']}")
    if d.get("open_time"):
        lines.append(f"开放时间：{d['open_time']}")
    if d.get("address"):
        lines.append(f"地址：{d['address']}")
    if d.get("telephone"):
        lines.append(f"联系电话：{d['telephone']}")
    return "\n".join(lines)


def _execute_sub_plan(sub_plan: List[dict], chat_history: List[Dict],
                      session_ctx: dict, session_id: str,
                      total_context_token: int, llm_cfg: dict, curr_mem_mgr=None) -> Tuple[str, List[Dict], list, List[str]]:
    """
    执行规划器产出的子问题清单：
    decision=answer → 每个子问题独立走一次 hybrid_query（pre_judge注入，跳过内部Judge）
    decision=clarify/skip → 输出反问/跳过说明
    返回 (合并草稿回答, 汇总docs, 汇总spots, 汇总资料原文)
    资料原文来源：docs的chunk_text + meta.spots结构化数据格式化文本，
    【关键】子问题的落地数据在meta.spots里，必须带回给核验器，否则溯源核验会拿着空资料误判"无出处"。
    """
    from rag_hybrid_query import hybrid_query
    sub_answers: List[str] = []
    sub_docs: List[Dict] = []
    sub_spots: list = []
    sub_materials: List[str] = []
    last_answer_intent = ""
    for idx, sub in enumerate(sub_plan, start=1):
        cq = (sub.get("complete_query") or "").strip()
        decision = (sub.get("decision") or "answer").strip().lower()
        display_q = cq or "该问题"
        # =====【授权链】数据类拒绝（未收录/未能确定所指）不直接采信：先过解析链，再过存在性查询 =====
        # 语境类澄清（候选多个请用户补充）是合法澄清，维持规划器语义，不做升级
        if decision in ("skip", "clarify"):
            spot_mention = (sub.get("spot_name") or "").strip()
            reason = (sub.get("reason") or "").strip()
            data_reject = any(k in reason for k in ("未收录", "没有", "暂无", "收录", "未能确定"))
            upgraded = False
            if spot_mention and data_reject:
                resolved = resolve_spot_mention(spot_mention)
                if resolved:
                    # 升级1：景区可解析 → 规划器误判，转answer执行（落到下方answer路径）
                    logger.info(f"[QueryPlanner] 子问题{idx} 景区[{spot_mention}]可解析为[{resolved}]，数据类拒绝升级为answer")
                    sub["decision"] = "answer"
                    sub["spot_name"] = resolved
                    sub["entity_status"] = f"已收录（标准名：{resolved}）"
                    decision = "answer"
                    upgraded = True
                else:
                    # 升级2：识别不出≠无数据，先跑存在性查询拿工具证据
                    probe = spot_mention if len(spot_mention) >= 2 else (cq or spot_mention)
                    if spot_content_exists(probe):
                        logger.info(f"[QueryPlanner] 子问题{idx} 景区[{spot_mention}]词典未解析但知识库有相关内容，按短语转入知识库问答")
                        u_docs, u_text = run_single_rag_tool(cq or probe, None, "", session_ctx, llm_cfg=llm_cfg)
                        sub_answers.append(u_text)
                        sub_docs.extend(u_docs or [])
                        sub_spots.extend(_aggregate_spot_from_docs(u_docs or []))
                        sub["entity_status"] = "知识库有相关内容（无标准名，已按短语检索）"
                        last_answer_intent = "rag_required"
                        continue
                    logger.info(f"[QueryPlanner] 子问题{idx} 存在性查询为空，拒绝成立（工具证据）")
                    sub["entity_status"] = "未解析（已查询无结果）"
                    sub_answers.append(f"关于「{display_q}」：经查询数据库，暂无收录「{spot_mention}」的相关资料。请您确认景区名称或指定城市。")
                    continue
            if not upgraded:
                if decision == "skip":
                    reason = (sub.get("reason") or "该问题超出景区客服服务范围").strip()
                    sub_answers.append(f"关于「{display_q}」：{reason}。")
                    logger.info(f"[QueryPlanner] 子问题{idx} skip：{reason}")
                    continue
                reason = (sub.get("reason") or "需要您补充更多信息才能准确查询").strip()
                sub_answers.append(f"关于「{display_q}」：{reason}，请您补充说明。")
                logger.info(f"[QueryPlanner] 子问题{idx} clarify：{reason}")
                continue
        ok, pre_judge, reject_msg = _build_pre_judge(sub)
        if not ok:
            sub["entity_status"] = "未解析（解析失败）"
            sub_answers.append(f"关于「{display_q}」：{reject_msg}")
            logger.info(f"[QueryPlanner] 子问题{idx} 实体校验未通过：{reject_msg}")
            continue
        sub["entity_status"] = (f"已收录（标准名：{pre_judge['spot_name']}）"
                                if pre_judge.get("spot_name") else "知识库问答（无固定景区）")
        sub_ctx = copy.deepcopy(session_ctx)
        try:
            logger.info("[QueryPlanner] 执行子问题%d/%d (intent=%s)", idx, len(sub_plan), pre_judge["intent_type"])
            ans, docs, meta = hybrid_query(
                user_query=cq,
                chat_history=chat_history,
                session_ctx=sub_ctx,
                session_id=session_id,
                total_context_token=total_context_token,
                llm_cfg=llm_cfg,
                return_metadata=True,
                is_inner_sub_task=False,
                pre_judge=pre_judge,
                curr_mem_mgr=curr_mem_mgr,
            )
            sub_answers.append(ans)
            sub_docs.extend(docs or [])
            spots = meta.get("spots", []) or []
            sub_spots.extend(spots)
            spot_materials = []
            for sp in spots:
                if isinstance(sp, dict):
                    t = _spot_dict_to_material_text(sp)
                    if t:
                        spot_materials.append(t)
            sub_materials.extend(spot_materials)
            # 纯统计/空清单等确定性SQL结果没有结构化行可格式化：回答文本本身即资料依据
            if not spot_materials and not (docs or []):
                sub_materials.append(f"（系统查询结果）{ans}")
            last_answer_intent = pre_judge["intent_type"]
        except LLMServiceError:
            raise
        except Exception as sub_err:
            logger.error("[QueryPlanner] 子问题执行失败: type=%s", type(sub_err).__name__)
            sub_answers.append(f"关于「{display_q}」：系统暂时获取该信息失败，请稍后重试。")
    # 子任务在会话副本上运行，这里统一回写主会话上下文
    if last_answer_intent:
        session_ctx["last_intent_type"] = _PLANNER_INTENT_ROUTE_MAP.get(last_answer_intent, ROUTE_RAG)
    return "\n\n".join(sub_answers), sub_docs, sub_spots, sub_materials


# =========【第二部分】回答核验闭环 =========
def _verify_truncate(text: str, head: int = 700, tail: int = 300) -> str:
    """核验资料降载：超长片段保留首尾（门票/时间等追加字段位于尾部），中段截断"""
    t = (text or "").strip()
    if len(t) <= head + tail + 30:
        return t
    return t[:head] + "\n…【中段略】…" + t[-tail:]

def _build_numbered_materials(material_parts: List[str]) -> str:
    """把资料原文按片段编号，供核验器标注出处（'片段N'与material_parts下标+1对应）"""
    lines = []
    for idx, text in enumerate(material_parts, start=1):
        lines.append(f"====资料片段{idx}====\n{(text or '').strip()}")
    return "\n".join(lines)


def _material_by_marker(source, material_parts: List[str]) -> str | None:
    """按核验器给出的'片段N'标记取回资料原文；取不到返回None"""
    if not source:
        return None
    m = re.search(r"(\d+)", str(source))
    if not m:
        return None
    idx = int(m.group(1)) - 1
    if 0 <= idx < len(material_parts):
        text = (material_parts[idx] or "").strip()
        return text or None
    return None


def verify_answer_quality(user_query: str, checklist: List[str], draft_answer: str,
                          materials_text: str, session_ctx: dict, llm_cfg: dict) -> Dict[str, Any] | None:
    """
    【第二部分核心】回答核验器：覆盖完整 + 内容可溯源 双维判决。
    返回 {coverage, missing_points:[{point,source}], unsupported_claims:[], action, reason}
    空回答或截断最多重试一次；失败返回 None，调用方明确标记未完成核验。
    """
    checklist_text = "\n".join(f"{i}. {c}" for i, c in enumerate(checklist, start=1))
    pref = session_ctx.get("user_preference") or {}
    pref_line = ""
    if pref:
        pref_line = f"【会话参考】用户累计偏好：{json.dumps(pref, ensure_ascii=False)}\n"
    sys_prompt = (
        "你是景区客服回答核验器，只输出JSON。用【子问题清单】（括号内景区收录状态=数据库事实）对账【回答草稿】，资料原文是唯一事实依据。\n"
        f"【用户提问】{user_query}\n"
        f"【子问题清单】\n{checklist_text}\n"
        f"【回答草稿】\n{draft_answer}\n"
        f"【资料原文（编号片段）】\n{materials_text}\n"
        + pref_line +
        "【规则】\n"
        "1.覆盖（负向声明核对）：清单每个诉求点是否被回应；资料确无、回答如实声明“暂无”=已覆盖；状态“已收录”的景区答“暂无”=缺失（missing_points，action=sql用标准名重查）；“知识库有相关内容”的答“暂无”=缺失（action=rag）；“未解析（已查询无结果）”的答“暂无”=如实覆盖。\n"
        "2.溯源：草稿关键事实（价格/时间/地址/名称/等级/电话）须能在资料原文找到依据，无依据的列入unsupported_claims。\n"
        "3.action：有缺失→sql（门票/时间/地址/等级/电话类）或rag（简介/特色类）；无从补充或仅unsupported（无缺失）→done。\n"
        '【输出】{"coverage":"fully或partially","missing_points":[{"point":"…","source":"片段N或null"}],"unsupported_claims":["…"],"action":"done或rag或sql","reason":"一句话"}'
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "输出核验结果JSON。"},
    ]
    raw = None
    for attempt, output_limit in enumerate((1500, 3000), start=1):
        try:
            raw = generate_answer_by_api(messages, llm_cfg=llm_cfg, timeout=20,
                                         max_tokens=output_limit, purpose="verification")
            if not raw or not raw.strip():
                raise LLMOutputError("核验未返回内容", "empty")
            break
        except LLMOutputError as exc:
            logger.warning("[AnswerVerify] 核验输出不可用: attempt=%d reason=%s", attempt, exc.reason)
            if attempt == 2 or exc.reason not in ("empty", "length"):
                return None
        except Exception as exc:
            # Verification is auxiliary; a provider failure must not discard
            # an existing draft. Do not log provider bodies or credentials.
            logger.warning("[AnswerVerify] 核验调用失败: type=%s", type(exc).__name__)
            return None
    try:
        from protocol.internal.internal_schema_types import extract_json_from_llm_text
        json_str = extract_json_from_llm_text(raw)
        data = None
        if json_str:
            data = json.loads(json_str)
        else:
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                data = json.loads(m.group(0))
        if (not isinstance(data, dict)
                or data.get("coverage") not in ("fully", "partially")
                or not isinstance(data.get("missing_points"), list)
                or not isinstance(data.get("unsupported_claims"), list)
                or data.get("action") not in (REFLECT_ACTION_DONE, REFLECT_ACTION_RAG, REFLECT_ACTION_SQL)):
            logger.warning("[AnswerVerify] 核验输出缺少有效判决字段")
            return None
        coverage = data["coverage"]
        missing = []
        for mp in data.get("missing_points") or []:
            if isinstance(mp, str) and mp.strip():
                missing.append({"point": mp, "source": None})
            elif isinstance(mp, dict) and mp.get("point"):
                missing.append({"point": str(mp["point"]), "source": mp.get("source")})
            else:
                logger.warning("[AnswerVerify] 核验缺失点格式无效")
                return None
        if any(not isinstance(item, str) for item in data["unsupported_claims"]):
            return None
        unsupported = [str(u) for u in (data.get("unsupported_claims") or []) if u]
        if missing:
            coverage = "partially"
        action = data["action"]
        return {
            "coverage": coverage,
            "missing_points": missing,
            "unsupported_claims": unsupported,
            "action": action,
            "reason": str(data.get("reason", "")),
        }
    except Exception as e:
        logger.warning("[AnswerVerify] 核验JSON解析失败: type=%s", type(e).__name__)
        return None



def run_single_rag_tool(user_query: str, spot_name: str, city_name: str, session_ctx: Dict[str, Any], llm_cfg: dict) -> Tuple[List[Dict], str]:
    q_vec = get_query_embedding(user_query, emb_tokenizer, emb_model)
    vec_res = vector_search_numpy(q_vec, vector_array, meta_data, TOP_K_VECTOR)
    kw_res = keyword_search_db(user_query, TOP_K_KEYWORD)
    merge = vec_res + kw_res
    unique_chunk = []
    seen = set()
    for item in merge:
        ck_key = item.get("spot_name", "") + "|||" + item.get("chunk_text", "")
        if ck_key not in seen:
            seen.add(ck_key)
            unique_chunk.append(item)
    filtered_docs = post_filter_docs(
        unique_chunk,
        target_city=city_name,
        allow_cross_city=session_ctx.get("allow_cross_city", False),
        spot_white_list=[spot_name] if spot_name else []
    )
    rerank_docs = rerank(user_query, filtered_docs, RERANK_TOP)
    fake_history = []
    msg = build_chat_messages(user_query, rerank_docs, fake_history, session_ctx)
    rag_part_text = generate_answer_by_api(msg, llm_cfg=llm_cfg, user_query=user_query)
    return rerank_docs, rag_part_text

def run_single_sql_tool(spot_name: str) -> str:
    spot_detail = get_single_spot_detail(spot_name)
    detail_lines = {
        "ticket": f"门票信息：{spot_detail['ticket']}",
        "open_time": f"开放时间：{spot_detail['open_time']}",
        "address": f"地址：{spot_detail['address']}",
        "level": f"景区等级：{spot_detail['level']}",
        "telephone": f"联系电话：{spot_detail['telephone']}",
    }
    sql_text = f"【{spot_detail['spot_name']}】\n" + "\n".join(detail_lines.values())
    return sql_text


def agent_reflective_query(
    user_query: str,
    chat_history: List[Dict],
    session_ctx: Dict[str, Any],
    session_id: str,
    total_context_token: int,
    llm_cfg: dict,
    return_metadata: bool = False,
    is_inner_sub_task: bool = False,
    curr_mem_mgr=None,
):
    """llm_cfg 透传用户独立密钥；V3.0：问句规划 + 回答核验闭环"""
    if is_inner_sub_task:
        from rag_hybrid_query import hybrid_query
        return hybrid_query(
            user_query=user_query,
            chat_history=chat_history,
            session_ctx=session_ctx,
            session_id=session_id,
            total_context_token=total_context_token,
            llm_cfg=llm_cfg,
            return_metadata=return_metadata,
            is_inner_sub_task=True,
            pre_judge=None,
            curr_mem_mgr=curr_mem_mgr
        )

    iter_count = 0
    collected_answer_parts: List[str] = []
    collected_material_parts: List[str] = []
    collected_docs: List[Dict] = []
    seen_content_set = set()

    route_info = classify_query(user_query, session_ctx)
    route = route_info["route"]
    entities = route_info["entities"]
    city_name = entities.get("city", "")
    spot_name = entities.get("spot_name", "")
    logger.info(f"[AgentReflect] 初始路由route={route}, spot={spot_name}, city={city_name}")
    is_lookup_meta = detect_lookup_reference(user_query)
    # 混合提问（"你能做什么？顺便告诉我广州有哪些5A景区"）：业务回答前拼接身份介绍
    wants_intro = any(kw in user_query for kw in AGENT_PROFILE_KEYWORDS) and route not in ("agent_profile", "greeting")

    # =====【授权链】疑似未收录景区：识别层不定"无数据"，先跑存在性查询拿工具证据 =====
    unknown_spot = entities.get("unknown_spot")
    unknown_handled = False
    draft_answer: str | None = None
    draft_docs: List[Dict] = []
    draft_meta: Dict[str, Any] = {}
    draft_spots: list = []
    sub_materials: List[str] = []
    sub_plan = None
    if unknown_spot and route == ROUTE_CLARIFY:
        exists = spot_content_exists(unknown_spot)
        if not exists:
            # 工具查无 → 数据层结论成立（附查询证据），明示边界
            logger.info(f"[UnknownSpot] 「{unknown_spot}」存在性查询为空：数据库无收录（工具证据）")
            hint = f"经查询数据库，暂无收录「{unknown_spot}」的相关资料，无法回答该景区的问题。"
            if entities.get("city"):
                hint += f"如果您想了解{entities['city']}已收录的A级景区，我可以为您列出或推荐。"
            else:
                hint += "您可以换一个已收录的景区名称咨询，或告诉我城市，我为您列出该市的A级景区。"
            result = _query_result(hint, [], return_metadata, "clarify", "clarification")
            return _attach_route_metadata(result, return_metadata, route_info)
        # 库内有相关内容 → 转入知识库问答（不因识别不出而终止）
        logger.info(f"[UnknownSpot] 「{unknown_spot}」存在性查询命中，转入知识库问答")
        u_docs, u_text = run_single_rag_tool(user_query, None, entities.get("city") or "", session_ctx, llm_cfg=llm_cfg)
        draft_answer, draft_docs = u_text, u_docs
        draft_meta = {"route": "rag", "answer_type": "rag_answer", "spots": _aggregate_spot_from_docs(u_docs)}
        draft_spots = draft_meta["spots"]
        unknown_handled = True

    # =====【第一部分】问句分析与补全：歧义信号触发 =====
    if not unknown_handled and _needs_query_planning(user_query, route_info, is_lookup_meta):
        sub_plan = plan_sub_questions(user_query, session_ctx, chat_history, route_info, llm_cfg)
        if sub_plan:
            logger.info(f"[QueryPlanner] 规划到{len(sub_plan)}个子问题：{[s.get('complete_query') for s in sub_plan]}")

    if sub_plan:
        draft_answer, draft_docs, draft_spots, sub_materials = _execute_sub_plan(
            sub_plan, chat_history, session_ctx, session_id, total_context_token, llm_cfg, curr_mem_mgr
        )
        _update_session_subject_by_spots(session_ctx, draft_spots)
        has_answer_sub = any((s.get("decision") or "answer") == "answer" for s in sub_plan)
        if has_answer_sub and len(sub_plan) > 1:
            draft_route = "planner_multi"
        elif has_answer_sub:
            draft_route = route
        else:
            draft_route = "clarify"
        draft_meta = {"route": draft_route, "answer_type": "agent_reflective_answer", "spots": draft_spots}
    elif not unknown_handled:
        from rag_hybrid_query import hybrid_query
        draft_answer, draft_docs, draft_meta = hybrid_query(
            user_query=user_query,
            chat_history=chat_history,
            session_ctx=session_ctx,
            session_id=session_id,
            total_context_token=total_context_token,
            llm_cfg=llm_cfg,
            return_metadata=True,
            is_inner_sub_task=False,
            pre_judge=None,
            curr_mem_mgr=curr_mem_mgr
        )
        draft_spots = draft_meta.get("spots", [])

    # =====【筛选语义分析/空结果回流】带筛选假设或空结果 → 完整原话交LLM语义分析（一次为限） =====
    # 关键词层对条件语义（"平日免费节假日收费"等）存在天然盲区：只有LLM看到完整原话才能正确理解
    _has_filter_assumption = entities.get("ticket_free") or entities.get("budget_max") is not None
    _is_empty_result = draft_answer and "没有查询到" in draft_answer
    if (draft_meta.get("route", "") in ("scenic_list", "scenic_count")
            and (_has_filter_assumption or _is_empty_result)):
        logger.info("[EmptyRetry] 触发LLM语义分析（一次为限）")
        retry_plan = plan_sub_questions(user_query, session_ctx, chat_history, route_info, llm_cfg)
        if retry_plan:
            retry_answer, retry_docs, retry_spots, _retry_materials = _execute_sub_plan(
                retry_plan, chat_history, session_ctx, session_id, total_context_token, llm_cfg, curr_mem_mgr)
            retry_clean = (retry_answer or "").strip()
            if retry_clean and "没有查询到" not in retry_clean:
                draft_answer, draft_docs = retry_answer, retry_docs
                draft_spots = retry_spots
                draft_meta = {"route": "planner_retry", "answer_type": "agent_reflective_answer", "spots": retry_spots}
                logger.info("[EmptyRetry] 工具重查命中，空答案已替换，进入核验闭环")
            else:
                logger.info("[EmptyRetry] 重查仍无结果，保留原空答案")

    # ===== 固定答案路由直接透传：问候/身份/清单/数量不进核验；澄清默认透传，回看/复述类除外 =====
    first_route = draft_meta.get("route", "")
    if first_route in ("greeting", "agent_profile", "scenic_count", "scenic_list") or (
        first_route == "clarify" and not is_lookup_meta
    ):
        logger.info(f"[AgentReflect] 首轮路由={first_route}为固定答案，直接透传，不进入核验迭代")
        passthrough_ans = draft_answer
        if wants_intro:
            passthrough_ans = AGENT_PROFILE_ANSWER + "\n\n" + draft_answer
        result = _query_result(
            passthrough_ans,
            draft_docs,
            return_metadata,
            route=first_route,
            answer_type=draft_meta.get("answer_type", "text"),
            spots=draft_spots,
        )
        return _attach_route_metadata(result, return_metadata, route_info)

    # ===== 收集阶段：回答全文与资料原文分轨保留（不再压缩摘要） =====
    def _collect(ans_text: str, docs):
        stripped = (ans_text or "").strip()
        if stripped and stripped not in seen_content_set:
            seen_content_set.add(stripped)
            collected_answer_parts.append(ans_text)
        for d in docs or []:
            collected_docs.append(d)
            ck = (d.get("chunk_text") or "").strip()
            if ck:
                collected_material_parts.append(ck)

    _collect(draft_answer, draft_docs)
    # 【关键】把 meta.spots 里的结构化落地数据（景区详情/清单行）统一注入核验资料：
    # 规划器子任务与直连 hybrid_query 的 spot_detail/清单意图都把落地数据放在 meta.spots 而非 docs，
    # 不注入的话溯源核验会拿着空资料把真实有据的回答误判为"无出处"，进而触发无谓的合并重写
    for sp in draft_spots:
        if isinstance(sp, dict):
            t = _spot_dict_to_material_text(sp)
            if t and t not in collected_material_parts:
                collected_material_parts.append(t)
    if sub_plan:
        for t in sub_materials:
            if t not in collected_material_parts:
                collected_material_parts.append(t)
    # 兜底：完全没有任何资料时（纯统计类结果），以确定性查询结果文本自证，避免核验全量误报
    if not collected_material_parts and (draft_answer or "").strip():
        collected_material_parts.append(f"（系统查询结果）{draft_answer.strip()}")

    # =====【第二部分】回答核验闭环：子问题清单 × 草稿回答 逐条核验 =====
    checklist: List[str] = []
    if sub_plan:
        for s in sub_plan:
            # 全部子问题入清单（含被拒绝的）：拒绝声明的合理性同样要交核验层对账
            cq_s = (s.get("complete_query") or "").strip()
            status = s.get("entity_status")
            # 收录状态是数据库侧事实，随清单下发供核验器做负向声明对账
            checklist.append(f"{cq_s}（景区收录状态：{status}）" if status and cq_s else cq_s)
    if not sub_plan and draft_spots:
        # 非规划器路径同样下发收录状态：草稿涉及已收录景区时，核验器才能识别"资料有却答暂无"
        spot_names = [s.get('spot_name') for s in draft_spots if isinstance(s, dict) and s.get('spot_name')]
        if spot_names:
            checklist = [f"{user_query}（景区收录状态：已收录（标准名：{'、'.join(spot_names)}））"]
    checklist = [c for c in checklist if c] or [user_query]

    verify_res = None
    verification_unavailable = False
    rewrite_unavailable = False
    supplemented_spots = set()
    materials_text = ""
    material_parts: List[str] = []
    while iter_count < MAX_AGENT_REFLECT_ITER:
        iter_count += 1
        draft_text = "\n\n".join(collected_answer_parts)
        materials_text = _build_numbered_materials([_verify_truncate(x) for x in collected_material_parts])
        material_parts = list(collected_material_parts)
        verify_res = verify_answer_quality(user_query, checklist, draft_text, materials_text, session_ctx, llm_cfg)
        if verify_res is None:
            verification_unavailable = True
            logger.warning("[AnswerVerify] 核验器不可用，保留现有回答并标记未完成核验")
            break
        # 【提速】没有缺失点时补充检索无的放矢：unsupported由更正机制处理
        if not verify_res["missing_points"]:
            verify_res["action"] = REFLECT_ACTION_DONE
        if verify_res["coverage"] == "fully" and not verify_res["unsupported_claims"]:
            logger.info(f"[AnswerVerify] 第{iter_count}轮核验通过：{verify_res['reason']}")
            break
        logger.info(
            f"[AnswerVerify] 第{iter_count}轮核验未完全通过 coverage={verify_res['coverage']} "
            f"missing={verify_res['missing_points']} unsupported={verify_res['unsupported_claims']} action={verify_res['action']}"
        )
        action = verify_res["action"]
        supplemented = False
        targets = []
        if action in (REFLECT_ACTION_RAG, REFLECT_ACTION_SQL):
            # 【提速】一轮补齐所有可解析缺失目标：避免每个目标各花一轮核验（每轮数分钟）
            for mp in verify_res["missing_points"] or []:
                r = resolve_spot_mention(str(mp.get("point", "")))
                if r and r not in supplemented_spots and r not in targets:
                    targets.append(r)
            if not targets and spot_name and spot_name not in supplemented_spots:
                targets.append(spot_name)
            if not targets:
                logger.info("[AnswerVerify] 缺失点涉及景区均已补充过，无新目标，结束迭代")
                break
            logger.info(f"[AnswerVerify] 本轮批量补充目标：{targets}")
        if action == REFLECT_ACTION_RAG and targets:
            for target_spot in targets:
                new_docs, new_rag_text = run_single_rag_tool(user_query, target_spot, city_name, session_ctx, llm_cfg=llm_cfg)
                exist_chunks = {d.get("chunk_text") for d in collected_docs}
                new_valid = [d for d in new_docs if d.get("chunk_text") not in exist_chunks]
                if new_valid:
                    _collect(new_rag_text, new_valid)
                    supplemented = True
                    supplemented_spots.add(target_spot)
        elif action == REFLECT_ACTION_SQL and targets:
            for target_spot in targets:
                new_sql_text = run_single_sql_tool(target_spot)
                if new_sql_text.strip() not in seen_content_set:
                    collected_answer_parts.append(new_sql_text)
                    seen_content_set.add(new_sql_text.strip())
                    collected_material_parts.append(new_sql_text)
                    supplemented = True
                    supplemented_spots.add(target_spot)
        if not supplemented:
            logger.info("[AnswerVerify] 无可执行补充动作或无新数据，结束迭代")
            break

    # ===== 记忆轮 =====
    memory_round_content = ""
    try:
        if curr_mem_mgr is not None:
            need_retrieve = detect_lookup_reference(user_query)
            history_fragments = []
            if need_retrieve:
                q_words = jieba.lcut(user_query)
                q_kw = list({w.strip() for w in q_words if len(w.strip())>=2})[:20]
                hit_index = curr_mem_mgr.search_short2_index(q_kw)
                history_fragments = process_retrieve_hit_items(hit_index, curr_mem_mgr)
            long_mem_data = curr_mem_mgr.get_long_memory()
            memory_round_content = combine_memory_fragments(history_fragments, long_mem_data)
    except Exception as e:
        logger.warning(f"[MemoryModule]记忆模块读取异常，跳过记忆轮，error:{str(e)}", exc_info=True)
        memory_round_content = ""

    # ===== 生成最终回答（一次通过且无需记忆注入时直接采用草稿，省一次重写） =====
    one_pass = (verify_res is not None
                and verify_res["coverage"] == "fully"
                and not verify_res["unsupported_claims"]
                and len(collected_answer_parts) == 1
                and not memory_round_content.strip())
    # 【提速】无据表述的清理（更正行/后置核验共用）
    def _clean_uc(uc):
        s = str(uc).strip()
        for ch in (chr(34), chr(0x201C), chr(0x201D)):
            s = s.replace(ch, "")
        s = s.strip()
        if len(s) > 50:
            s = s[:50] + "…"
        return s

    # 【提速】覆盖完整但有无据细节：不做全量重写，仅附更正行（省一次巨型重写）
    corrections_only = (verify_res is not None
                        and verify_res["coverage"] == "fully"
                        and not verify_res["missing_points"]
                        and bool(verify_res["unsupported_claims"]))
    if verification_unavailable:
        final_answer = "\n\n".join(collected_answer_parts)
    elif one_pass:
        final_answer = collected_answer_parts[0]
        logger.info("[AnswerVerify] 草稿一次通过核验，直接采用，跳过最终重写")
    elif corrections_only:
        final_answer = (collected_answer_parts[0] if len(collected_answer_parts) == 1
                        else ("\n\n").join(collected_answer_parts))
        fix_lines = []
        for uc in verify_res["unsupported_claims"]:
            uc_clean = _clean_uc(uc)
            if uc_clean:
                fix_lines.append(f"【更正】回答中关于“{uc_clean}”的表述未在资料原文中找到依据，请以景区官方信息为准。")
        if fix_lines:
            final_answer = final_answer + ("\n\n").join([""] + fix_lines)
        logger.info(f"[AnswerVerify] 覆盖完整但有无据表述{len(fix_lines)}处，跳过全量重写，仅附更正")
    else:
        merge_sys_prompt = (
            "你是广东省A级景区客服助手，请严格遵守下面全部规则生成回答。\n"
            "【硬性输出校验规则】\n"
            "1. 严格只使用下面提供的工具资料、会话历史参考片段生成回答，禁止编造任何不在资料内的景区、门票、开放时间等信息；\n"
            "2. 逐条核对用户原始提问全部诉求，每一个问题点都要回应；资料没有该信息如实回复“暂无相关信息”；\n"
            "资料片段中已明确给出的字段值（门票/开放时间/地址/等级/电话），必须如实引用，禁止改写为“暂无相关信息”；\n"
            "3. 禁止多答：不要输出用户没有询问的无关背景；**但是对于用户明确询问的游玩、看点等诉求，请保留资料中的有效景点、游玩细节，不要刻意压缩删减；**\n"
            "4. 不要直接堆砌原始资料碎片，提炼整理为通顺分层的中文。\n" + OUTPUT_FORMAT_RULE + "\n"
            f"用户原始提问：{user_query}\n【多轮Agent工具收集的全部资料】\n"
        )
        for idx, part in enumerate(collected_answer_parts, start=1):
            merge_sys_prompt += f"\n====资料片段{idx}====\n{part}\n"
        if memory_round_content.strip():
            merge_sys_prompt += "\n====会话记忆轮====\n"
            merge_sys_prompt += memory_round_content
            merge_sys_prompt += "\n"
        session_pref = session_ctx.get("user_preference") or {}
        if session_pref:
            merge_sys_prompt += "\n====会话累计筛选偏好（用户本会话中明确提出过的要求）====\n"
            merge_sys_prompt += json.dumps(session_pref, ensure_ascii=False)
            merge_sys_prompt += "\n"
        if verify_res:
            if verify_res["missing_points"]:
                pts = "；".join(mp["point"] for mp in verify_res["missing_points"])
                merge_sys_prompt += f"\n【核验提示】上一版回答遗漏了以下用户诉求点：{pts}。若下面资料中有相关信息必须补齐；资料确无则如实声明暂无。\n"
            if verify_res["unsupported_claims"]:
                ucs = "；".join(verify_res["unsupported_claims"])
                merge_sys_prompt += (f"\n【核验提示】以下表述在资料原文中无出处：{ucs}。\n"
                                     "处理要求：只删除或修正这些无出处的细节表述；有出处的景区名称与关键事实（如门票免费、门票价格、景区名单）"
                                     "必须保留并正常作答，禁止因个别细节无据而把整个景区或整段推荐回复为“暂无相关信息”。\n")

        final_messages = [
            {"role": "system", "content": merge_sys_prompt},
            *chat_history,
            {"role": "user", "content": user_query}
        ]
        # A rewrite must not spend its entire output budget on reasoning and
        # discard an otherwise usable draft. Retry output exhaustion only once.
        for output_limit in (2000, 4000):
            try:
                final_answer = generate_answer_by_api(
                    final_messages, llm_cfg=llm_cfg, timeout=25,
                    max_tokens=output_limit, purpose="rewrite",
                )
                if not final_answer.strip():
                    raise LLMOutputError("回答整理未返回内容", "empty")
                break
            except LLMOutputError as exc:
                if exc.reason not in ("empty", "length"):
                    raise
                logger.warning("[AnswerRewrite] 输出不可用 reason=%s max_tokens=%d", exc.reason, output_limit)
                if output_limit == 4000:
                    final_answer = "\n\n".join(collected_answer_parts)
                    if not final_answer.strip():
                        raise
                    rewrite_unavailable = True
                    verification_unavailable = True
                    logger.warning("[AnswerRewrite] 重试后仍不可用，保留已有回答并附核验说明")

        # ===== 后置核验：最终回答输出前的最后一道闸（确定性修补，零额外生成） =====
        # When keeping the draft, apply the verdict already obtained for it.
        # Do not silently drop known unsupported claims or trigger more LLM calls.
        post_res = (verify_res if rewrite_unavailable else
                    verify_answer_quality(user_query, checklist, final_answer, materials_text, session_ctx, llm_cfg))
        if post_res is None:
            verification_unavailable = True
        if post_res and (post_res["missing_points"] or post_res["unsupported_claims"]):
            fix_lines = []
            for mp in post_res["missing_points"]:
                point = mp.get("point", "")
                part_text = _material_by_marker(mp.get("source"), material_parts)
                if part_text:
                    fix_lines.append(f"【补充】{point}：\n{part_text}")
                else:
                    fix_lines.append(f"【补充】关于{point}：现有资料中暂未检索到，建议以景区官方渠道信息为准。")
            for uc in post_res["unsupported_claims"]:
                # 清理核验器抄录的引号/长文，避免更正行拼接出乱引文
                uc_clean = str(uc).strip()
                for ch in ('“', '”', '"', '「', '」', '『', '』', '\'', '‘', '’'):
                    uc_clean = uc_clean.replace(ch, '')
                uc_clean = uc_clean.strip()
                if len(uc_clean) > 50:
                    uc_clean = uc_clean[:50] + "…"
                if uc_clean:
                    fix_lines.append(f"【更正】回答中关于“{uc_clean}”的表述未在资料原文中找到依据，请以景区官方信息为准。")
            if fix_lines:
                logger.warning(f"[AnswerVerify] 后置核验发现{len(fix_lines)}处问题，追加补充说明")
                final_answer = final_answer + "\n\n" + "\n\n".join(fix_lines)

    if verification_unavailable:
        notice = ("提示：本次回答整理未完成，已保留现有回答并附上核验说明，请结合参考资料核实关键信息。"
                  if rewrite_unavailable else
                  "提示：本次回答未完成自动核验，请结合参考资料核实关键信息。")
        final_answer = final_answer.rstrip() + "\n\n" + notice
    if wants_intro:
        final_answer = AGENT_PROFILE_ANSWER + "\n\n" + final_answer
    # 主体回写【修复】：规划器路径以各子问题涉及景区为准（draft_spots含全部主体）；
    # 不再用collected_docs的频率聚合覆盖，避免多主体退化成单一主体，导致后续"它"默认第一个景区
    if sub_plan:
        _update_session_subject_by_spots(session_ctx, draft_spots)
    else:
        _update_session_subject_by_spots(session_ctx, collected_docs)
    result = _query_result(
        final_answer,
        collected_docs,
        return_metadata,
        route=route,
        answer_type="agent_unverified_answer" if verification_unavailable else "agent_reflective_answer",
        spots=draft_spots
    )
    result = _attach_route_metadata(result, return_metadata, route_info)
    return result
