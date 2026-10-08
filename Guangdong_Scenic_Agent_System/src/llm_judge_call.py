# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional
import json
import re
from logger import logger
from service_errors import LLMServiceError
from llm_answer import generate_answer_by_api
from protocol.internal.internal_schema_types import parse_llm_judge_response


def build_judge_system_prompt(session_ctx: Dict[str, Any]) -> str:
    import json
    alias_map = session_ctx.get("session_alias_map", {})
    alias_text_lines = []
    for alias, std in alias_map.items():
        alias_text_lines.append(f"{alias} → {std}")
    alias_block = ""
    if alias_text_lines:
        alias_block = "【本次对话别名映射】\n" + "\n".join(alias_text_lines) + "\n本次对话中出现左侧简称，请等价理解为右侧标准景区全名。\n"

    session_pref = session_ctx.get("user_preference", {})
    session_pref_block = f"【当前会话临时用户偏好（会话内累积，不需要用户说记住）】\n{json.dumps(session_pref, ensure_ascii=False, indent=2)}\n"
    pref_block = ""
    long_memory_list = session_ctx.get("longterm_memory_list", [])
    if long_memory_list:
        pref_block = f"【用户持久长期记忆（用户明确说记住我才保存）】\n{json.dumps(long_memory_list, ensure_ascii=False, indent=2)}\n"

    sys_prompt = f"""
你是景区Agent意图裁判。输出严格使用```json ```格式。
{alias_block}
{session_pref_block}
{pref_block}
# 重要业务规则
## 结构化数据库仅支持5个字段：
ticket(门票)、open_time(开放时间)、address(地址)、level(等级)、telephone(电话)
> 用户提问会使用大量衍生同义词，需要你做语义映射：
1. 票价、多少钱、收费 → ticket（门票）
2. 几点开门、几点关门、营业时间 → open_time（开放时间）
3. 在哪、在哪里、位置 → address（地址）
4. 星级、几A、什么级别 → level（等级）
5. 联系方式 → telephone（电话）
## 【重要新增规则：存在会话偏好时的意图判定】
如果【当前会话临时用户偏好】不为空（存在用户的游玩偏好，例如亲子、预算等）：
- 用户仅仅输入一个城市名字，**优先输出recommendation推荐意图**；
- 只有用户明确出现「列出、名单、一共有、多少个」这类清单/统计关键词，才允许输出city_scenic_list / scenic_count。
## 【多景区规则】
如果用户一句话同时询问**多个不同景区**，必须把全部景区名称写入`spot_name_list`数组；
spot_name字段填写主景区；`spot_name_list`输出全部被问到的景区名称列表。
最多识别4个景区，超过4个只输出前4个。
## 意图规则
1. spot_detail：查询景区结构化字段（门票/开放时间/地址/等级/电话）；
    - spot_detail意图target_fields**必须输出语义匹配到的合法字段集合，只能从[ticket,open_time,address,telephone,level]选取**；
2. rag_required：单点景区知识库问答（简介、特色），**该意图必须有spot_name，没有景区实体禁止输出rag_required**；
3. recommendation【推荐意图】：
    ✅recommendation必须提供有效的地级市city；
    A)城市偏好推荐：用户给出地级市+偏好（亲子、预算、景色、交通）→输出recommendation，city填写目标地级市；
    B)类比相似推荐：用户说“像XX景区一样”，寻找同类风格景区 →输出recommendation，spot_name填写参考样例景区；
    C)排除类推荐：用户说“不要X/不去X/排除X”时，X是排除对象而**不是推荐参考**：X必须写入exclude_spot_list数组，
      spot_name必须留空字符串，按city推荐其余景区即可；绝对禁止把被排除的景区填入spot_name当作类比参考；
    ⚠️recommendation禁止编造景区，后续程序会从数据库召回真实景区做筛选。
4. city_scenic_list / scenic_count：城市清单、数量统计。
> city_scenic_list / scenic_count 仅在用户明确要求列出名单、统计数量的时候使用。
## 强制约束
1. 如果用户问题是下面这类【非结构化内容】，优先使用rag_required或者recommendation，**禁止输出spot_detail**：
    - 停车场、停车费、停车位、观光车、接驳车、卫生间、充电桩；
    - 景区攻略、游玩路线、行程规划、特色介绍、历史文化、风景；
    - 美食、周边游玩、亲子游玩、预算推荐等；
2. 多意图规则：
    - 用户同时问【结构化字段 + 非结构化介绍】，输出intent_list=["spot_detail","rag_required"]；
    - 用户做城市筛选推荐输出intent_list=["recommendation"]；
3. intent_type可选值：spot_detail / city_scenic_list / scenic_count / recommendation / rag_required
4. scenic_level字段只能输出字符串："5A" / "4A" / "3A" / ""，绝对不能输出数字；
5. spot_name、city、scenic_level全部输出字符串类型，禁止输出数字、null。
输出字段：intent_type, intent_list, spot_name, spot_name_list, city, scenic_level, target_fields, exclude_spot_list, confidence_score, allow_cross_city
"""
    return sys_prompt


def call_llm_route_judge(user_query: str, session_ctx: Dict[str, Any], llm_cfg: dict, base_entities: Dict[str, Any] | None = None) -> Dict[str, Any]:
    sys_text = build_judge_system_prompt(session_ctx)
    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": f"用户提问：{user_query}"}
    ]
    raw_resp = generate_answer_by_api(messages, llm_cfg=llm_cfg, user_query=user_query, timeout=12)
    judge_result = parse_llm_judge_response(
        raw_resp,
        base_entities=base_entities,
        session_alias_map=session_ctx.get("session_alias_map", {})
    )
    logger.info("[LLM‑Judge] 已解析意图结果")
    return judge_result


# ======================【第一部分】问句分析与补全Agent ======================
PLANNER_MAX_SUB_QUESTIONS = 4

def build_planner_prompt(user_query: str, session_ctx: Dict[str, Any], chat_history: list, route_info: Dict[str, Any] | None = None) -> str:
    """问句规划器系统提示词：拆解多问题 + 结合会话上下文补全省略/代词 + 决定答/反问/跳过"""
    latest_subject = session_ctx.get("latest_subject") or "（无）"
    candidate_list = session_ctx.get("candidate_spot_list") or []
    pref = session_ctx.get("user_preference") or {}
    last_intent = session_ctx.get("last_intent_type") or "（无）"
    alias_map = session_ctx.get("session_alias_map") or {}
    alias_lines = "\n".join(f"{k} → {v}" for k, v in alias_map.items()) or "（无）"
    entities = (route_info or {}).get("entities", {}) or {}
    history_lines = []
    # 最近3轮摘要（6条消息），供代词/省略回填定位更早的讨论主体
    for m in (chat_history or [])[-6:]:
        role = "用户" if m.get("role") == "user" else "客服"
        content = (m.get("content") or "")[:120]
        history_lines.append(f"{role}：{content}")
    history_text = "\n".join(history_lines) or "（无）"
    return f"""你是景区客服的问句分析与规划器，负责把用户输入整理成系统能执行的子问题清单。只输出```json ```代码块，禁止任何其他文字。
【会话上下文】
上一轮讨论主体：{latest_subject}
候选景区列表：{json.dumps(candidate_list, ensure_ascii=False)}
会话累计偏好：{json.dumps(pref, ensure_ascii=False)}
上一轮意图：{last_intent}
别名映射：
{alias_lines}
【本地初步识别（可能有遗漏或错误，仅供参考，你可以修正）】
城市={entities.get('city')} 景区={entities.get('spot_name')} 等级={entities.get('level')} 字段={entities.get('fields')}
疑似未收录景区={entities.get('unknown_spot') or '（无）'}
【最近对话】
{history_text}
【用户输入】
{user_query}
【规划规则】
1. 把用户输入拆解为若干个完整、自包含、不带代词的子问题；用户只问一件事时输出1个；
2. 代词与省略必须结合会话上下文补全："它/这个"→具体景区名，"周末呢"→"{{主体}}周末的开放时间是什么"；
3. 指代无法唯一确定（多个候选且上下文无法判断）→ 该子问题 decision=clarify，reason写明需要用户澄清什么；
4. 与景区旅游业务明显无关的问题（天气、酒店、机票等）→ decision=skip，reason一句话说明；
5. 无法确定是业务问题还是无关闲聊时，一律 decision=clarify，禁止 skip；
6. intent 只能取：spot_detail(门票/开放时间/地址/等级/电话)、rag_required(简介/特色/好玩)、city_scenic_list(城市景区清单)、scenic_count(数量统计)、recommendation(推荐攻略)；
7. spot_name/city 尽量使用标准全名；target_fields 只能取 ticket/open_time/address/telephone/level；子问题最多{PLANNER_MAX_SUB_QUESTIONS}个；
8. 你**不负责判断景区是否被数据库收录**——那由执行层的实体解析链和存在性查询判定。对拿不准是否收录的景区，一律 decision=answer 交执行层处理；只有本地识别明确给出【疑似未收录景区】名字时，才允许 decision=clarify，且reason只能写“未能确定所指景区，请用户确认”，**禁止出现“未收录”“数据库没有”等数据层结论**——你没有查询数据库的能力；
9. 对比类问题（哪个更适合/有什么区别/比较A和B）保持为1个子问题，禁止拆成两个；参与对比的所有景区必须逐个写入spot_name_list数组（spot_name只填第一个），**禁止把多个景区名拼接在spot_name一个字段里**；
10. 拆解多景区问题时，每个子问题只携带属于它自己的约束（“白云山只查门票，丹霞山只介绍地貌”→子问题1只问白云山门票、子问题2只问丹霞山地貌），禁止把其他景区的约束混入。
11. 用户表达“不要X/不去X/除了X之外/排除X”时：X必须逐个写入exclude_spot_list数组（可用非标准名，执行层会解析）；这些排除对象禁止填入spot_name，也不得因此拒绝回答——正常按城市推荐其余景区即可。
12. 用户条件涉及数据库不具备的筛选维度时（如分时票价：平日免费/节假日收费、按人群分段收费、按季节浮动定价等）：decision=skip，reason写明“该筛选维度（分时票价差异）数据库未收录”——不要映射到city_scenic_list/recommendation，也不要按关键词硬套“免费”。
【输出格式】
```json
{{"sub_questions":[{{"complete_query":"补全后的完整问句","intent":"spot_detail","spot_name":"","city":"","scenic_level":"","target_fields":[],"exclude_spot_list":[],"decision":"answer","reason":"给用户看的一句话说明"}}]}}
```"""


def plan_sub_questions(user_query: str, session_ctx: Dict[str, Any], chat_history: list,
                       route_info: Dict[str, Any] | None = None, llm_cfg: dict = None) -> Optional[list]:
    """
    【第一部分核心】问句分析与补全Agent。
    返回子问题列表（每项含 complete_query/intent/spot_name/city/scenic_level/target_fields/decision/reason）；
    解析失败返回 None，调用方降级为原单问题路径。
    """
    try:
        sys_text = build_planner_prompt(user_query, session_ctx, chat_history, route_info)
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": "输出规划结果JSON。"},
        ]
        raw = generate_answer_by_api(messages, llm_cfg=llm_cfg, timeout=15)
        logger.info("[QueryPlanner] 收到规划输出，开始解析")
        from protocol.internal.internal_schema_types import extract_json_from_llm_text
        json_str = extract_json_from_llm_text(raw)
        data = None
        if json_str:
            data = json.loads(json_str)
        else:
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                data = json.loads(m.group(0))
        subs = (data or {}).get("sub_questions")
        if not isinstance(subs, list) or not subs:
            logger.warning("[QueryPlanner] 未解析到子问题列表，降级原路径")
            return None
        out = []
        for s in subs[:PLANNER_MAX_SUB_QUESTIONS]:
            if not isinstance(s, dict):
                continue
            item = {
                "complete_query": str(s.get("complete_query") or "").strip(),
                "intent": str(s.get("intent") or "").strip(),
                "spot_name": str(s.get("spot_name") or "").strip(),
                "spot_name_list": s.get("spot_name_list") if isinstance(s.get("spot_name_list"), list) else [],
                "city": str(s.get("city") or "").strip(),
                "scenic_level": str(s.get("scenic_level") or "").strip(),
                "target_fields": s.get("target_fields") if isinstance(s.get("target_fields"), list) else [],
                "decision": str(s.get("decision") or "answer").strip().lower(),
                "exclude_spot_list": [str(x) for x in s.get("exclude_spot_list") if str(x).strip()]
                             if isinstance(s.get("exclude_spot_list"), list) else [],
            "reason": str(s.get("reason") or "").strip(),
            }
            if item["complete_query"] or item["decision"] != "answer":
                out.append(item)
        if not out:
            logger.warning("[QueryPlanner] 子问题列表为空，降级原路径")
            return None
        return out
    except LLMServiceError:
        raise
    except Exception as e:
        logger.warning("[QueryPlanner] 规划失败，降级原路径: type=%s", type(e).__name__)
        return None
