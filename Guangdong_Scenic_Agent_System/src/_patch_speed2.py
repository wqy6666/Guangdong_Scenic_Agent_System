# -*- coding: utf-8 -*-
"""提速第二轮：A一轮补齐所有缺失目标 B核验限长 C生成限长 D核验prompt瘦身"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
NL = chr(10)
BS = chr(92)

# ===== B前置：generate_answer_by_api 支持 max_tokens =====
p = 'llm_answer.py'
src = open(p, encoding='utf-8').read()
old0 = 'def generate_answer_by_api(messages, llm_cfg: dict, user_query: str = "", timeout: int = 25) -> str:'
new0 = 'def generate_answer_by_api(messages, llm_cfg: dict, user_query: str = "", timeout: int = 25, max_tokens: int = None) -> str:'
assert src.count(old0) == 1, ('sig', src.count(old0))
src = src.replace(old0, new0)

old0b = '''            stream_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                timeout=timeout
            )'''
new0b = '''            stream_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                timeout=timeout,
                **({"max_tokens": max_tokens} if max_tokens else {})
            )'''
assert src.count(old0b) == 1, ('create', src.count(old0b))
src = src.replace(old0b, new0b)
open(p, 'w', encoding='utf-8').write(src)
print('B ok')

# ===== A+C：agent_reflection_loop.py =====
p = 'agent_reflection_loop.py'
src = open(p, encoding='utf-8').read()

# A1. 删除 _pick_target_spot（被批量补齐取代）
old_a1 = ('def _pick_target_spot(verify_res: dict, default_spot: str, exclude=None) -> str:' + NL +
          '    """' + NL +
          '    【授权链末端】补充检索的目标景区：优先从核验器的缺失点描述中解析具体景区' + NL +
          '    （核验器知道收录状态标准名），解析不出才退回初始识别实体。' + NL +
          '    exclude=已补充过的景区集合：缺失点同时提及多个景区时按轮转取未补充的那个，' + NL +
          '    避免包含匹配的顺序偏差导致反复补同一个景区。全部已补充或无法解析返回None。' + NL +
          '    """' + NL +
          '    exclude = exclude or set()' + NL +
          '    for mp in (verify_res or {}).get("missing_points") or []:' + NL +
          '        r = resolve_spot_mention(str(mp.get("point", "")))' + NL +
        '        if r and r not in exclude:' + NL +
        '            return r' + NL +
        '    if default_spot and default_spot not in exclude:' + NL +
        '        return default_spot' + NL +
        '    return None' + NL +
        NL)
assert src.count(old_a1) == 1, ('A1-del', src.count(old_a1))
src = src.replace(old_a1, '')

# A2. 补充循环：一轮批量补齐所有可解析缺失目标
old_a2 = ('''        action = verify_res["action"]
        supplemented = False
        target_spot = None
        if action in (REFLECT_ACTION_RAG, REFLECT_ACTION_SQL):
            # 【修复】补充目标按缺失点解析并轮转去重：缺失点同时提及多景区时轮流补充
            target_spot = _pick_target_spot(verify_res, spot_name, supplemented_spots)
            if not target_spot:
                logger.info("[AnswerVerify] 缺失点涉及景区均已补充过，无新目标，结束迭代")
                break
        if action == REFLECT_ACTION_RAG and target_spot:
            new_docs, new_rag_text = run_single_rag_tool(user_query, target_spot, city_name, session_ctx, llm_cfg=llm_cfg)
            exist_chunks = {d.get("chunk_text") for d in collected_docs}
            new_valid = [d for d in new_docs if d.get("chunk_text") not in exist_chunks]
            if new_valid:
                _collect(new_rag_text, new_valid)
                supplemented = True
        elif action == REFLECT_ACTION_SQL and target_spot:
            new_sql_text = run_single_sql_tool(target_spot)
            if new_sql_text.strip() not in seen_content_set:
                collected_answer_parts.append(new_sql_text)
                seen_content_set.add(new_sql_text.strip())
                collected_material_parts.append(new_sql_text)
                supplemented = True
        if supplemented:
            supplemented_spots.add(target_spot)
        else:
            logger.info("[AnswerVerify] 无可执行补充动作或无新数据，结束迭代")
            break''')
new_a2 = ('''        action = verify_res["action"]
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
            break''')
assert src.count(old_a2) == 1, ('A2-loop', src.count(old_a2))
src = src.replace(old_a2, new_a2)

# C1. 对比类RAG生成限长（多白名单分支）
old_c1 = '''                        rerank_chunk = rerank(user_query, filtered, RERANK_TOP)
                        msg = build_chat_messages(user_query, rerank_chunk, merged_history_for_llm, session_ctx)
                        rag_ans = generate_answer_by_api(msg, llm_cfg, user_query)
'''
# 该片段在rag_required分支内（前面是whitelist过滤），以whitelist行做锚
old_c1 = ('''                        filtered = post_filter_docs(unique_chunk, session_ctx.get("city"), session_ctx["allow_cross_city"], whitelist)
                        if len(multi_list) >= 2:
                            # 【修复】对比类配额重排：每景区各取前4条，防止单边挤占TOP5导致另一边资料缺失
                            rerank_chunk = []
                            for w in whitelist:
                                w_docs = [d for d in filtered if d.get("spot_name") == w]
                                rerank_chunk.extend(rerank(user_query, w_docs, 4))
                        else:
                            rerank_chunk = rerank(user_query, filtered, RERANK_TOP)
                        msg = build_chat_messages(user_query, rerank_chunk, merged_history_for_llm, session_ctx)
                        rag_ans = generate_answer_by_api(msg, llm_cfg, user_query)
''')
new_c1 = (old_c1.replace('rag_ans = generate_answer_by_api(msg, llm_cfg, user_query)',
                         'rag_ans = (generate_answer_by_api(msg, llm_cfg, user_query, max_tokens=800) if len(multi_list) >= 2 else generate_answer_by_api(msg, llm_cfg, user_query))'))
assert src.count(old_c1) == 1, ('C1', src.count(old_c1))
src = src.replace(old_c1, new_c1)

# C2. 合并重写限长2000（防跑飞的长文）
old_c2 = '        final_answer = generate_answer_by_api(final_messages, llm_cfg=llm_cfg, timeout=18)'
new_c2 = '        final_answer = generate_answer_by_api(final_messages, llm_cfg=llm_cfg, timeout=18, max_tokens=2000)'
assert src.count(old_c2) == 1, ('C2', src.count(old_c2))
src = src.replace(old_c2, new_c2)

# B+D. 核验调用限长500 + prompt瘦身
old_d = '''    raw = generate_answer_by_api(messages, llm_cfg=llm_cfg, timeout=20)
    logger.info(f"[AnswerVerify] 核验器原始输出: {raw}")'''
new_d = '''    raw = generate_answer_by_api(messages, llm_cfg=llm_cfg, timeout=20, max_tokens=500)
    logger.info(f"[AnswerVerify] 核验器原始输出: {raw}")'''
assert src.count(old_d) == 1, ('B-verify', src.count(old_d))
src = src.replace(old_d, new_d)

# D. 核验prompt瘦身（压缩规则文本，保留全部核验能力与关键短语）
old_e_start = '    sys_prompt = ('
i = src.find(old_e_start)
j = src.find('    )' + NL + '    messages = [', i)
assert i != -1 and j != -1, ('D-locate', i, j)
old_prompt = src[i:j + len('    )')]
new_prompt = '''    sys_prompt = (
        "你是景区客服回答核验器，只输出JSON。用【子问题清单】（括号内景区收录状态=数据库事实）对账【回答草稿】，资料原文是唯一事实依据。\\n"
        f"【用户提问】{user_query}\\n"
        f"【子问题清单】\\n{checklist_text}\\n"
        f"【回答草稿】\\n{draft_answer}\\n"
        f"【资料原文（编号片段）】\\n{materials_text}\\n"
        + pref_line +
        "【规则】\\n"
        "1.覆盖（负向声明对账）：清单每个诉求点是否被回应；资料确无、回答如实声明“暂无”=已覆盖；状态“已收录”的景区答“暂无”=缺失（missing_points，action=sql用标准名重查）；“知识库有相关内容”的答“暂无”=缺失（action=rag）；“未解析（已查询无结果）”的答“暂无”=如实覆盖。\\n"
        "2.溯源：草稿关键事实（价格/时间/地址/名称/等级/电话）须能在资料原文找到依据，无依据的列入unsupported_claims。\\n"
        "3.action：有缺失→sql（门票/时间/地址/等级/电话类）或rag（简介/特色类）；无从补充或仅unsupported（无缺失）→done。\\n"
        '【输出】{"coverage":"fully或partially","missing_points":[{"point":"…","source":"片段N或null"}],"unsupported_claims":["…"],"action":"done或rag或sql","reason":"一句话"}'
    )'''
src = src.replace(old_prompt, new_prompt)
open(p, 'w', encoding='utf-8').write(src)
print('A/C/D ok')
