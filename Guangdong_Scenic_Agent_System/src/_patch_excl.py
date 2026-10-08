# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
NL = chr(10)
BS = chr(92)

# ===== 修复1：rag_hybrid_query 意图循环——推荐意图一律不回填参考主体 =====
p = 'rag_hybrid_query.py'
src = open(p, encoding='utf-8').read()
old = ('                # 【修复】排除类推荐：被排除景区不得经本地回填变成推荐参考主体' + NL +
       "                if intent == \"recommendation\" and base_entities.get(\"spot_name\") in (sub_judge.get(\"exclude_spot_list\") or []):" + NL +
       '                    logger.info(f"【实体回填拦截】{base_entities.get(\'spot_name\')}在排除清单中，不回填为推荐参考主体")' + NL +
       "                elif not sub_judge.get(\"spot_name\") and base_entities.get(\"spot_name\"):" + NL +
       '                    sub_judge["spot_name"] = base_entities["spot_name"]')
new = ('                # 【修复·加强】推荐意图一律不回填参考主体：ref_spot仅用于"像X一样"类比，' + NL +
       '                # 回填会把排除对象/顺带提及的景区错当类比参考（"不要白云山"→类比白云山）。' + NL +
       '                # 类比场景由Judge/规划器自行填写spot_name，本地只兜底city。' + NL +
       "                if intent == \"recommendation\":" + NL +
       '                    if base_entities.get("spot_name"):' + NL +
       '                        logger.info(f"【实体回填拦截】推荐意图不回填参考主体（本地识别到{base_entities[\'spot_name\']}，仅作排除/顺带提及处理）")' + NL +
       "                elif not sub_judge.get(\"spot_name\") and base_entities.get(\"spot_name\"):" + NL +
       '                    sub_judge["spot_name"] = base_entities["spot_name"]')
assert src.count(old) == 1, ('fix1', src.count(old))
src = src.replace(old, new)
open(p, 'w', encoding='utf-8').write(src)
print('fix1 ok')

# ===== 修复2：merge_base_entities——推荐意图不回填参考主体（Judge路径同规则） =====
p = 'protocol/internal/internal_schema_types.py'
src = open(p, encoding='utf-8').read()
old = ('    if base_spot is not None and base_spot.strip() != "":' + NL +
       '        # 【修复】排除类推荐：被用户排除的景区不得经本地回填变成推荐参考主体' + NL +
       '        if base_spot in exclude_names:' + NL +
       '            logger.info(f"【实体合并拦截】{base_spot}在排除清单中，不回填为spot_name")' + NL +
       '        elif out.get("spot_name", "") != base_spot:' + NL +
       '            logger.warning(f"【实体合并】LLM‑Judge spot_name={out.get(\'spot_name\')}，覆盖使用本地base_spot={base_spot}")' + NL +
       '            out["spot_name"] = base_spot')
new = ('    if base_spot is not None and base_spot.strip() != "":' + NL +
       '        # 【修复】排除类推荐：被用户排除的景区不得经本地回填变成推荐参考主体' + NL +
       '        if base_spot in exclude_names:' + NL +
       '            logger.info(f"【实体合并拦截】{base_spot}在排除清单中，不回填为spot_name")' + NL +
       '        elif out.get("intent_type") == "recommendation":' + NL +
       '            # 推荐意图一律不回填参考主体：ref_spot仅用于"像X一样"类比，回填会把排除对象错当参考' + NL +
       '            logger.info(f"【实体合并拦截】推荐意图不回填参考主体（本地识别到{base_spot}）")' + NL +
       '        elif out.get("spot_name", "") != base_spot:' + NL +
       '            logger.warning(f"【实体合并】LLM‑Judge spot_name={out.get(\'spot_name\')}，覆盖使用本地base_spot={base_spot}")' + NL +
       '            out["spot_name"] = base_spot')
assert src.count(old) == 1, ('fix2', src.count(old))
src = src.replace(old, new)
open(p, 'w', encoding='utf-8').write(src)
print('fix2 ok')

# ===== 修复3：规划器提示词——排除语义入 exclude_spot_list =====
p = 'llm_judge_call.py'
src = open(p, encoding='utf-8').read()
old = ('10. 拆解多景区问题时，每个子问题只携带属于它自己的约束（“白云山只查门票，丹霞山只介绍地貌”→子问题1只问白云山门票、子问题2只问丹霞山地貌），禁止把其他景区的约束混入。')
new = (old + NL +
       '11. 用户表达“不要X/不去X/除了X之外/排除X”时：X必须逐个写入exclude_spot_list数组（可用非标准名，执行层会解析）；'
       '这些排除对象禁止填入spot_name，也不得因此拒绝回答——正常按城市推荐其余景区即可。')
assert src.count(old) == 1, ('fix3-rule', src.count(old))
src = src.replace(old, new)

old_f = ('            "reason": str(s.get("reason") or "").strip(),' + NL)
new_f = ('            "exclude_spot_list": [str(x) for x in s.get("exclude_spot_list") if str(x).strip()]' + NL +
         '                         if isinstance(s.get("exclude_spot_list"), list) else [],' + NL +
         '            "reason": str(s.get("reason") or "").strip(),' + NL)
assert src.count(old_f) == 1, ('fix3-field', src.count(old_f))
src = src.replace(old_f, new_f)

old_c = ('【输出格式】' + BS + 'n' +
         '```json' + BS + 'n' +
         '{{"sub_questions":[{{"complete_query":"补全后的完整问句","intent":"spot_detail","spot_name":"","city":"","scenic_level":"","target_fields":[],"decision":"answer","reason":"给用户看的一句话说明"}}]}}' + BS + 'n')
new_c = ('【输出格式】' + BS + 'n' +
         '```json' + BS + 'n' +
         '{{"sub_questions":[{{"complete_query":"补全后的完整问句","intent":"spot_detail","spot_name":"","city":"","scenic_level":"","target_fields":[],"exclude_spot_list":[],"decision":"answer","reason":"给用户看的一句话说明"}}]}}' + BS + 'n')
assert src.count(old_c) == 1, ('fix3-contract', src.count(old_c))
src = src.replace(old_c, new_c)
open(p, 'w', encoding='utf-8').write(src)
print('fix3 ok')

# ===== 修复4：_build_pre_judge 透传解析后的 exclude_spot_list =====
p = 'agent_reflection_loop.py'
src = open(p, encoding='utf-8').read()
old = ('        "target_fields": [f for f in (sub.get("target_fields") or [])' + NL +
       '                          if f in ("ticket", "open_time", "address", "telephone", "level")],')
new = ('        "target_fields": [f for f in (sub.get("target_fields") or [])' + NL +
       '                          if f in ("ticket", "open_time", "address", "telephone", "level")],' + NL +
       '        # 规划器表达的排除对象经解析链归一后透传，供推荐分支确定性剔除' + NL +
       '        "exclude_spot_list": [r for r in (resolve_spot_mention(str(n)) for n in (sub.get("exclude_spot_list") or [])) if r],')
assert src.count(old) == 1, ('fix4', src.count(old))
src = src.replace(old, new)
open(p, 'w', encoding='utf-8').write(src)
print('fix4 ok')
