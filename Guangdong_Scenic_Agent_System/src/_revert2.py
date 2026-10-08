# -*- coding: utf-8 -*-
"""精确复原：删除宽口径重查/分时披露/relaxed话术（行号锚定，基于当前文件状态）"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
p = 'rag_hybrid_query.py'
lines = open(p, encoding='utf-8').read().split(chr(10))
NL = chr(10)

# ---- 区段1：elif ticket_free 分支内的宽口径重查 + 分时披露（行1389-1397，0-based 1388-1396）----
# 锚：'sql_context = query_city_scenic_list(city_name, target_level, ticket_free=True)' 行
i_narrow = None
for i, line in enumerate(lines):
    if 'query_city_scenic_list(city_name, target_level, ticket_free=True)' in line and 'if not' not in line:
        i_narrow = i
        break
assert i_narrow is not None, 'narrow line not found'
# 从下一行开始，删除到 "分时票价差异" 拼接行（含）
j = i_narrow + 1
assert lines[j].strip() == 'if not sql_context:', lines[j]
k = j
while k < len(lines):
    if '分时票价差异，数据库未收录' in lines[k]:
        k += 1
        break
    k += 1
assert k > j, 'end not found'
del lines[j:k]
print('removed wide-retry block: lines', j + 1, '->', k)

# ---- 区段2：count 空话术去掉已放宽标记 ----
for i, line in enumerate(lines):
    if '（已尝试放宽免费口径仍无结果）' in line and 'unknown_note' in line:
        lines[i] = line.replace(' + "（已尝试放宽免费口径仍无结果）"', '')
        print('cleaned count empty at', i + 1)
        break

# ---- 区段3：list 空话术去掉已放宽标记并恢复单行 ----
i_list = None
for i, line in enumerate(lines):
    if '（已尝试放宽免费口径仍无结果），请确认筛选条件或更换城市。' in line:
        indent = line[:len(line) - len(line.lstrip())]
        base = '            answer = f"没有查询到{scope_text}记录。" + unknown_note + "请确认筛选条件或更换城市。"'
        # 用 unknown_note 行替换整个 if/elif/else 结构：从 if unknown_note: 向上找
        j_start = i
        while not lines[j_start].strip().startswith('if unknown_note:'):
            j_start -= 1
        lines[j_start:i + 1] = [indent + 'answer = f"没有查询到{scope_text}记录。" + unknown_note + "请确认筛选条件或更换城市。"']
        print('cleaned list empty at', i + 1)
        break

# ---- 区段4：删除 relaxed_note 追加两行 ----
i_rel = None
for i, line in enumerate(lines):
    if line.strip() == 'if relaxed_note:':
        i_rel = i
        break
assert i_rel is not None, 'relaxed append not found'
del lines[i_rel:i_rel + 2]
print('removed relaxed append at', i_rel + 1)

# ---- 区段5：删除 relaxed_note 初始化两处 ----
out = [line for line in lines if line.strip() != 'relaxed_note = ""']
print('removed inits:', len(lines) - len(out))
open(p, 'w', encoding='utf-8').write(NL.join(out))
print('revert done')
