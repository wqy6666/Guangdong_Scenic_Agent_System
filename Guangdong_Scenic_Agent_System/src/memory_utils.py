# -*- coding: utf-8 -*-
"""
memory_utils.py 记忆工具工具集
功能：
1. extract_keywords：jieba轻量分词提取关键词，过滤停用词，最多30个
2. detect_lookup_reference：检测用户query是否包含回看/指代关键词，判断是否触发short2检索
3. process_retrieve_hit_items：检索命中结果后处理流水线：去重、排序取top5、文本压缩保原意
"""
import jieba
from typing import List, Dict, Tuple

# ===================== 配置常量 =====================
MAX_KEYWORD_COUNT = 30
REFERENCE_TRIGGER_WORDS = {
    "刚才", "刚刚", "上一次", "上次", "之前", "前面", "前边",
    "先前", "还记得", "之前说的", "上回", "刚刚说的",
    # 【新增】代词/指代/复述要求类提问也触发历史检索，覆盖"它的开放时间""复述我的全部要求"等
    "它", "这个", "该景区", "前面提到的", "复述", "全部要求", "具体"
}
# 基础中文停用词
STOP_WORDS = {
    "的", "了", "是", "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "这", "那", "就", "都", "而", "及", "和", "与", "也", "还", "又", "再",
    "很", "比较", "非常", "太", "个", "一些", "有些", "什么", "哪里", "怎么",
    "为什么", "吗", "呢", "吧", "啊", "哦", "嗯", "噢", "之", "以", "对于", "关于"
}
# 文本压缩最大单段字符阈值，超过该值执行精简压缩
SINGLE_PARA_COMPRESS_THRESHOLD = 600
# ====================================================


def extract_keywords(question: str, answer: str) -> List[str]:
    """
    轻量本地分词提取关键词
    :param question: 用户问题
    :param answer: agent回答
    :return: 关键词列表，去重、过滤停用词，最多返回MAX_KEYWORD_COUNT个
    """
    full_text = f"{question} {answer}"
    words = jieba.cut(full_text)
    result = set()
    for w in words:
        w_strip = w.strip()
        if len(w_strip) < 2:
            continue
        if w_strip in STOP_WORDS:
            continue
        result.add(w_strip)
    kw_list = list(result)[:MAX_KEYWORD_COUNT]
    return kw_list


def detect_lookup_reference(query: str) -> bool:
    """
    检测用户输入是否包含回看历史指代关键词
    :param query: 用户原始提问
    :return: True 需要触发short2索引检索；False 不触发检索
    """
    for word in REFERENCE_TRIGGER_WORDS:
        if word in query:
            return True
    return False


def _simple_text_compress(text: str, keep_chars: int = 450) -> str:
    """
    简单文本压缩，保留原意，截断后补充省略标记；只用于检索返回片段
    :param text: 原始长文本
    :param keep_chars: 保留最大字符数
    :return: 压缩后的文本
    """
    if len(text) <= keep_chars:
        return text
    # 按换行分割，优先保留前面对话主体
    lines = text.splitlines()
    output_lines = []
    total = 0
    for line in lines:
        if total + len(line) > keep_chars:
            break
        output_lines.append(line)
        total += len(line)
    compressed = "\n".join(output_lines)
    return compressed + "\n……【内容过长已精简】"


def process_retrieve_hit_items(hit_index_list: List[Dict], mem_mgr) -> List[str]:
    """
    检索命中结果完整处理流水线
    1.去重，同一关键词内容优先最新段落；
    2.简单评分排序，取top5；
    3.字符超限做压缩，保证原意不变
    :param hit_index_list: search_short2_index返回的索引列表 [{"para_id":int,"keywords":list,"is_bad":bool}]
    :param mem_mgr: SessionMemoryManager实例，用于读取段落原文
    :return: 处理完成的对话片段文本列表
    """
    # step1：按para_id去重，保留最大para_id（最新）
    para_id_set = set()
    unique_items: List[Dict] = []
    for item in hit_index_list:
        pid = item["para_id"]
        if pid in para_id_set:
            continue
        para_id_set.add(pid)
        unique_items.append(item)

    # step2：简易评分，para_id越大（越新）分数越高；取top‑5
    # 简易策略：直接按段落id降序，越新得分越高
    unique_items.sort(key=lambda x: x["para_id"], reverse=True)
    top5_items = unique_items[:5]

    # step3读取段落原文
    fragment_list: List[str] = []
    for idx_item in top5_items:
        para_text = mem_mgr.read_paragraph_by_id(idx_item["para_id"])
        if para_text is None:
            continue
        # 超过阈值执行压缩
        if len(para_text) > SINGLE_PARA_COMPRESS_THRESHOLD:
            para_text = _simple_text_compress(para_text)
        fragment_list.append(para_text)
    return fragment_list


def combine_memory_fragments(history_fragments: List[str], long_memory_dict: Dict) -> str:
    """
    将【检索得到历史片段】与【长期记忆偏好】组合，输出统一文本送入记忆轮
    :param history_fragments: process_retrieve_hit_items输出的历史对话片段
    :param long_memory_dict: get_long_memory返回的长期记忆字典 hobby/price/time/spot_type
    :return: 组合完成的完整记忆轮文本
    """
    blocks = []
    if history_fragments:
        blocks.append("【会话历史参考片段】")
        blocks.extend(history_fragments)

    # 组装长期偏好
    preference_lines = []
    hobby = long_memory_dict.get("hobby")
    price = long_memory_dict.get("price")
    time_pref = long_memory_dict.get("time")
    spot_type = long_memory_dict.get("spot_type")
    if hobby:
        preference_lines.append(f"用户爱好偏好：{hobby}")
    if price:
        preference_lines.append(f"用户价格偏好：{price}")
    if time_pref:
        preference_lines.append(f"用户时间偏好：{time_pref}")
    if spot_type:
        preference_lines.append(f"用户景区类型偏好：{spot_type}")

    if preference_lines:
        blocks.append("【用户长期偏好】")
        blocks.extend(preference_lines)

    return "\n".join(blocks)


if __name__ == "__main__":
    # 简易单元测试
    q = "刚才说的丹霞山还有什么看点？"
    a = "丹霞山有阳元石、长老峰等景点。"
    kw = extract_keywords(q, a)
    print("提取关键词：", kw)

    ref_hit = detect_lookup_reference(q)
    print("是否触发回看检索：", ref_hit)

    frag = ["用户：介绍丹霞山\n助手：丹霞山是5A景区……"]
    long_pref = {"hobby":"山水风景","price":"中等","time":"周末","spot_type":"自然风光"}
    out = combine_memory_fragments(frag, long_pref)
    print("\n组合记忆轮输出：\n", out)
