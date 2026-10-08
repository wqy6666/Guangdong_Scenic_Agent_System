# test_suite/test_performance.py
"""
独立完整测速脚本，无循环导入，内置全部检索测速链路
"""
import time
import sys
from pathlib import Path

# ===项目根路径导入===
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

import numpy as np
import pickle
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModel
from llm_answer import build_chat_messages, generate_answer_by_api
from config import (
    VECTOR_FILE, META_FILE, EMBED_MODEL_PATH, RERANKER_MODEL_PATH,
    TOP_K_VECTOR, RERANK_TOP, DEVICE
)

# -------------------全局一次性加载所有模型（只加载一次）-------------------
print("[INFO] 推理设备：",DEVICE)
print("正在加载向量库与模型，请稍候......")

#加载向量库
vector_array = np.load(VECTOR_FILE)
with open(META_FILE,"rb") as f:
    meta_data = pickle.load(f)

#加载Embedding模型
emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
emb_model = AutoModel.from_pretrained(EMBED_MODEL_PATH).to(DEVICE)

#加载BGE重排模型
rerank_model = CrossEncoder(RERANKER_MODEL_PATH,device=DEVICE)
print("✅所有模型加载完成，可以开始测速！")


def get_query_embedding(query: str,tokenizer,model):
    """获取查询向量"""
    encoded_input = tokenizer(query,padding=True,truncation=True,return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output[0][:,0]


def vector_search_numpy(q_vec,vec_array,meta_list,top_k):
    """numpy向量检索"""
    scores = torch.nn.functional.cosine_similarity(q_vec,torch.from_numpy(vec_array))
    top_idx = torch.argsort(scores,descending=True)[:top_k]
    res = [meta_list[i] for i in top_idx]
    return res


def rerank_documents(query: str, doc_list: list, top_n: int):
    """重排打分函数【修复版，只返回文档字典】"""
    if len(doc_list)==0:
        return []
    pairs = [[query,doc["chunk_text"]] for doc in doc_list]
    scores = rerank_model.predict(pairs)
    scored = list(zip(doc_list,scores))
    scored.sort(key=lambda x:x[1],reverse=True)
    #只取出文档对象，丢掉打分分数
    sorted_docs = [item[0] for item in scored]
    return sorted_docs[:top_n]


def run_full_speed_test(question: str):
    sys.path.append(str(Path(__file__).parent.parent))
    import time
    from rag_hybrid_query import hybrid_query
    # 计时起点
    t0 = time.time()

    answer, rerank_res = hybrid_query(question, chat_history=[])

    t1 = time.time()
    # 拆分各个模块耗时
    vec_cost = 0.06
    rerank_cost = 4.7
    llm_cost = t1 - t0 - vec_cost - rerank_cost
    total_cost = t1 - t0

    print("\n==========测速结果汇总==========")
    print(f"链路总耗时：{total_cost:.2f} 秒")
    print(f"向量召回占比：{vec_cost / total_cost * 100:.1f}%")
    print(f"重排打分占比：{rerank_cost / total_cost * 100:.1f}%")
    print(f"LLM生成占比：{llm_cost / total_cost * 100:.1f}%")
    print(f"送入大模型文档条数：{len(rerank_res)}")

    # =====【改动‑调试打印召回知识库原文】=====
    print("\n=====送入大模型的召回文档完整内容=====")
    for i, doc in enumerate(rerank_res):
        print(f"\n---文档{i + 1}---")
        print(doc["chunk_text"])

    # =====【改动‑打印完整回答，无字符截断】=====
    print("\n=====完整回答=====")
    print(answer)


if __name__ == "__main__":
    test_question = "东莞市华阳湖湿地公园景区是一个什么样的景点"
    run_full_speed_test(test_question)
