"""
vector_index((beixuan).py
知识库切片加载 bge‑small‑zh‑v1.5生成向量，写入Elasticsearch
依赖安装：pip install elasticsearch transformers torch -i https://pypi.tuna.tsinghua.edu.cn/simple
"""
import torch
from transformers import AutoModel, AutoTokenizer
from elasticsearch import Elasticsearch
from pathlib import Path

# =====================配置=====================
ROOT_PATH = Path(__file__).parent.parent
# 修正模型文件夹名称：bge-small-zh-v1.5
EMBED_MODEL_PATH = str(ROOT_PATH / "model" / "bge-small-zh-v1.5")
ES_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "guangdong_scenic_vector"
BATCH_SIZE = 16
# bge‑small‑zh‑v1.5 向量维度依旧 512维
EMBED_DIM = 512
# 自动优先使用GPU(RTX3050)，无GPU自动切CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] 当前推理设备：{DEVICE}")
# =============================================

def load_embedding_model():
    """加载本地bge‑small‑zh‑v1.5向量化模型"""
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
    model = AutoModel.from_pretrained(EMBED_MODEL_PATH).to(DEVICE)
    model.eval()
    return tokenizer, model

def get_embedding(text, tokenizer, model):
    """生成单条文本向量，遵循bge官方前置指令"""
    instruction = "为这个句子生成表示以用于检索："
    input_text = instruction + text
    encoded_input = tokenizer(
        [input_text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(DEVICE)
    with torch.no_grad():
        model_output = model(**encoded_input)
    # bge取CLS向量
    embedding = model_output[0][:, 0].cpu().numpy()[0].tolist()
    return embedding

def init_es_index(es_client):
    """初始化ES向量索引，如已存在先删除重建"""
    if es_client.indices.exists(index=INDEX_NAME):
        es_client.indices.delete(index=INDEX_NAME)
        print(f"[INFO] 旧索引 {INDEX_NAME} 已删除")
    mapping = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "chunk_text": {"type": "text"},
                "spot_id": {"type": "keyword"},
                "spot_name": {"type": "keyword"},
                "city": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "vector": {
                    "type": "dense_vector",
                    "dims": EMBED_DIM,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    es_client.indices.create(index=INDEX_NAME, body=mapping)
    print(f"[INFO] 索引 {INDEX_NAME} 创建成功")


def batch_insert_es(chunk_list, tokenizer, model, es_client):
    total = len(chunk_list)
    for i in range(0, total, BATCH_SIZE):
        batch = chunk_list[i:i+BATCH_SIZE]
        for item in batch:
            vec = get_embedding(item["chunk_text"], tokenizer, model)
            doc = {
                "chunk_text": item["chunk_text"],
                "spot_id": item["spot_id"],
                "spot_name": item["spot_name"],
                "city": item["city"],
                "chunk_index": item["chunk_index"],
                "vector": vec
            }
            es_client.index(index=INDEX_NAME, document=doc)
        print(f"[INFO] 已入库 {min(i+BATCH_SIZE,total)} / {total}")
    print("[SUCCESS] 全部向量入库完成！")


if __name__ == "__main__":
    # 先导入切片模块，复用切片结果
    from data_preprocess import split_scenic_text, load_scenic_data_from_sqlite
    df = load_scenic_data_from_sqlite()
    chunks = split_scenic_text(df)

    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise ConnectionError("Elasticsearch服务未启动！请先运行ES，端口9200")

    tokenizer, model = load_embedding_model()
    init_es_index(es)
    batch_insert_es(chunks, tokenizer, model, es)
