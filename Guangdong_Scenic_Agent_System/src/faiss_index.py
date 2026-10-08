import torch
import numpy as np
import pickle
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

# =====================配置=====================
ROOT_PATH = Path(__file__).parent.parent
EMBED_MODEL_PATH = str(ROOT_PATH / "model" / "bge-small-zh-v1.5")
VECTOR_NPY_FILE = str(ROOT_PATH / "src" / "vector_array.npy")
META_FILE = str(ROOT_PATH / "src" / "chunk_meta.pkl")
EMBED_DIM = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] 当前推理设备：{DEVICE}")
print("向量保存路径：",VECTOR_NPY_FILE)
print("元数据保存路径：",META_FILE)
# =============================================

def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
    model = AutoModel.from_pretrained(EMBED_MODEL_PATH).to(DEVICE)
    model.eval()
    return tokenizer, model

def get_embedding(text, tokenizer, model):
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
    embedding = model_output[0][:, 0].cpu().numpy()[0]
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


if __name__ == "__main__":
    from data_preprocess import split_scenic_text, load_scenic_data_from_sqlite
    df = load_scenic_data_from_sqlite()
    chunks = split_scenic_text(df)

    tokenizer, model = load_embedding_model()
    meta_list = []
    embeddings = []
    total = len(chunks)

    for idx, item in enumerate(chunks):
        vec = get_embedding(item["chunk_text"], tokenizer, model)
        embeddings.append(vec)
        meta_list.append(item)
        if (idx + 1) % 500 == 0:
            print(f"[INFO] 已生成向量 {idx+1}/{total}")

    # numpy保存向量文件
    embeddings_np = np.array(embeddings).astype(np.float32)
    np.save(VECTOR_NPY_FILE, embeddings_np)
    with open(META_FILE, "wb") as f:
        pickle.dump(meta_list, f)

    print(f"[SUCCESS] 向量库保存成功！一共 {total} 条切片")
