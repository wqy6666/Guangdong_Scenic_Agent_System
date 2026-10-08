""" data_preprocess.py 安全稳定版 无Langchain依赖，景区知识库文本切片脚本
已修复死循环BUG + 绝对路径定位 +【追加门票、开放时间、intro_val简介进切片文本】
"""
import sqlite3
import pandas as pd
from pathlib import Path

# ========================配置项========================
ROOT_PATH = Path(__file__).parent.parent
DB_PATH = ROOT_PATH / "shujuku" / "travel_final2.db"
TABLE_NAME = "tourism_spots_cleaned"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
# ======================================================

def recursive_split_text(text: str, chunk_size: int, chunk_overlap: int):
    separators = ["\n\n", "\n", "。", "！", "？", "，", " "]
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        if end < text_len:
            for sep in separators:
                cut_pos = chunk.rfind(sep)
                if cut_pos > 0:
                    end = start + cut_pos + len(sep)
                    chunk = text[start:end]
                    break
        chunks.append(chunk)
        new_start = end - chunk_overlap
        if new_start <= start:
            new_start = start + 1
        start = new_start
    return chunks


def load_scenic_data_from_sqlite() -> pd.DataFrame:
    db_file = DB_PATH
    print(f"[DEBUG] 数据库完整路径: {db_file.resolve()}")
    if not db_file.exists():
        raise FileNotFoundError(f"数据库文件不存在：{str(db_file)}")
    conn = sqlite3.connect(str(db_file))
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    print(f"[INFO] 成功读取数据库，景区记录总数：{len(df)}")
    return df


def split_scenic_text(df: pd.DataFrame):
    chunk_result = []
    for _, row in df.iterrows():
        raw_text = row["cleaned_content"]

        # ============核心修复：追加三块字段到检索文本============
        intro_info = row["intro_val"] if pd.notna(row["intro_val"]) else ""
        open_time = row["open_time_val"] if pd.notna(row["open_time_val"]) else ""
        ticket_price = row["ticket_val"] if pd.notna(row["ticket_val"]) else ""

        if intro_info:
            raw_text += f"\n景区简介：{intro_info}"
        if open_time:
            raw_text += f"\n景区开放时间：{open_time}"
        if ticket_price:
            raw_text += f"\n景区门票价格：{ticket_price}"

        spot_meta = {
            "spot_id": str(row["spot_id"]),
            "spot_name": str(row["spot_name"]),
            "city": str(row["city"]) if row["city"] is not None else ""
        }

        if not raw_text or pd.isna(raw_text):
            continue
        chunks = recursive_split_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk_text in enumerate(chunks):
            chunk_item = {
                "chunk_index": idx,
                "chunk_text": chunk_text,
                **spot_meta
            }
            chunk_result.append(chunk_item)
    print(f"[INFO] 文本切片完成，生成chunk总数量：{len(chunk_result)}")
    # 打印预览追加后的内容，用来校验
    print("\n====切片预览(前2条)====")
    for preview in chunk_result[:2]:
        print(f"景区名称:{preview['spot_name']},切片内容:\n{preview['chunk_text'][:350]}\n")
    return chunk_result


if __name__ == "__main__":
    df_data = load_scenic_data_from_sqlite()
    chunk_list = split_scenic_text(df_data)

