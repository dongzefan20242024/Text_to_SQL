import os
import json
from sentence_transformers import SentenceTransformer
from app.utils.pinecone_client import init_pinecone, create_or_connect_index, upsert_vectors

# 配置
DATA_FILE = "../../data/train.jsonl"
INDEX_NAME = "text-to-sql-index"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_jsonl(file_path):
    """加载 JSONL 数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def generate_embeddings(questions, model_name):
    """生成问题的嵌入"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(questions, show_progress_bar=True, batch_size=32)
    return embeddings

def upsert_to_pinecone(index_name, dataset, embeddings):
    """将嵌入和元数据上传到 Pinecone"""
    pc = init_pinecone()
    index = create_or_connect_index(pc, index_name, dimension=len(embeddings[0]))
    print(f"Connected to Pinecone index: {index_name}")

    vectors = []
    for i, data in enumerate(dataset):
        # 提取问题、SQL和table_id并构造metadata
        metadata = {
            "question": data["question"],
            "sql": json.dumps(data["sql"]),  # 将嵌套 JSON 数据转换为字符串
            "table_id": data.get("table_id", "unknown_table")  # 包含 table_id 信息
        }
        # 创建向量结构
        vectors.append({"id": f"query_{i}", "values": embeddings[i], "metadata": metadata})

        # 打印上传进度
        if i % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} records")

    print("Upserting vectors to Pinecone...")
    upsert_vectors(pc, index_name, vectors)
    print(f"Upserted {len(vectors)} vectors to Pinecone index: {index_name}")

if __name__ == "__main__":
    try:
        # 加载数据
        print(f"Loading records from {DATA_FILE}...")
        dataset = load_jsonl(DATA_FILE)
        print(f"Loaded {len(dataset)} records.")

        # 生成嵌入
        print(f"Generating embeddings using model: {MODEL_NAME}")
        questions = [record["question"] for record in dataset]
        embeddings = generate_embeddings(questions, MODEL_NAME)

        # 上传到 Pinecone
        print("Initializing Pinecone...")
        upsert_to_pinecone(INDEX_NAME, dataset, embeddings)
        print("Data upload completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
