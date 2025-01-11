import logging
from pinecone import Pinecone, ServerlessSpec
from config import Config

# 设置日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def init_pinecone():
    """初始化 Pinecone 客户端"""
    try:
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        logging.info("Initialized Pinecone client successfully.")
        return pc
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone: {e}")
        raise

def create_or_connect_index(pinecone_client, index_name, dimension):
    """创建或连接到 Pinecone 索引"""
    try:
        index_list = pinecone_client.list_indexes().names()
        if index_name not in index_list:
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=Config.PINECONE_ENV
                )
            )
            logging.info(f"Created new Pinecone index: {index_name}")
        else:
            logging.info(f"Index {index_name} already exists.")
        return pinecone_client.Index(index_name)
    except Exception as e:
        logging.error(f"Failed to create or connect to index {index_name}: {e}")
        raise

def upsert_vectors(pinecone_client, index_name, vectors, batch_size=100):
    """上传向量到 Pinecone"""
    try:
        index = pinecone_client.Index(index_name)
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            logging.info(f"Upserted batch {i // batch_size + 1} to Pinecone.")
        logging.info(f"All {len(vectors)} vectors upserted successfully.")
    except Exception as e:
        logging.error(f"Failed to upsert vectors to index {index_name}: {e}")
        raise

def query_pinecone(pinecone_client, index_name, vector, top_k=1):
    """查询 Pinecone 索引"""
    try:
        print(f"Querying Pinecone index: {index_name}...")
        index = pinecone_client.Index(index_name)

        # 查询索引
        results = index.query(vector=vector, top_k=top_k, include_metadata=True)

        # 检查是否有匹配结果
        if not results or "matches" not in results or len(results["matches"]) == 0:
            print("No matches found. Possible reasons:")
            print("1. The vector is not similar to existing vectors in the index.")
            print("2. The index name is incorrect or does not contain relevant data.")
            print("3. Pinecone query parameters need adjustment (e.g., top_k).")
            return None

        # 打印匹配结果
        print("Pinecone Query Results for Text-to-SQL:")
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            question = metadata.get("question", "No question")
            sql = metadata.get("sql", "No SQL")
            print(f"ID: {match['id']}, Score: {match['score']}, Question: {question}, SQL: {sql}")

        return results
    except Exception as e:
        print(f"Error querying Pinecone index: {e}")
        return None
