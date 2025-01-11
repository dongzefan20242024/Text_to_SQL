from app.utils.pinecone_client import init_pinecone, create_or_connect_index, upsert_vectors, query_vector

# 初始化 Pinecone
pc_instance = init_pinecone()

# 索引名称
index_name = "rag"

# 创建或连接索引
index_info = create_or_connect_index(index_name, dimension=128)
print(f"index: {index_info}")

# 插入数据
embeddings = [
    {"id": "vector1", "values": [0.1] * 128},  # 示例嵌入向量
    {"id": "vector2", "values": [0.2] * 128}
]
print("insert：", embeddings)
upsert_vectors(pc_instance, index_name, embeddings)
print("over！")

# 查询数据
query_vector_data = [0.15] * 128  # 示例查询向量
print("query：", query_vector_data)

results = query_vector(pc_instance, index_name, query_vector_data, top_k=5)

# 打印检索结果
print("result：")
if results:
    for match in results:
        print(f"ID: {match['id']}, Score: {match['score']}")
else:
    print("not found")
