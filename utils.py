from sentence_transformers import SentenceTransformer
import pinecone

# 初始化嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 初始化 Pinecone
pinecone.init(api_key="your_api_key", environment="us-west1-gcp")
index = pinecone.Index("text-to-sql")

def retrieve_context(user_query, schema):
    """检索与用户问题相关的数据库架构上下文"""
    query_embedding = model.encode([user_query])[0]
    results = index.query(query_embedding, top_k=1, include_metadata=True)
    return results['matches'][0]['metadata']['description']

def generate_sql(user_query, context):
    """生成 SQL 查询"""
    from transformers import pipeline
    generator = pipeline("text-generation", model="gpt-neo-2.7B")
    prompt = f"Context: {context}\nQuestion: {user_query}\nSQL:"
    result = generator(prompt, max_length=150, num_return_sequences=1)
    return result[0]['generated_text']

def generate_schema_embeddings(schema):
    """生成数据库架构的嵌入并存储到 Pinecone"""
    descriptions = [
        f"Table {table['name']} with columns {', '.join(table['columns'])}" for table in schema['tables']
    ]
    embeddings = model.encode(descriptions)
    for i, embedding in enumerate(embeddings):
        index.upsert([(f"schema_{i}", embedding, {"description": descriptions[i]})])
