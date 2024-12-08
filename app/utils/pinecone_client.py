from pinecone import Pinecone, ServerlessSpec
from config import Config


def init_pinecone():
    """
    初始化 Pinecone 客户端实例并返回。
    """
    pc = Pinecone(
        api_key=Config.PINECONE_API_KEY  # 使用您的 API Key
    )
    return pc


def create_or_connect_index(index_name, dimension=128):
    """
    创建或连接到 Pinecone 索引。

    Args:
        index_name (str): 索引名称
        dimension (int): 嵌入向量的维度（默认128）
    Returns:
        dict: 已连接的索引的详细信息
    """
    pc = init_pinecone()

    # 检查索引是否存在
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",  # 度量方法
            spec=ServerlessSpec(
                cloud="aws",  # 云服务提供商
                region="us-east-1"  # 区域
            )
        )

    # 获取索引的详细信息
    return pc.describe_index(index_name)


def upsert_vectors(pc_instance, index_name, embeddings):
    """
    插入或更新向量到索引中。

    Args:
        pc_instance (Pinecone): Pinecone 客户端实例
        index_name (str): 索引名称
        embeddings (list): 包含向量的字典列表，例如:
                           [{"id": "vector1", "values": [...]}]
    """
    index = pc_instance.Index(index_name)
    index.upsert(vectors=embeddings)


def query_vector(pc_instance, index_name, query_vector, top_k=5):
    """
    检索最相似的向量。

    Args:
        pc_instance (Pinecone): Pinecone 客户端实例
        index_name (str): 索引名称
        query_vector (list): 用户查询向量
        top_k (int): 返回最相似的向量数
    Returns:
        list: 匹配结果的列表
    """
    index = pc_instance.Index(index_name)
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return response.get("matches", [])
