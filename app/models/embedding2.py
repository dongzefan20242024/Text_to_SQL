import logging
import json
import requests
from sentence_transformers import SentenceTransformer
from typing import Optional
from app.utils.pinecone_client import init_pinecone, query_pinecone
from app.lib.query import Query

#################################
# 配置区域
#################################
LOG_LEVEL = logging.INFO
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
TEXT_TO_SQL_INDEX_NAME = "text-to-sql-index"
TABLE_TO_SQL_INDEX_NAME = "table-to-sql-index"
TOP_K = 1  # 查询 Pinecone 返回的匹配数目

# 日志配置
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

#################################
# 初始化向量模型
#################################
logging.info("Initializing embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
logging.info("Embedding model loaded.")

#################################
# 工具函数：从表头补全 SQL 列名
#################################
def build_real_sql(q: Query, table_metadata: dict) -> str:
    """
    基于 Query 对象和表元数据构建完整的 SQL 查询语句。
    """
    headers = table_metadata["headers"]
    col_map = {i: header for i, header in enumerate(headers)}

    # 解析 Query 对象
    agg_op = q.agg_ops[q.agg_index]
    sel_col = col_map.get(q.sel_index, f"col{q.sel_index}")

    if agg_op == "":
        select_part = f"SELECT `{sel_col}`"
    else:
        select_part = f"SELECT {agg_op}(`{sel_col}`)"

    from_part = f"FROM `{table_metadata['table_id']}`"

    where_clauses = []
    for col_i, op_i, val in q.conditions:
        col_name = col_map.get(col_i, f"col{col_i}")
        op_symbol = q.cond_ops[op_i]
        escaped_val = val.replace("'", "''")  # 防止 SQL 注入
        where_clauses.append(f"`{col_name}` {op_symbol} '{escaped_val}'")

    where_part = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    sql_query = f"{select_part} {from_part} {where_part};"
    return sql_query

#################################
# 查询表元数据
#################################
def query_table_pinecone(table_id: str) -> Optional[dict]:
    """
    从 Pinecone 的 table-to-sql-index 中获取指定 table_id 的列头信息和行数据。
    """
    try:
        pc = init_pinecone()
        query_vector = embedding_model.encode(table_id).tolist()
        results = query_pinecone(pc, TABLE_TO_SQL_INDEX_NAME, query_vector, top_k=TOP_K)

        if not results or not results.get("matches"):
            logging.error(f"No matches found for table_id: {table_id} in table-to-sql-index.")
            return None

        # 解析查询结果
        for match in results["matches"]:
            metadata = match.get("metadata", {})
            if metadata.get("table_id") == table_id:
                headers = metadata.get("headers", [])
                return {"table_id": table_id, "headers": headers}

        logging.error(f"No headers found for table_id: {table_id}")
        return None
    except Exception as e:
        logging.error(f"Error fetching table metadata from Pinecone for table_id {table_id}: {e}")
        return None

#################################
# 调用 Ollama 服务
#################################
def generate_response_with_ollama(prompt: str, model: str = "llama3.2") -> str:
    try:
        logging.info(f"Sending request to Ollama server with prompt: {prompt}")
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": model, "prompt": prompt},
            timeout=60
        )

        if response.status_code != 200:
            logging.error(f"Ollama server error - status code {response.status_code}")
            return f"Ollama server error: {response.text}"

        results = []
        for line in response.text.strip().split("\n"):
            try:
                line_data = json.loads(line)
                results.append(line_data.get("response", ""))
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse line: {line} - Error: {e}")
                continue

        final_response = "".join(results).strip()
        logging.info(f"Received response from Ollama server: {final_response}")
        return final_response

    except Exception as e:
        logging.error(f"Unexpected error while communicating with Ollama: {e}")
        return "An unexpected error occurred while communicating with Ollama."

#################################
# 查询数据库获取实际答案
#################################
def query_database(sql_query: str) -> str:
    """
    模拟数据库查询，实际代码需替换为对应数据库查询代码。
    """
    logging.info(f"Executing SQL: {sql_query}")
    # TODO: 替换为实际数据库查询逻辑
    return f"Query executed: {sql_query}"

#################################
# 核心函数：RAG 查询 + 调用 Ollama
#################################
def generate_response(question: str, possible_answer: str) -> dict:
    """
    RAG 查询的完整流程：
    1. 从 text-to-sql-index 获取最匹配的问题。
    2. 根据 table_id 从 table-to-sql-index 获取表的元数据。
    3. 构建 SQL 查询语句。
    4. 调用 Ollama 修正 SQL。
    5. 查询数据库获取最终答案。
    """
    try:
        # 查询问题对应的 Pinecone 数据
        pc = init_pinecone()
        query_vector = embedding_model.encode(question).tolist()
        results = query_pinecone(pc, TEXT_TO_SQL_INDEX_NAME, query_vector, top_k=TOP_K)

        if not results or not results.get("matches"):
            logging.warning("No matches found in Pinecone query results.")
            return {"sql": "", "answer": "No relevant data found in the database."}

        # 获取问题的最佳匹配
        best_match = results["matches"][0]
        table_id = best_match["metadata"].get("table_id", "unknown_table")
        logging.info(f"Processing table_id: {table_id}")

        # 查询表的元数据
        table_metadata = query_table_pinecone(table_id)
        if not table_metadata:
            return {"sql": "", "answer": f"Table {table_id} does not exist or has no metadata."}

        # 构建 SQL 查询语句
        query_data = json.loads(best_match["metadata"]["sql"])
        real_sql = build_real_sql(Query.from_dict(query_data), table_metadata)

        # 调用 Ollama 修正 SQL
        llama_prompt = (
            f"Question: {question}\n"
            f"Initial SQL: {real_sql}\n"
            f"Possible Answer: {possible_answer}\n"
            "Instruction: Modify the SQL query to ensure it aligns with the question and database structure. "
            "Return only the final SQL query."
        )
        final_sql = generate_response_with_ollama(llama_prompt)

        # 查询数据库获取答案
        actual_answer = query_database(final_sql)

        return {"sql": final_sql, "answer": actual_answer}

    except Exception as e:
        logging.error(f"Unexpected error in generate_response: {e}")
        return {"sql": "", "answer": "An unexpected error occurred while processing your request."}

#################################
# 示例调用
#################################
if __name__ == "__main__":
    question = "Tell me what the notes are for South Australia"
    possible_answer = ""
    response = generate_response(question, possible_answer)
    print(response)
