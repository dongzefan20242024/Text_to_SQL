import json
import logging
import sqlite3
import requests
import os
from sentence_transformers import SentenceTransformer
from app.utils.pinecone_client import init_pinecone, query_pinecone
from typing import Optional

#################################
# 配置区域
#################################
LOG_LEVEL = logging.INFO
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "../../data/train.db")
PINECONE_INDEX_NAME = "text-to-sql-index"
TOP_K = 1

# 日志配置
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

#################################
# 初始化向量模型
#################################
logging.info("Initializing embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
logging.info("Embedding model loaded.")

#################################
# 工具函数
#################################
def extract_sql_from_ollama_response(response_text: str) -> str:
    """从 Ollama 的分段响应中提取 SQL 查询"""
    try:
        sql_parts = []
        for line in response_text.split("\n"):
            try:
                json_obj = json.loads(line)
                if "response" in json_obj:
                    sql_parts.append(json_obj["response"])
            except json.JSONDecodeError:
                continue
        final_sql = "".join(sql_parts).strip()
        logging.info(f"Extracted SQL from Ollama: {final_sql}")
        return final_sql
    except Exception as e:
        logging.error(f"Error extracting SQL: {e}")
        return "An error occurred while processing Ollama response."

def clean_sql_query(sql: str) -> str:
    """清理 SQL 查询格式"""
    sql = sql.replace("SELECT ", "SELECT ").replace(" col ", " col").replace("col ", "col")
    sql = sql.replace("table _", "table_").replace("_ ", "_").replace(" _", "_").strip()
    if "WHERE" in sql:
        parts = sql.split("WHERE")
        if len(parts) == 2:
            condition = parts[1].strip()
            condition = condition.replace(" = ' ", "='").replace(" ' ", "'").replace(" '", "'").strip()
            sql = f"{parts[0].strip()} WHERE {condition}"
    sql = " ".join(sql.split())
    logging.info(f"Cleaned SQL query: {sql}")
    return sql

def get_table_headers(table_id: str) -> dict:
    """从数据库中获取表的列名"""
    table_name = f"table_{table_id.replace('-', '_')}"
    try:
        logging.info(f"Fetching headers for table: {table_name}")
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        headers = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return headers
    except Exception as e:
        logging.error(f"Error fetching table headers for {table_id}: {e}")
        return {}

def query_database(sql_query: str) -> str:
    """执行 SQL 查询并返回结果"""
    try:
        logging.info(f"Executing SQL: {sql_query}")
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        return "An error occurred while querying the database."

#################################
# Ollama 调用
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
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Ollama: {e}")
        return "An error occurred while calling Ollama."

#################################
# 核心函数
#################################
def generate_response(question: str, possible_answer: str, index_name: Optional[str] = None) -> dict:
    if index_name is None:
        index_name = PINECONE_INDEX_NAME
    try:
        # 查询 Pinecone 获取问题相关的元数据
        pc = init_pinecone()
        query_vector = embedding_model.encode(question).tolist()
        results = query_pinecone(pc, index_name, query_vector, top_k=TOP_K)

        if not results or not results.get("matches"):
            logging.warning("No matches found in Pinecone query results.")
            return {"sql": "", "answer": "No relevant data found in the database."}

        best_match = results["matches"][0]
        metadata = best_match.get("metadata", {})
        table_id = metadata.get("table_id", "")
        raw_sql = json.loads(metadata.get("sql", "{}"))
        headers = get_table_headers(table_id)

        if not headers:
            return {"sql": "", "answer": "Table metadata not found."}

        # 构造初始 SQL
        col_sel = headers.get(raw_sql["sel"], f"col{raw_sql['sel']}")
        col_conds = [
            f"{headers.get(cond[0], f'col{cond[0]}')} {['=', '>', '<'][cond[1]]} '{cond[2]}'"
            for cond in raw_sql["conds"]
        ]
        initial_sql = f"SELECT {col_sel} FROM table_{table_id.replace('-', '_')} WHERE {' AND '.join(col_conds)};"
        logging.info(f"Initial SQL: {initial_sql}")

        # 调用 Ollama
        llama_prompt = (
            f"Question: {question}\n"
            f"Initial SQL: {initial_sql}\n"
            f"Instruction: Modify the SQL query to ensure the following:\n"
            f"- The SQL syntax is valid and correct.\n"
            f"- Do not change the table name or column names unless necessary.\n"
            f"- Ensure all text comparisons (e.g., WHERE, AND, OR conditions) include 'COLLATE NOCASE' immediately after the value or column being compared to handle case-insensitivity.\n"
            f"- For example: col_name = 'value' COLLATE NOCASE or 'value' COLLATE NOCASE = col_name.\n"
            f"- Do not apply 'COLLATE NOCASE' to numerical or non-string comparisons.\n"
            f"- Ensure that the SQL is optimized and matches the structure of the database.\n"
            f"Return only the final SQL query."
        )

        ollama_response = generate_response_with_ollama(llama_prompt)
        final_sql = extract_sql_from_ollama_response(ollama_response)

        # 清理 SQL 查询
        cleaned_sql = clean_sql_query(final_sql)

        # 查询数据库
        actual_answer = query_database(cleaned_sql)
        return {"sql": cleaned_sql, "answer": actual_answer}
    except Exception as e:
        logging.error(f"Unexpected error in generate_response: {e}")
        return {"sql": "", "answer": "An unexpected error occurred while processing your request."}

#################################
# 主程序
#################################
if __name__ == "__main__":
    question = "Tell me what the notes are for South Australia"
    possible_answer = ""
    response = generate_response(question, possible_answer)
    print(response)
