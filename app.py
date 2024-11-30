from flask import Flask, render_template, request
import json
from utils import retrieve_context, generate_sql

# 初始化 Flask 应用
app = Flask(__name__)

# 加载数据库架构
def load_schema():
    with open('data/schema.json') as f:
        return json.load(f)

db_schema = load_schema()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query']
    context = retrieve_context(user_query, db_schema)
    sql_query = generate_sql(user_query, context)
    return f"Generated SQL Query: {sql_query}"

if __name__ == "__main__":
    app.run(debug=True)
