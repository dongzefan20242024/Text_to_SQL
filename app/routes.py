from flask import Blueprint, request, jsonify, render_template
from app.models.rag_model import generate_response
import logging
import whisper
# 定义一个 Blueprint 实例，命名为 main
main = Blueprint("main", __name__)

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@main.route("/", methods=["GET"])
def index():
    """
    渲染前端页面：
    只允许 GET 方法，用于返回 HTML 页面。
    """
    return render_template("index.html")

@main.route('/query', methods=['POST'])
def query():
    """
    接收来自前端的问题，调用 RAG 系统，并返回结果。
    """
    try:
        # 获取前端发送的数据
        data = request.json
        if not data or "question" not in data:
            logger.warning("No question provided in the request.")
            return jsonify({"error": "No question provided"}), 400

        question = data.get("question", "").strip()
        if not question:
            logger.warning("Received an empty question.")
            return jsonify({"error": "Question cannot be empty"}), 400

        # 记录收到的问题
        logger.info(f"Received question: {question}")

        # 调用 RAG 系统
        rag_result = generate_response(question, possible_answer="")  # 传入用户问题

        # 检查 RAG 系统返回结果
        if not rag_result or "answer" not in rag_result:
            logger.error("RAG system failed to provide a valid answer.")
            return jsonify({"error": "Failed to process the question"}), 500

        # 获取并返回回答
        answer = rag_result.get("answer", "No answer provided.")
        logger.info(f"RAG system response: {answer}")
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Error handling query: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500
