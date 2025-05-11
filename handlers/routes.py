from flask import render_template, request, jsonify
from modules import modules
from modules.config import Config
from modules.ai_provider import HomeworkAI
from modules.session_manager import SessionManager
from modules.rate_limiter import limiter
from uuid import uuid4
import structlog

logger = structlog.get_logger(__name__)

def configure_routes(app):
    config = Config()
    config.validate()

    session_manager = SessionManager(config)
    homework_ai = HomeworkAI(config, session_manager)

    @app.route("/")
    def index():
        hello = modules.hello()
        content = modules.content()
        return render_template("index.html", hello=hello, content=content)

    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'request_id': str(uuid4())
        })

    @app.route("/api/generate_answer", methods=["POST"])
    @limiter.limit("50/hour")
    def generate_answer():
        request_id = str(uuid4())
        logger.info("Processing generate_answer request", request_id=request_id)

        data = request.get_json(silent=True)
        if not data:
            logger.warning("Invalid request data", request_id=request_id)
            return {
                'error': 'Invalid request data',
                'request_id': request_id
            }, 400

        question = data.get('question')
        session_id = data.get('session_id')

        if not question:
            logger.warning("No question provided", request_id=request_id)
            return {
                'error': 'No question provided',
                'request_id': request_id
            }, 400

        if not session_id or not session_manager.session_exists(session_id):
            session_id = homework_ai.start_session()
            logger.info("Created new session", session_id=session_id, request_id=request_id)

        response = homework_ai.generate_response(session_id, question)
        logger.info("message sent by AI", response=response, request_id=request_id)
        return response

    @app.route("/api/chat_history/<string:session_id>", methods=["GET"])
    def chat_history(session_id):
        request_id = str(uuid4())
        logger.info("Fetching chat history", request_id=request_id, session_id=session_id)

        if not session_manager.session_exists(session_id):
            logger.warning("Invalid session ID", request_id=request_id, session_id=session_id)
            return {
                'error': 'Invalid session ID',
                'request_id': request_id
            }, 404

        history = session_manager.get_all_chats(session_id)
        return {
            'session_id': session_id,
            'history': history,
            'request_id': request_id
        }