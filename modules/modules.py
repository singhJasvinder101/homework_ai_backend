from .config import Config
from .ai_provider import HomeworkAI
from .session_manager import SessionManager
from .logger_config import setup_logging

def hello():
    return "Hello, World!"

def content():
    return '''
            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
            Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
            Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
            Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
            '''

def init_homework_ai():
    setup_logging()
    config = Config()
    config.validate()
    session_manager = SessionManager(config)
    homework_ai = HomeworkAI(config, session_manager)
    return homework_ai, session_manager

def start_session(homework_ai):
    return homework_ai.start_session()

def generate_response(homework_ai, session_manager, session_id, question):
    if not session_manager.session_exists(session_id):
        session_id = homework_ai.start_session()
    return homework_ai.generate_response(session_id, question), session_id

def get_chat_history(session_manager, session_id):
    return session_manager.get_all_chats(session_id)