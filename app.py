import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from modules.config import Config
from modules.logger_config import setup_logging
from modules.rate_limiter import limiter
from handlers.routes import configure_routes

load_dotenv()

config = Config()
config.validate()

app = Flask(__name__)

CORS(app, resources={r"*": {"origins": config.allowed_origins}})

limiter.init_app(app)

configure_routes(app)

if __name__ == "__main__":
    app.run(debug=True)
