from dataclasses import dataclass
import os

@dataclass
class Config:
    google_api_key: str = os.getenv('GOOGLE_API_KEY', '')
    debug: bool = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    host: str = os.getenv('HOST', '0.0.0.0')
    port: int = int(os.getenv('PORT', 5000))
    allowed_origins: str = os.getenv('ALLOWED_ORIGINS', '*')
    rate_limit: str = os.getenv('RATE_LIMIT', '50/hour')
    max_history_length: int = int(os.getenv('MAX_HISTORY_LENGTH', 5))

    def validate(self) -> None:
        if not self.google_api_key:
            raise ValueError("Missing required environment variable: GOOGLE_API_KEY")