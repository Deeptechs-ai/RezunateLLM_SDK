"""Configuration settings for the landing page application."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    database_url: str
    debug: bool = False
    app_name: str = "LLM-Router"
    openai_api_key: str = ""
    google_api_key: str = ""
    demo_rate_limit: int

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
