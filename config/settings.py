from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown fields (e.g. commented-out remote DB)
    )

    # LLM API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # PostgreSQL — individual components from .env
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "als-rag"
    postgres_user: str = "superadmin"
    postgres_password: str = "admin"

    # Embedding model
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384

    # Paths (relative to project root)
    data_dir: str = "data"
    results_dir: str = "results"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def async_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
