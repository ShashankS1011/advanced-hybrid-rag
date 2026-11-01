import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_BASE_DIR = Path(__file__).resolve().parent.parent
_env_path = _BASE_DIR / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=True)
else:
    load_dotenv(find_dotenv(), override=True)


@dataclass
class Settings:
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    CHROMA_DB_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "advanced_hybrid_rag"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 4
    DENSE_K: int = 12
    SPARSE_K: int = 12
    RERANK_TOP_N: int = 4
    DENSE_WEIGHT: float = 0.5
    SPARSE_WEIGHT: float = 0.5
    TEMPERATURE: float = 0.1
    RERANK_MODEL: str = "ms-marco-TinyBERT-L-2-v2"
    DEVICE: str = "cpu"
    SCORE_THRESHOLD: float = 0.0
    MAX_OUTPUT_TOKENS: int = 1024
    MAX_QUERY_LENGTH: int = 500
    MAX_SESSION_QUERIES: int = 25
    RATE_LIMIT_SECONDS: float = 2.5
    DAILY_GLOBAL_QUERY_LIMIT: int = 150
    ENABLE_QUERY_CACHE: bool = True
    ENABLE_ADAPTIVE_RETRIEVAL: bool = True

    def __post_init__(self) -> None:
        self.ENABLE_QUERY_CACHE = os.getenv("ENABLE_QUERY_CACHE", "true").lower() in ("true", "1", "yes")
        self.ENABLE_ADAPTIVE_RETRIEVAL = os.getenv("ENABLE_ADAPTIVE_RETRIEVAL", "true").lower() in ("true", "1", "yes")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", self.GROQ_API_KEY).strip()
        if not self.GROQ_API_KEY:
            try:
                import streamlit as st

                if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                    self.GROQ_API_KEY = str(st.secrets["GROQ_API_KEY"]).strip()
                    os.environ["GROQ_API_KEY"] = self.GROQ_API_KEY
            except Exception:
                pass
        self.GROQ_MODEL = os.getenv("GROQ_MODEL", self.GROQ_MODEL).strip()
        self.CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", self.CHROMA_DB_DIR)
        self.CHROMA_COLLECTION_NAME = os.getenv(
            "CHROMA_COLLECTION_NAME",
            self.CHROMA_COLLECTION_NAME,
        )
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", self.EMBEDDING_MODEL)

        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", self.CHUNK_SIZE))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", self.CHUNK_OVERLAP))
        self.TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", self.TOP_K_RETRIEVAL))
        self.DENSE_K = int(os.getenv("DENSE_K", self.DENSE_K))
        self.SPARSE_K = int(os.getenv("SPARSE_K", self.SPARSE_K))
        self.RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", self.RERANK_TOP_N))

        self.DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", self.DENSE_WEIGHT))
        self.SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", self.SPARSE_WEIGHT))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", self.TEMPERATURE))
        self.RERANK_MODEL = os.getenv("RERANK_MODEL", self.RERANK_MODEL).strip()
        self.DEVICE = os.getenv("DEVICE", self.DEVICE).strip()
        self.SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", self.SCORE_THRESHOLD))
        self.MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", self.MAX_OUTPUT_TOKENS))
        self.MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", self.MAX_QUERY_LENGTH))
        self.MAX_SESSION_QUERIES = int(os.getenv("MAX_SESSION_QUERIES", self.MAX_SESSION_QUERIES))
        self.RATE_LIMIT_SECONDS = float(os.getenv("RATE_LIMIT_SECONDS", self.RATE_LIMIT_SECONDS))
        self.DAILY_GLOBAL_QUERY_LIMIT = int(os.getenv("DAILY_GLOBAL_QUERY_LIMIT", self.DAILY_GLOBAL_QUERY_LIMIT))

        self._validate()

    def _validate(self) -> None:
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero.")
        if self.CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP must be zero or greater.")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.TOP_K_RETRIEVAL <= 0:
            raise ValueError("TOP_K_RETRIEVAL must be greater than zero.")
        if self.DENSE_K <= 0:
            raise ValueError("DENSE_K must be greater than zero.")
        if self.SPARSE_K <= 0:
            raise ValueError("SPARSE_K must be greater than zero.")
        if self.RERANK_TOP_N <= 0:
            raise ValueError("RERANK_TOP_N must be greater than zero.")
        if not (0.0 <= self.DENSE_WEIGHT <= 1.0):
            raise ValueError("DENSE_WEIGHT must be between 0.0 and 1.0.")
        if not (0.0 <= self.SPARSE_WEIGHT <= 1.0):
            raise ValueError("SPARSE_WEIGHT must be between 0.0 and 1.0.")
        if not (0.0 <= self.TEMPERATURE <= 2.0):
            raise ValueError("TEMPERATURE must be between 0.0 and 2.0.")
        if self.MAX_OUTPUT_TOKENS <= 0:
            raise ValueError("MAX_OUTPUT_TOKENS must be greater than zero.")
        if self.MAX_QUERY_LENGTH <= 0:
            raise ValueError("MAX_QUERY_LENGTH must be greater than zero.")
        if self.MAX_SESSION_QUERIES <= 0:
            raise ValueError("MAX_SESSION_QUERIES must be greater than zero.")
        if self.RATE_LIMIT_SECONDS < 0:
            raise ValueError("RATE_LIMIT_SECONDS cannot be negative.")
        if self.DAILY_GLOBAL_QUERY_LIMIT <= 0:
            raise ValueError("DAILY_GLOBAL_QUERY_LIMIT must be greater than zero.")

    def validate_runtime_configuration(self) -> None:
        if not self.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set in environment or .env file. "
                "For Streamlit Cloud, add GROQ_API_KEY to App Settings > Secrets."
            )
        if not self.GROQ_MODEL:
            raise ValueError("GROQ_MODEL is not set.")
        if not self.CHROMA_DB_DIR:
            raise ValueError("CHROMA_DB_DIR is not set.")


settings = Settings()
