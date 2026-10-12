import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


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

    def __post_init__(self) -> None:
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", self.GROQ_API_KEY).strip()
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

    def validate_runtime_configuration(self) -> None:
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
        if not self.GROQ_MODEL:
            raise ValueError("GROQ_MODEL is not set.")
        if not self.CHROMA_DB_DIR:
            raise ValueError("CHROMA_DB_DIR is not set.")


settings = Settings()