from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str

    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    K_RETRIEVED: int

    EMBEDDING_MODEL: str
    EMBEDDING_SIZE: int

    COHERE_API_KEY: str
    RERANKER_MODEL: str
    NUM_RERANKED: int

    LLM: str
    MAX_TOKENS_OUTPUT: int
    TEMPERATURE: float
    TOP_K: int

    OLLAMA_URL: str
    VISION_MODEL: str

    TAVILY_API_KEY: str

    class Config:
        env_file = r".env"


def get_settings():
    return Settings()
