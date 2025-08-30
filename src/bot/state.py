from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState

from ..config import get_settings

settings = get_settings()

QWEN = ChatOllama(
    model=settings.LLM,
    num_predict=settings.MAX_TOKENS_OUTPUT,
    temperature=settings.TEMPERATURE,
    top_k=settings.TOP_K,
)

GEMMA = ChatOllama(
    model="gemma3:4b",
    temperature=0.0,
)


class State(MessagesState):
    context: str
    image: str
    transcription: str
    search: str
