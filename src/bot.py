from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

from .store import VectorStore
from .config import get_settings
from .prompt import QUERY_TEMPLATE, RAG_TEMPLATE

settings = get_settings()
docstore = VectorStore.get_vector_database()

LLM = ChatOllama(
    model=settings.LLM,
    num_predict=settings.MAX_TOKENS_OUTPUT,
    temperature=settings.TEMPERATURE,
    top_k=settings.TOP_K,
)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=QUERY_TEMPLATE,
)

retriever = MultiQueryRetriever.from_llm(
    retriever=docstore.as_retriever(search_kwargs={"k": settings.K_RETRIEVED}),
    llm=LLM,
    prompt=QUERY_PROMPT,
)

template = RAG_TEMPLATE

PROMPT = ChatPromptTemplate.from_template(template=template)

CHAIN = (
    {"context": retriever, "question": RunnablePassthrough()}
    | PROMPT
    | LLM
    | StrOutputParser()
)


def answer(query: str):
    return CHAIN.invoke(query.strip())
