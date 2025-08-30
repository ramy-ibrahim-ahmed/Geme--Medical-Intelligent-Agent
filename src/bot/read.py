import cohere
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

from ..store import VectorStore
from ..config import get_settings
from .state import State, QWEN
from .prompt import retrival_prompt


settings = get_settings()
co = cohere.ClientV2(api_key=settings.COHERE_API_KEY)


def Read(state: State) -> State:
    print("__read__")
    query = state["messages"][-1].content

    enhance_retrival_prompt = PromptTemplate(
        input_variables=["question"],
        template=retrival_prompt,
    )

    docstore = VectorStore.get_vector_database()
    retriever = MultiQueryRetriever.from_llm(
        retriever=docstore.as_retriever(search_kwargs={"k": settings.K_RETRIEVED}),
        llm=QWEN,
        prompt=enhance_retrival_prompt,
    )

    documents = retriever.invoke(input=query)
    documents_2_rerank = [doc.page_content for doc in documents]

    reranked = co.rerank(
        model=settings.RERANKER_MODEL,
        query=query,
        documents=documents_2_rerank,
        top_n=settings.NUM_RERANKED,
        return_documents=True,
    )

    reranked_documents = "\n\n".join(
        [
            f"Document {i + 1}: {doc.document.text}"
            for i, doc in enumerate(reranked.results)
        ]
    )
    return {"context": reranked_documents}
