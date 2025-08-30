import os
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from .state import State, QWEN
from ..config import get_settings

settings = get_settings()
os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
tavily = TavilySearchResults(max_results=3)


class SearchQuery(BaseModel):
    query: str = Field(
        description="A detailed and specific search query about medicine. "
        "Include all relevant keywords, context, and parameters to ensure the search returns precise and accurate information."
    )


def search_by_tavily(query):
    search_docs = tavily.invoke(query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    print(len(search_docs))
    return formatted_search_docs


def Search(state: State) -> State:
    print("__search__")
    transcription = state.get("transcription", "")
    user_message = state["messages"][-1]

    query_prompt = f"Based on the following user question:\n\n{user_message}\n\nAnd the following context:\n\n{transcription}\n\n"
    "Generate a Web Search Query to help on answering the question with the following instruction:\n\n"
    "- Expanded search query by adding additional context and details to get better results."

    query_llm = QWEN.with_structured_output(SearchQuery)
    search_query: SearchQuery = query_llm.invoke(query_prompt).query
    print(f"__{search_query}__")

    res = search_by_tavily(query=search_query)
    return {"search": res, "transcription": ""}
