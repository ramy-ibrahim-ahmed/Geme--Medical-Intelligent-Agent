from langgraph.graph import START, END, StateGraph

from .state import State
from .read import Read
from .chatbot import Chatbot
from .ocr import OCR
from .search import Search
from .router import router


workflow = StateGraph(State)
workflow.add_node("Geme", Chatbot)
workflow.add_node("OCR", OCR)
workflow.add_node("Read", Read)
workflow.add_node("Search", Search)

workflow.add_conditional_edges(
    START, router, {"Read": "Read", "Geme": "Geme", "OCR": "OCR"}
)
workflow.add_edge("OCR", "Search")
workflow.add_edge("Search", "Geme")
workflow.add_edge("Read", "Geme")
workflow.add_edge("Geme", END)


def get_geme():
    return workflow.compile()
