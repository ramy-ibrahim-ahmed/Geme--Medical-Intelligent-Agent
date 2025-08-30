from typing import Literal
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

from .state import State, QWEN
from .prompt import classify_prompt


class Classes(BaseModel):
    category: Literal["RAG", "GENERAL", "OUTSIDE"] = Field(
        ..., description="The classification category of the question."
    )


def router(state: State):
    if state.get("image", ""):
        return "OCR"

    question = state["messages"][-1]
    prompt = PromptTemplate.from_template(template=classify_prompt)
    message = prompt.format(question=question)

    classifier = QWEN.with_structured_output(Classes)
    semantic: Classes = classifier.invoke(message)

    if semantic.category == "RAG":
        return "Read"

    return "Geme"
