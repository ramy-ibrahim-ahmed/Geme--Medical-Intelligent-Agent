import uuid
from langchain_core.messages import SystemMessage, ToolMessage

from .state import State, GEMMA
from .prompt import persona_prompt, return_prompt
from .prompt import chat_general_prompt, chat_read_prompt, chat_ocr_prompt


def Chatbot(state: State) -> State:
    messages = state.get("messages", "")
    context = state.get("context", "")
    search = state.get("search", "")

    if search:
        print("__chat_ocr__")
        system_ocr_prompt = persona_prompt + chat_ocr_prompt + return_prompt
        system_ocr_message = SystemMessage(content=system_ocr_prompt)
        tool_message = ToolMessage(content=search, tool_call_id=str(uuid.uuid4()))

        res = GEMMA.invoke([system_ocr_message] + messages + [tool_message])
        return {"messages": [res], "search": ""}

    if context:
        print("__chat_rag__")
        system_rag_prompt = persona_prompt + chat_read_prompt + return_prompt
        system_rag_message = SystemMessage(content=system_rag_prompt)
        tool_message = ToolMessage(content=context, tool_call_id=str(uuid.uuid4()))

        res = GEMMA.invoke([system_rag_message] + messages + [tool_message])
        return {"messages": [res], "context": ""}

    print("__chat_general__")
    system_general_prompt = persona_prompt + chat_general_prompt + return_prompt
    system_general_message = SystemMessage(content=system_general_prompt)

    res = GEMMA.invoke([system_general_message] + messages)
    return {"messages": [res]}
