from langchain_ollama import ChatOllama

from assistant.tools.rag_tool import search_academic_documents
from assistant.tools.eureca_api_tool import get_course_prerequisites


def create_academic_agent():

    llm = ChatOllama(
        model="granite4:3b-h",
        temperature=0,
        streaming=False
    )

    tools = [
        search_academic_documents,
        get_course_prerequisites
    ]

    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools