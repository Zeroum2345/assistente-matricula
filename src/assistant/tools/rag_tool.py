from langchain.tools import tool
from assistant.rag.rag_chain import ask_rag


@tool
def search_academic_documents(question: str) -> str:
    """
    Busca informações em documentos acadêmicos como ementas,
    planos de ensino e guias da universidade.
    """

    result = ask_rag(question)

    answer = result["answer"]
    sources = result["sources"]

    sources_text = "\n".join(sources)

    return f"""
Resposta:
{answer}

Fontes:
{sources_text}
"""