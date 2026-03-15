from langchain_community.chat_models import ChatOllama
from .retriever import get_retriever


def ask_rag(question):

    retriever = get_retriever()

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatOllama(model="granite4:3b-h")

    prompt = f"""
Responda usando apenas as informações do contexto.

Contexto:
{context}

Pergunta:
{question}

Se não houver evidência suficiente no contexto, responda:
"Não encontrei evidências suficientes nos documentos."
"""

    response = llm.invoke(prompt)

    sources = [
        f"{doc.metadata.get('source')} (página {doc.metadata.get('page')})"
        for doc in docs
    ]

    return {
        "answer": response.content,
        "documents": docs,
        "sources": sources
    }