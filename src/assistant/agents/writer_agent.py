from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="granite4:3b-h",
    temperature=0,
    streaming=False
)


def writer_node(state):

    docs = state["documents"]
    question = state["question"]

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Use os documentos abaixo para responder.

Pergunta:
{question}

Documentos:
{context}

Responda citando as fontes.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }