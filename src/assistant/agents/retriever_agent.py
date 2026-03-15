from assistant.rag.retriever import get_retriever

def retriever_node(state):

    question = state["question"]

    retriever = get_retriever()

    docs = retriever.invoke(question)

    return {
        "documents": docs 
    }