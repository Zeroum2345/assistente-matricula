from assistant.rag.retriever import get_retriever

def main():

    retriever = get_retriever()

    query = "pré requisitos de estruturas de dados"

    docs = retriever.invoke(query)

    print("\nConsulta:", query)
    print("\nDocumentos recuperados:\n")

    for i, doc in enumerate(docs):

        print(f"Resultado {i+1}")
        print(doc.page_content[:500])
        print("\nMetadata:", doc.metadata)
        print("\n-------------------\n")


if __name__ == "__main__":
    main()