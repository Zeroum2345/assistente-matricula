from assistant.rag.loader import load_documents
from assistant.rag.chunker import split_documents
from assistant.rag.vectorstore import create_vectorstore


def main():

    print("Carregando documentos...")
    docs = load_documents()

    print("Dividindo em chunks...")
    chunks = split_documents(docs)

    print(f"{len(chunks)} chunks criados")

    print("Criando banco vetorial...")
    create_vectorstore(chunks)

    print("Indexação finalizada!")


if __name__ == "__main__":
    main()