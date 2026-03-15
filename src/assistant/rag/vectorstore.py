from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

VECTOR_DB_PATH = "db"


def create_vectorstore(documents):

    embeddings = OllamaEmbeddings(model="granite4:3b-h")

    vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    return vectorstore


def load_vectorstore():

    embeddings = OllamaEmbeddings(model="granite4:3b-h")

    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )