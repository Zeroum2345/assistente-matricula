from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path


def load_documents(data_path="data/raw"):

    documents = []

    for pdf in Path(data_path).glob("*.pdf"):

        print("Carregando:", pdf)

        loader = PyPDFLoader(str(pdf))
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = pdf.name

        documents.extend(docs)

    print("Total documentos carregados:", len(documents))

    return documents