from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="granite4:3b-h",
    streaming=False
)

response = llm.invoke("Explique o que é machine learning em uma frase.")

print(response.content)