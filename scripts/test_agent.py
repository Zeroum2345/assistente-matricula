from assistant.agents.academic_agent import create_academic_agent
from langchain_core.messages import HumanMessage, SystemMessage


def main():

    agent = create_academic_agent()

    while True:

        question = input("\nPergunta: ")

        response = agent.invoke([
            SystemMessage(content="Você é um assistente acadêmico que ajuda alunos a planejar disciplinas."),
            HumanMessage(content=question)
        ])

        print("\nResposta:")
        print(response.content)


if __name__ == "__main__":
    main()