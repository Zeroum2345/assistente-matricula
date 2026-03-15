from assistant.graph.graph import build_graph


def main():

    graph = build_graph()

    while True:

        question = input("\nPergunta: ")

        result = graph.invoke({
            "question": question
        })

        print("\nResposta:")
        print(result["answer"])


if __name__ == "__main__":
    main()