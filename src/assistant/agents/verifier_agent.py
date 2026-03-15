def verifier_node(state):

    docs = state["documents"]
    answer = state["answer"]

    if len(docs) == 0:
        return {"verified": False}

    return {"verified": True}