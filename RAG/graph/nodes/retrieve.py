from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents based on the question in the state.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        Dict[str, Any]: A dictionary containing the retrieved documents and the original question.
    """
    print("Retrieving documents...")
    question = state["question"]
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}