from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate a response based on the question and documents in the state.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        Dict[str, Any]: A dictionary containing the generated response and the original question.
    """
    print("Generating response...")
    question = state["question"]
    documents = state["documents"]
    
    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "generation": generation, "question": question}