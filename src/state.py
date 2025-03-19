from typing_extensions import TypedDict, List
from langchain_core.documents import Document

class DeepSeekState(TypedDict):
    question: str              # Current user question
    context: List[Document]    # Current retrieved documents
    answer: str                # Current generated answer
    history: List[dict]        # List of previous {question, context, answer}
    needs_retrieval: bool      # Flag to determine if new retrieval is needed