from typing import List
from langgraph.graph import START, END, StateGraph
from langchain_ollama.llms import OllamaLLM
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document

from utils import parse_output
from database import get_vector_store
from prompts import basic_prompt, summary_prompt, retrieval_prompt  # Add retrieval_prompt
from state import DeepSeekState

# --- Modified Analyze Retrieval Need Node ---
def analyze_retrieval_need(state: DeepSeekState) -> DeepSeekState:
    """
    Use an LLM to determine if new document retrieval is needed based on history and current question.
    """
    print("Analyzing retrieval need with LLM")
    current_question = state["question"]
    history = state.get("history", [])

    if not history:
        # First question, always retrieve
        return {"needs_retrieval": True}

    # Format history for the prompt
    history_text = "\n".join(
        [f"Q: {entry['question']}\nA: {entry['answer']}" for entry in history]
    )

    # Ask LLM if retrieval is needed
    model = OllamaLLM(model="deepseek-r1:1.5b")
    messages = retrieval_prompt.invoke({
        "current_question": current_question,
        "history": history_text
    })
    response = model.invoke(messages).strip().upper()

    # Parse LLM response
    needs_retrieval = response == "YES"
    print(f"LLM decision: {'Retrieve' if needs_retrieval else 'Reuse context'}")

    if not needs_retrieval and history[-1]["context"]:
        # Reuse last context if available
        return {"needs_retrieval": False, "context": history[-1]["context"]}
    return {"needs_retrieval": True}

# --- Retrieve Node (Unchanged) ---
def retrieve(state: DeepSeekState) -> DeepSeekState:
    """
    Retrieve documents if needed, otherwise use existing context.
    """
    if not state.get("needs_retrieval", True):
        print("Skipping retrieval, using existing context")
        return state

    print("Retrieving new documents")
    query = state["question"]
    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    documents = retriever.invoke(query)
    return {"context": documents}

# --- Summarize History Node (Unchanged) ---
def summarize_history(state: DeepSeekState) -> DeepSeekState:
    """
    Summarize previous context and answers for use in generation.
    """
    print("Summarizing history")
    history = state.get("history", [])
    if not history:
        return {"summary": ""}

    past_contexts = "\n".join(
        [f"Q: {entry['question']}\nA: {entry['answer']}" for entry in history]
    )
    
    model = OllamaLLM(model="deepseek-r1:1.5b")
    messages = summary_prompt.invoke({"text": past_contexts})
    summary = model.invoke(messages)
    return {"summary": parse_output(summary)}

# --- Generate Node (Unchanged) ---
def generate(state: DeepSeekState) -> DeepSeekState:
    """
    Generate a response using current context and summarized history.
    """
    print("Generating Response")
    docs_contents = "\n\n".join([doc.page_content for doc in state["context"]])
    summary = state.get("summary", "")
    
    messages = basic_prompt.invoke({
        "question": state["question"],
        "context": f"Previous Summary: {summary}\n\nCurrent Context: {docs_contents}"
    })

    model = OllamaLLM(model="deepseek-r1:1.5b")
    response = model.invoke(messages)
    answer = parse_output(response)
    
    new_history = state.get("history", []) + [{
        "question": state["question"],
        "context": state["context"],
        "answer": answer
    }]
    
    return {"answer": answer, "history": new_history}

# --- Build the Graph (Unchanged) ---
builder = StateGraph(DeepSeekState)

builder.add_node("analyze_retrieval", analyze_retrieval_need)
builder.add_node("retrieve", retrieve)
builder.add_node("summarize_history", summarize_history)
builder.add_node("generate", generate)

builder.add_edge(START, "analyze_retrieval")
builder.add_edge("analyze_retrieval", "retrieve")
builder.add_edge("retrieve", "summarize_history")
builder.add_edge("summarize_history", "generate")
builder.add_edge("generate", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)