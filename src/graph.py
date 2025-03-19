from typing import List
from langgraph.graph import START, END, StateGraph
from langchain_ollama.llms import OllamaLLM
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document

from src.utils import parse_output
from src.database import get_vector_store
from src.prompts import basic_prompt, summary_prompt, retrieval_prompt
from src.state import DeepSeekState

# Ollama API endpoint (adjust if your Docker port/host differs)
OLLAMA_BASE_URL = "http://localhost:11434"

def analyze_retrieval_need(state: DeepSeekState) -> DeepSeekState:
    """
    Use an LLM to determine if new document retrieval is needed based on history and current question.
    """
    print("Analyzing retrieval need with LLM")
    current_question = state["question"]
    history = state.get("history", [])

    if not history:
        return {"needs_retrieval": True}

    history_text = "\n".join(
        [f"Q: {entry['question']}\nA: {entry['answer']}" for entry in history]
    )

    model = OllamaLLM(model="deepseek-r1:1.5b", base_url=OLLAMA_BASE_URL)
    messages = retrieval_prompt.invoke({
        "current_question": current_question,
        "history": history_text
    })
    response = model.invoke(messages).strip().upper()
    needs_retrieval = response == "YES"
    print(f"LLM decision: {'Retrieve' if needs_retrieval else 'Reuse context'}")

    if not needs_retrieval and history[-1]["context"]:
        return {"needs_retrieval": False, "context": history[-1]["context"]}
    return {"needs_retrieval": True}

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
    
    model = OllamaLLM(model="deepseek-r1:1.5b", base_url=OLLAMA_BASE_URL)
    messages = summary_prompt.invoke({"text": past_contexts})
    summary = model.invoke(messages)
    return {"summary": parse_output(summary)}

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

    model = OllamaLLM(model="deepseek-r1:1.5b", base_url=OLLAMA_BASE_URL)
    response = model.invoke(messages)
    answer = parse_output(response)
    
    new_history = state.get("history", []) + [{
        "question": state["question"],
        "context": state["context"],
        "answer": answer
    }]
    
    return {"answer": answer, "history": new_history}

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