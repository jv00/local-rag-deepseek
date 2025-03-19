# Local RAG with DeepSeek

This project implements a **Retrieval-Augmented Generation (RAG)** system running locally. It allows users to upload PDF files, store their contents in a vector database (Qdrant), and ask questions based on those documents. Responses, including reasoning, are generated by the DeepSeek R1 language model served via Ollama. The frontend is built with Streamlit, and the backend uses LangGraph for state management and conversation flow.

## Features

- **PDF Upload**: Upload PDF files through a Streamlit interface.
- **Vector Storage**: Extracts text from PDFs and stores embeddings in Qdrant for similarity-based retrieval.
- **Conversational AI**: Ask questions and receive structured responses with reasoning and answers, powered by DeepSeek R1 (1.5B parameters).
- **Chat History**: Persists conversation history, displaying reasoning and responses separately.
- **Follow-Up Detection**: Uses an LLM to determine if new document retrieval is needed for follow-up questions.
- **Dockerized Services**: Runs Qdrant and Ollama in Docker containers for easy setup.

## Architecture

- **Frontend**: Streamlit (`main.py`) for the user interface.
- **Backend**:
  - **State Management**: LangGraph (`graph.py`, `state.py`) orchestrates the RAG pipeline with memory persistence.
  - **Vector Database**: Qdrant (`vector_db.py`) stores document embeddings.
  - **LLM**: Ollama (`utils.py`) serves the DeepSeek R1 model via API.
  - **Utilities**: `utils.py` handles PDF extraction, response parsing, and Ollama invocation.
  - **Prompts**: `prompts.py` defines templates for retrieval, generation, and summarization.
- **Services**: Docker Compose (`docker-compose.yml`) manages Qdrant and Ollama.

## Prerequisites

- **Docker**: Installed and running (with Docker Compose).
- **Python**: Version 3.8+ for the Streamlit app.
- **Git**: Optional, for cloning the repository.

## Services (Docker Compose)

The `docker-compose.yml` defines two services:

### 1. Qdrant

- **Image**: `qdrant/qdrant:latest`
- **Purpose**: Vector database for storing and retrieving document embeddings.
- **Ports**:
  - `6333`: REST API and gRPC endpoint.
  - `6334`: Web UI (optional, for debugging).
- **Volumes**: Persists data in `qdrant_data` at `/qdrant/storage`.
- **Container Name**: `qdrant`
- **Restart Policy**: `always`.

### 2. Ollama

- **Image**: `ollama/ollama:latest`
- **Purpose**: Serves the DeepSeek R1 model via an HTTP API.
- **Ports**:
  - `11434`: API endpoint.
- **Volumes**: Persists model data in `ollama_data` at `/root/.ollama`.
- **Container Name**: `ollama`
- **Entry Command**: Starts the server, pulls `deepseek-r1:1.5b`, and keeps running.
- **Restart Policy**: `always`.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/local-rag-deepseek.git
cd local-rag-deepseek

python -m venv venv

pip install -r requirements.txt

docker-compose up -d

streamlit run main.py