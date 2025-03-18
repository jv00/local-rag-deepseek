import re
from typing import List, Optional, Dict, Any, Type
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from ollama import chat
from pydantic import BaseModel

from src.database import add_documents

class ParseOutputResult(BaseModel):
    """Pydantic model for structured parse output."""
    reasoning: str
    response: str

def parse_output(text: str) -> Dict[str, str]:
    """
    Parse text containing <think> tags to extract reasoning and response.

    Args:
        text: Raw text containing <think>reasoning</think> and response.

    Returns:
        Dictionary with 'reasoning' and 'response' keys.

    Raises:
        AttributeError: If regex pattern doesn't match expected format.
    """
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    output_match = re.search(r'</think>\s*(.*?)$', text, re.DOTALL)

    reasoning = think_match.group(1).strip() if think_match else ""
    response = output_match.group(1).strip() if output_match else text.strip()

    return {"reasoning": reasoning, "response": response}

def extract_text_from_pdf(pdf_file: Any) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_file: Uploaded file object (e.g., from Streamlit file_uploader).

    Returns:
        Concatenated text from all pages in the PDF.

    Raises:
        PyPDF2.errors.PdfReadError: If the PDF file is corrupted or unreadable.
    """
    reader = PdfReader(pdf_file)
    text_pages = [
        page.extract_text() for page in reader.pages if page.extract_text() is not None
    ]
    return "\n".join(text_pages)

def upload_files_to_db(uploaded_files: List[Any]) -> bool:
    """
    Convert uploaded PDF files to Documents and store them in the vector database.

    Args:
        uploaded_files: List of uploaded file objects (e.g., from Streamlit file_uploader).

    Returns:
        True if documents were successfully uploaded, False otherwise.
    """
    documents = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_text = extract_text_from_pdf(uploaded_file)

        if file_text:
            doc = Document(
                page_content=file_text,
                metadata={"file_name": file_name}
            )
            documents.append(doc)

    if documents:
        add_documents(documents)
        return True
    return False

def invoke_ollama(
    model: str,
    system_prompt: str,
    user_prompt: str,
    output_format: Optional[Type[BaseModel]] = None
) -> Any:
    """
    Invoke the Ollama model with system and user prompts.

    Args:
        model: Name of the Ollama model to use (e.g., "deepseek-r1:1.5b").
        system_prompt: System-level instruction for the model.
        user_prompt: User query or input.
        output_format: Optional Pydantic model for structured JSON output.

    Returns:
        Model response as a string or parsed Pydantic model if output_format is provided.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = chat(
        messages=messages,
        model=model,
        format=output_format.model_json_schema() if output_format else None
    )

    content = response["message"]["content"]  # Adjusted for Ollama response structure
    if output_format:
        return output_format.model_validate_json(content)
    return content