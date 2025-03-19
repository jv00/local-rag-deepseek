from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams

from src.embeddings import MiniLMEmbeddings

# Configuration constants
DB_CONFIG = {
    "HOST": "localhost",
    "PORT": 6333,
    "COLLECTION_NAME": "deep_seek_storage",
    "VECTOR_SIZE": 384,  # Matches MiniLM embedding size; consider LLM context window
    "DISTANCE": Distance.COSINE
}

class QdrantDBManager:
    """Manages interactions with Qdrant vector database."""
    
    def __init__(self):
        """Initialize Qdrant client with configured host and port."""
        self.client = QdrantClient(host=DB_CONFIG["HOST"], port=DB_CONFIG["PORT"])
        self.embeddings = MiniLMEmbeddings()

    def _ensure_collection_exists(self) -> None:
        """
        Create the main database collection if it doesn't exist.
        
        Uses the configured collection name and vector parameters.
        """
        if not self.client.collection_exists(DB_CONFIG["COLLECTION_NAME"]):
            self.client.create_collection(
                collection_name=DB_CONFIG["COLLECTION_NAME"],
                vectors_config=VectorParams(
                    size=DB_CONFIG["VECTOR_SIZE"],
                    distance=DB_CONFIG["DISTANCE"]
                )
            )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using semantic and character-based splitting.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of split Document objects
        """
        # Semantic splitting for meaningful chunks
        semantic_splitter = SemanticChunker(self.embeddings)
        semantic_docs = semantic_splitter.split_documents(documents)
        
        # Further split into smaller chunks with overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400
        )
        return text_splitter.split_documents(semantic_docs)

    def generate_points(self, split_docs: List[Document]) -> List[PointStruct]:
        """
        Generate Qdrant points from split documents with embeddings.
        
        Args:
            split_docs: List of split Document objects
            
        Returns:
            List of PointStruct objects with vectors and payloads
        """
        # Generate embeddings for all document contents
        vectors = self.embeddings.embed_documents(
            [doc.page_content for doc in split_docs]
        )
        
        # Create points with IDs, vectors, and payloads
        return [
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "page_content": doc.page_content or "",
                    "metadata": doc.metadata or {}
                }
            )
            for i, (vector, doc) in enumerate(zip(vectors, split_docs))
        ]

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the Qdrant vector store.
        
        Args:
            documents: List of Document objects to process and store
        """
        # Ensure collection exists before adding documents
        self._ensure_collection_exists()
        
        # Split documents into manageable chunks
        split_docs = self.split_documents(documents)
        
        # Generate points with embeddings
        points = self.generate_points(split_docs)
        
        # Upload points to Qdrant
        self.client.upload_points(
            collection_name=DB_CONFIG["COLLECTION_NAME"],
            points=points
        )

    def get_vector_store(self) -> QdrantVectorStore:
        """
        Retrieve the Qdrant vector store instance.
        
        Returns:
            QdrantVectorStore configured with client and embeddings
        """
        return QdrantVectorStore(
            client=self.client,
            collection_name=DB_CONFIG["COLLECTION_NAME"],
            embedding=self.embeddings
        )

# Exposed functions for external use
def create_main_db_collection() -> None:
    """Create the main database collection if it doesn't exist."""
    manager = QdrantDBManager()
    manager._ensure_collection_exists()

def add_documents(documents: List[Document]) -> None:
    """Add documents to the Qdrant vector store."""
    manager = QdrantDBManager()
    manager.add_documents(documents)

def get_vector_store() -> QdrantVectorStore:
    """Get the configured Qdrant vector store instance."""
    manager = QdrantDBManager()
    return manager.get_vector_store()