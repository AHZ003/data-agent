"""ChromaDB vector store for statistical knowledge RAG."""

import os
import chromadb
from typing import List

from config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR


class VectorStore:
    """ChromaDB-backed vector store for statistical knowledge."""

    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
        )

    def add_documents(self, documents: List[str], ids: List[str], metadatas: List[dict] = None):
        """Add documents to the collection."""
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )

    def query(self, query_text: str, n_results: int = 3) -> List[str]:
        """Query the collection and return relevant documents."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return results["documents"][0] if results["documents"] else []

    @property
    def count(self) -> int:
        return self.collection.count()
