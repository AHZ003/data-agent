"""Load statistical best practice documents into the vector store."""

import os
from typing import List, Tuple

from rag.vector_store import VectorStore

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")


def load_knowledge_base() -> VectorStore:
    """Load all markdown documents into a ChromaDB vector store."""
    store = VectorStore()

    if store.count > 0:
        return store  # Already loaded

    documents: List[str] = []
    ids: List[str] = []
    metadatas: List[dict] = []

    if os.path.exists(DOCUMENTS_DIR):
        for filename in os.listdir(DOCUMENTS_DIR):
            if filename.endswith(".md"):
                filepath = os.path.join(DOCUMENTS_DIR, filename)
                with open(filepath, "r") as f:
                    content = f.read()

                # Split long documents into chunks
                chunks = _chunk_document(content, max_chars=1000)
                for i, chunk in enumerate(chunks):
                    doc_id = f"{filename}_{i}"
                    documents.append(chunk)
                    ids.append(doc_id)
                    metadatas.append({"source": filename, "chunk": i})

    if documents:
        store.add_documents(documents, ids, metadatas)

    return store


def _chunk_document(text: str, max_chars: int = 1000) -> List[str]:
    """Split a document into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]
