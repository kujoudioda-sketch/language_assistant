from __future__ import annotations

from .config import AssistantConfig
from .documents import chunk_documents, load_documents
from .vector_store import build_faiss_index, save_index


def ingest(input_path: str, index_path: str, config: AssistantConfig) -> int:
    documents = load_documents(input_path)
    chunks = chunk_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    index = build_faiss_index(chunks, config.embedding_model)
    save_index(index, index_path)
    return len(chunks)
