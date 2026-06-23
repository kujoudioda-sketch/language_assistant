from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def build_embeddings(model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model)


def build_faiss_index(documents: list[Document], embedding_model: str) -> FAISS:
    embeddings = build_embeddings(embedding_model)
    return FAISS.from_documents(documents, embeddings)


def save_index(index: FAISS, path: str | Path) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    index.save_local(str(target))


def load_index(path: str | Path, embedding_model: str) -> FAISS:
    embeddings = build_embeddings(embedding_model)
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
