from __future__ import annotations

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def iter_document_paths(input_path: str | Path) -> list[Path]:
    root = Path(input_path)
    if root.is_file():
        paths = [root]
    else:
        paths = [path for path in root.rglob("*") if path.is_file()]
    return sorted(path for path in paths if path.suffix.lower() in SUPPORTED_EXTENSIONS)


def load_documents(input_path: str | Path) -> list[Document]:
    documents: list[Document] = []
    for path in iter_document_paths(input_path):
        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            loaded = TextLoader(str(path), encoding="utf-8").load()
        for doc in loaded:
            doc.metadata["source"] = str(path.as_posix())
            documents.append(doc)
    if not documents:
        raise ValueError(f"No supported documents found under {input_path}")
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 140,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        source = Path(str(chunk.metadata.get("source", "doc"))).stem
        page = chunk.metadata.get("page")
        page_part = f"p{page + 1}" if isinstance(page, int) else "p0"
        chunk.metadata["chunk_id"] = f"{source}-{page_part}-{idx}"
    return chunks
