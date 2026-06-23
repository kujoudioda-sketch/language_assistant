from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.json import JSON

from .config import load_config, merge_config
from .evaluation import evaluate
from .ingest import ingest as ingest_documents
from .rag import RagAssistant

app = typer.Typer(help="Academic document RAG assistant.")
console = Console()


@app.callback()
def bootstrap() -> None:
    load_dotenv()


@app.command()
def ingest(
    input: str = typer.Option(..., "--input", "-i", help="Document file or directory."),
    index: str = typer.Option("storage/faiss", "--index", help="FAISS index directory."),
    config_path: str = typer.Option("configs/default.yaml", "--config", help="YAML config path."),
) -> None:
    config = merge_config(load_config(config_path), index_path=index)
    count = ingest_documents(input, index, config)
    console.print(f"[green]Indexed {count} chunks into {Path(index).as_posix()}[/green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from indexed documents."),
    index: str = typer.Option("storage/faiss", "--index", help="FAISS index directory."),
    config_path: str = typer.Option("configs/default.yaml", "--config", help="YAML config path."),
    top_k: int | None = typer.Option(None, "--top-k", help="Override retrieval depth."),
) -> None:
    config = merge_config(load_config(config_path), index_path=index, top_k=top_k)
    assistant = RagAssistant(config)
    answer = assistant.ask(question, top_k=top_k)
    console.print(answer.render())


@app.command("evaluate")
def evaluate_cmd(
    benchmark_file: str = typer.Option(..., "--benchmark", "-b", help="JSONL benchmark file."),
    index: str = typer.Option("storage/faiss", "--index", help="FAISS index directory."),
    config_path: str = typer.Option("configs/default.yaml", "--config", help="YAML config path."),
) -> None:
    config = merge_config(load_config(config_path), index_path=index)
    result = evaluate(config, benchmark_file)
    console.print(JSON.from_data(result.to_dict()))


if __name__ == "__main__":
    app()
