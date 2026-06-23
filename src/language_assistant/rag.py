from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from .citations import Citation, format_citation_list, format_context
from .config import AssistantConfig
from .vector_store import load_index


SYSTEM_PROMPT = """你是一个学术文档知识助手。你必须遵守：
1. 只依据给定的检索片段回答。
2. 每个关键事实后添加引用标签，例如 [C1]。
3. 如果片段无法支持答案，明确说“检索到的资料不足以回答”。
4. 不要编造论文、作者、页码、实验结果或引用。
"""


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    citations: list[Citation]

    def render(self) -> str:
        return f"{self.answer.strip()}\n\n引用：\n{format_citation_list(self.citations)}"


class RagAssistant:
    def __init__(self, config: AssistantConfig) -> None:
        self.config = config
        self.index = load_index(config.index_path, config.embedding_model)
        self.llm = ChatOpenAI(model=config.chat_model, temperature=config.temperature)

    def ask(self, question: str, top_k: int | None = None) -> RagAnswer:
        k = top_k or self.config.top_k
        docs = self.index.similarity_search(question, k=k)
        context, citations = format_context(docs)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"检索片段：\n{context}\n\n"
            f"问题：{question}\n\n"
            "请用中文给出简洁、可验证的回答。"
        )
        response = self.llm.invoke(prompt)
        return RagAnswer(question=question, answer=str(response.content), citations=citations)
