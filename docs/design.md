# 设计说明

## 目标

本项目实现一个可复用的学术文档知识助手。它不追求“自由发挥”的聊天，而强调回答必须可追溯到原文证据。

## RAG 流程

1. 文档解析：PDF 使用 `PyPDFLoader`，Markdown/TXT 使用 `TextLoader`。
2. 文本分块：使用 `RecursiveCharacterTextSplitter`，保留 chunk overlap，降低跨段落信息丢失。
3. 向量化：使用 OpenAI Embeddings。
4. 检索：使用 FAISS 本地向量索引。
5. 生成：将检索片段格式化为 `[C1]`、`[C2]` 证据块，提示模型只依据证据回答。
6. 引用追踪：输出答案和引用列表，引用保留 `source`、`page`、`chunk_id`。

## 降低幻觉策略

- Prompt 明确禁止编造未检索到的信息。
- 每个关键事实必须带引用标签。
- 当上下文不足时要求模型直接说明无法回答。
- 评估阶段统计引用覆盖率，并可启用 LLM 忠实度 judge。
