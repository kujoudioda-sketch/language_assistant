# 评估说明

## 基准格式

基准文件为 JSONL，每行一个问题：

```json
{"question":"...","expected_sources":["paper.md"],"reference_answer":"..."}
```

## 指标

- `recall_at_k`：top-k 检索结果是否命中任一期望来源。
- `mrr`：第一个命中来源的倒数排名均值。
- `citation_coverage`：回答中使用的引用标签数量占可用引用数量的比例。
- `faithfulness`：可选 LLM judge，判断回答是否被检索上下文支持。

## 运行

```bash
python scripts/rag_cli.py evaluate --benchmark data/benchmarks/sample_qa.jsonl --index storage/faiss
```
