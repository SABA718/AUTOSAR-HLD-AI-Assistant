# Configuration and Retrieval Parameters

**Source of truth:** [config/settings.py](../config/settings.py), [coordinator](../backend/coordinator.py), [retriever](../core/retrieval/retriever.py), [prompt builder](../core/rag/prompt_builder.py), [evaluator](../core/evaluation/evaluator.py).

| Parameter | Actual implementation/default |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` |
| Embedding device | `cpu` |
| Gemini model | `gemini-3-flash-preview` Settings default vs `gemini-2.5-flash` in .env.example |
| LLM default | `mock`, can select `gemini` |
| Vector path | `./data/chroma_db` |
| SQLite path | `./data/autosar_assistant.db` |
| Retriever `top_k` | 5 default |
| Retriever similarity threshold | `None` default (no filter until explicitly passed) |
| Coordinator query threshold | `None` default |
| Golden evaluator threshold | 0.3 function default; application differs |
| Prompt context | 12,000 characters max by default (approximate token budget) |
| Q&A temperature | 0.0 supplied by coordinator |
| API / UI | 127.0.0.1:8000 / localhost:8501 |

**Distinctions:** Top-k is a candidate count, not relevance confidence. Similarity is not a calibrated probability of correctness. The evaluator's keyword-match ratio is not a semantic correctness score. In `core/evaluation/evaluator.py`, the relevance-threshold argument is treated differently for answerable/unanswerable items; inspect its implementation before quoting system-wide results.

Do not upload actual `.env` or runtime credentials. Verify the model and retrieval settings displayed during each real demonstration.
