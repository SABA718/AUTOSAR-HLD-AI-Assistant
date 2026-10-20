# Code — Existing executable source

To avoid two diverging application copies, the actual code remains at the **repository root**. This folder is a direct module index satisfying the academic folder structure.

| Domain | Source |
|---|---|
| PDF and cleaning | [core/ingestion](../core/ingestion) |
| Structural chunking | [core/chunking](../core/chunking) |
| Sentence embeddings | [core/embeddings](../core/embeddings) |
| ChromaDB | [core/vector_store](../core/vector_store) |
| Retrieval | [core/retrieval](../core/retrieval) |
| Prompt and citations | [core/rag](../core/rag) |
| Gemini and MockLLM | [core/llm](../core/llm) |
| SWC/Port/Interface | [core/extraction](../core/extraction) |
| Evaluation | [core/evaluation](../core/evaluation) |
| Coordinator and API | [backend](../backend) |
| Streamlit | [frontend](../frontend) |
| SQLite | [storage](../storage) |
| Settings | [config/settings.py](../config/settings.py) |
| Tests | [tests](../tests) |
| Dependencies | [requirements.txt](../requirements.txt) |

Use the repository root as working directory when installing, testing and starting the application. This is a code index, **not** a separately copied source tree.
