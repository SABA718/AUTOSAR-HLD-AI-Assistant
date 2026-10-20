# Architecture and Workflows

```mermaid
flowchart TD
 UI[Streamlit Engineering Workbench] --> API[FastAPI REST]
 API --> C[PipelineCoordinator]
 C --> PDF[PyMuPDF PDF extraction]
 PDF --> CLEAN[Text cleaning]
 CLEAN --> CHUNK[Hierarchical chunking]
 CHUNK --> EM[Sentence-Transformer embeddings]
 EM --> DB[(ChromaDB vector index)]
 CHUNK --> ENT[SWC/Port/Interface rule extraction]
 ENT --> SQL[(SQLite structured entities)]
 C --> RET[Semantic Retriever top-k]
 RET --> DB
 RET --> PROMPT[Grounded Prompt Builder]
 PROMPT --> LLM[MockLLM or Gemini]
 LLM --> VERIFY[Citation ID Verifier]
 VERIFY --> API
 SQL --> API
```

**Ingestion:** PDF → page/block text → cleaning → chunks → embeddings → vector index + deterministic entity extraction → SQLite.

**Q&A:** question → query embedding → ChromaDB nearest neighbors → grounded prompt → provider → cited answer → Python citation validation → answer/evidence in Streamlit.

**Source of truth:** [backend/coordinator.py](../backend/coordinator.py), [core/rag](../core/rag), [frontend/app.py](../frontend/app.py).

The citation check validates the *reference ID*, not semantic entailment. Human reviewers must open the cited source before accepting engineering claims.
