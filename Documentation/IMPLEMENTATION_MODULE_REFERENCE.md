# Implementation Module Reference

| Module | Primary responsibility |
|---|---|
| `config/settings.py` | Pydantic BaseSettings, environment overrides and storage paths |
| `core/logger.py` | Structured logging |
| `core/ingestion/pdf_extractor.py` | PyMuPDF document/pages/blocks extraction |
| `core/ingestion/cleaner.py` | Hyphenation, whitespace, recurring header/footer cleaning |
| `core/chunking/hierarchical_chunker.py` | Structural/section chunks |
| `core/embeddings/embedder.py` | Pretrained Sentence Transformers inference |
| `core/vector_store/chroma_manager.py` | Chroma persistent index, vector lookups |
| `core/retrieval/retriever.py` | Ranked chunks, top-k and optional threshold |
| `core/rag/prompt_builder.py` | Evidence delimitation and grounding instructions |
| `core/llm/base.py` | Provider-neutral response interface |
| `core/llm/mock_provider.py` | Deterministic mock response |
| `core/llm/gemini_provider.py` | Gemini inference |
| `core/llm/factory.py` | Provider selection |
| `core/rag/citation_engine.py` | Valid/invalid citation ID mapping |
| `core/extraction/entity_extractor.py` | SWC/Port/Interface candidates, validation, deduplication |
| `storage/database.py` | SQLite access/upserts |
| `backend/coordinator.py` | Ingest, query, entities orchestration |
| `backend/main.py` | FastAPI endpoints |
| `frontend/app.py` | UI upload, question and entity tables |
| `core/evaluation/evaluator.py` | Dataset-based proxy metrics |
| `tests` | Component/API/UI regression tests |
