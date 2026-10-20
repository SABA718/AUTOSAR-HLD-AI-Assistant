# Automotive Engineering AI — Project Context

## Project
AUTOSAR HLD Document Analysis Assistant

## Goal
Build a local AI engineering assistant that ingests AUTOSAR High-Level Design (HLD) PDFs, creates a searchable knowledge base, answers architecture questions using grounded RAG, provides source/page citations, and extracts core AUTOSAR architecture entities.

## 2-Day MVP
1. PDF upload and text extraction
2. Page/section metadata preservation
3. Structural chunking
4. Local embeddings
5. ChromaDB vector search
6. Grounded RAG question answering
7. Deterministic citation mapping to retrieved chunks/pages
8. Core entity extraction: Software Components, Ports, Interfaces
9. Streamlit UI
10. Minimal FastAPI layer where useful
11. Focused tests and demo evaluation

## Core Pipeline
PDF → extraction → cleaning → chunking → embeddings → ChromaDB → retrieval → grounded prompt → LLM → citation mapping → UI

## Technology Direction
- Python
- PyMuPDF
- Sentence Transformers
- ChromaDB
- Streamlit
- FastAPI
- SQLite for lightweight metadata/workflow data
- One real LLM provider plus MockLLM for tests

## Grounding Rules
- Answer only from retrieved document evidence.
- Never invent architecture relationships or facts.
- If evidence is insufficient, explicitly say so.
- Preserve source document, page, section, and chunk metadata.
- Prefer deterministic citation IDs over free-form page-number generation by the LLM.
- Human engineering review is required for consequential outputs.

## MVP Entity Scope
Required: Software Components, Ports, Interfaces.
Optional if time permits: Signals.
Future: Runnables, full Functional Flows, interactive architecture graph.

## Development Rules
- Implement one bounded phase/task at a time.
- Do not redesign unrelated modules.
- Keep prompts and agent tasks small.
- Use Gemini Flash as the default agent model; reserve Gemini 3.1 Pro High for difficult multi-module reasoning/review.
- Run simple commands/tests directly instead of asking the agent to do them when practical.
- Keep implementation understandable; avoid abstractions that hide core RAG mechanics.

## Reference
See implementation_plan.md for the full architecture, rationale, stages, risks, MVP boundary, evaluation approach, and security considerations.
