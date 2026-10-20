# Viva and Interview Questions

**What is the project?** A document-grounded AUTOSAR HLD assistant for engineering Q&A and structured SWC/Port/Interface exploration.

**Why RAG?** Index changing PDFs and retrieve source evidence without retraining model weights; expose source citations.

**Which AI algorithms?** Pretrained transformer embeddings (`all-MiniLM-L6-v2`), vector similarity retrieval, and a Gemini generative model.

**Did you train a model or apply PCA?** No. The project uses pretrained models; no PCA/t-SNE/UMAP stage is implemented.

**What are preprocessing steps?** PyMuPDF block/page extraction, header/footer and hyphen cleaning, whitespace normalization, hierarchical chunking.

**What is top-k?** Number of candidate nearest chunks (5 by default).

**What is the relevance threshold?** Minimum retrieval similarity if enabled. Application default is `None`, so not always applied.

**Why two databases?** ChromaDB for semantic chunks/embeddings; SQLite for structured SWCs/Ports/Interfaces.

**What exactly does Gemini do?** Converts relevant retrieved HLD text into a readable answer; it is not the PDF retriever.

**What does citation verification prove?** Only that cited IDs point to retrieved source; a separate human check is needed for actual claim support.

**What is an extra feature over simple PDF chatbot?** Domain-aware entity extraction and architecture explorer, source provenance, deterministic citations and offline MockLLM testing.

**How do you evaluate?** Tests, golden questions, retrieval/citation-ID metrics, human factual support, refusal correctness and latency.

**What failed during testing?** A generic MockLLM answer with a valid citation was observed even for a missing-MCU question; it is a limitation, not a successful negative-grounding demonstration.

**What next?** Hybrid lexical+vector search for AUTOSAR identifiers, ARXML support, graph relations and stronger factual-entailment checks.
