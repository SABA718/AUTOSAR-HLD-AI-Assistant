# Models and AI Methods

| Task | Technology | Role |
|---|---|---|
| Ingestion | PyMuPDF | Extract text, blocks, pages |
| Cleaning / chunking | Python | Normalize PDF text and section chunks |
| Embedding | `all-MiniLM-L6-v2` via Sentence Transformers | Pretrained text-to-vector inference |
| Retrieval | ChromaDB | Similarity-ranked evidence search |
| Answer generation | Google Gemini | Grounded natural-language output |
| Test provider | `MockLLMProvider` | Repeatable offline responses, no API costs |
| Citation verification | Python `CitationVerifier` | Validate retrieved citation IDs |
| Entity extraction | Regex/table rules, optional LLM fallback | SWC/Port/Interface records |

No new transformer was trained or fine-tuned. No PCA/t-SNE/UMAP dimensionality reduction is implemented. A sentence embedding is a learned vector representation, not PCA.

**Model ID discrepancy:** [config/settings.py](../config/settings.py) defaults to `gemini-3-flash-preview`, whereas [root .env.example](../.env.example) specifies `gemini-2.5-flash`. The real effective model depends on the private `.env` used for the specific run; do not report a model that was not verified.

**Responsible interpretation:** a valid citation identifier confirms source provenance, **not** that each claim is factually supported. A human reviewer checks the quoted text.
