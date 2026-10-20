# AUTOSAR HLD AI Assistant

An intelligent, local RAG and architectural entity extraction assistant tailored for automotive **AUTOSAR High-Level Design (HLD)** specifications.

---

## 🚀 Key Features

1. **Local PDF Ingestion & Layout Preservation**: Extracts digital text layers, font hierarchies, and tabular boundaries with coordinate bounding boxes using PyMuPDF.
2. **Structural Hierarchical Chunking**: Respects section headers (`1.0`, `1.1`), avoids breaking contiguous tables across chunks, and applies sentence-boundary sliding overlaps.
3. **Local Vector Search**: Encodes document passages via local `SentenceTransformers` (`all-MiniLM-L6-v2`) and persists embeddings in ChromaDB.
4. **Strictly Grounded Engineering Q&A**: Delivers answers verified against authentic retrieved chunks with bracketed page and section citations (`[chunk_id]`).
5. **Deterministic Negative Grounding**: Admits missing technical context (*"Based on the provided AUTOSAR specification, there is insufficient evidence to answer this question."*) rather than hallucinating plausible automotive details.
6. **Architecture Entity Extraction**: Extracts Software Components (SWCs), Ports (with directions), and Interfaces (Sender-Receiver, Client-Server) into a local SQLite repository.
7. **Dual Interface**: Includes a minimal FastAPI REST backend and an interactive Streamlit UI.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / Linux / macOS

### 2. Environment Setup
```powershell
# Clone or navigate to the project directory
cd "d:\New folder (5)"

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```

Configuration variables in `.env`:
- `DEFAULT_LLM_PROVIDER`: Set to `mock` for deterministic offline testing (requires no API keys). Set to `gemini` if an API key is provided.
- `GEMINI_API_KEY`: (Optional) Your Google Gemini API key.
- `CHROMA_PERSIST_DIR`: Vector store path (defaults to `./data/chroma_db`).
- `SQLITE_DB_PATH`: SQLite database path (defaults to `./data/autosar_assistant.db`).

---

## 💻 Running the Application

### 1. Seed Demo Sample Document
To generate the sample AUTOSAR braking system specification PDF:
```powershell
python scripts/seed_demo_data.py
```
*Generated PDF is saved at: `data/samples/AUTOSAR_HLD_Sample_BrakingSystem.pdf`.*

### 2. Launch FastAPI Service
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)

### 3. Launch Streamlit UI
In a separate terminal:
```powershell
streamlit run frontend/app.py
```
- Open your browser to [http://localhost:8501](http://localhost:8501).

---

## 🎬 5–8 Minute Live Demo Walkthrough

A complete, scripted demonstration guide is available in [`DEMO_SCRIPT.md`](file:///d:/New%20folder%20%285%29/DEMO_SCRIPT.md).

### Summary of Demo Steps:
1. **Document Ingestion**: Open the **Document Upload** tab, upload `data/samples/AUTOSAR_HLD_Sample_BrakingSystem.pdf`, and observe immediate processing metrics (pages, chunks, SWCs, ports, interfaces).
2. **Engineering Q&A**: Ask a factual question such as *"What ports are defined for SWC_BrakingController and what are their directions?"*. Review the answer and verified source citations.
3. **Negative Grounding**: Ask an unanswerable question (e.g. *"What is the quantum encryption algorithm and cryptographic seed used for satellite uplink telematics?"*). Observe the standardized insufficient-evidence response.
4. **Architecture Explorer**: Inspect extracted Software Components, Ports, and Interfaces in the interactive tabs.

---

## 🧪 Running Automated Tests & Evaluation

To execute the full test suite (74 tests covering ingestion, chunking, embeddings, vector store, retrieval, prompt construction, LLMs, citation verification, entity extraction, FastAPI endpoints, Streamlit UI, and golden dataset evaluation):

```powershell
python -m pytest -v
```

### Golden Evaluation Benchmark Metrics
To run only the golden dataset evaluation:
```powershell
python -m pytest tests/test_evaluation.py -v
```

Expected benchmark scores against the 10-question evaluation dataset:
- **Retrieval Success Rate**: 100.0%
- **Groundedness Rate**: 100.0%
- **Citation Validity Rate**: 100.0%
- **Negative Grounding Accuracy**: 100.0%
