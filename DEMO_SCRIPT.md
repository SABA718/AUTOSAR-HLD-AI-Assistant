# DEMO SCRIPT: AUTOSAR HLD AI Assistant (5–8 Minute Walkthrough)

This script provides an end-to-end, live walkthrough demonstration of the **AUTOSAR High-Level Design (HLD) AI Assistant** for automotive software engineers and technical evaluators.

---

## 🎯 Demo Objectives
Demonstrate that the system:
1. Ingests and indexes authentic AUTOSAR HLD PDF specifications locally.
2. Extracts architectural domain entities (Software Components, Ports, Interfaces).
3. Answers engineering queries with strict grounding and verifiable page-level citations.
4. Admits ignorance with standardized negative grounding when evidence is missing.

---

## ⏱️ Timeline Breakdown (Total: 6–7 Minutes)

| Time | Segment | Primary Activity |
|---|---|---|
| **0:00 - 1:00** | Introduction & Architecture Context | OEM confidentiality & RAG hallucination problem |
| **1:00 - 2:30** | Document Upload & Processing | Upload sample PDF & review processing stats |
| **2:30 - 4:00** | Grounded Engineering Q&A | Ask factual question & inspect verified citations |
| **4:00 - 5:00** | Negative Grounding Admission | Ask unanswerable question ("insufficient evidence") |
| **5:00 - 6:30** | Architecture Explorer | Review SWCs, Ports, directions, and interfaces |
| **6:30 - 7:00** | Wrap-up & Evaluation Metrics | Review RAG Triad scores & local reproducibility |

---

## 🛠️ Step-by-Step Live Walkthrough

### 1. Preparation & Startup (Before Presentation)
Start the local FastAPI service and Streamlit frontend in separate terminals:

```powershell
# Terminal 1: Backend Service
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Streamlit Frontend
streamlit run frontend/app.py
```
*Open browser to `http://localhost:8501`.*

---

### 2. Segment 1: Introduction (0:00 - 1:00)
- **Speaker Script:**
  > *"Automotive Tier-1 suppliers and OEMs handle massive, 200+ page AUTOSAR High-Level Design documents. Standard LLM chatbots hallucinate architectural details or leak proprietary IP to external clouds. Today, I am demonstrating a strictly local, grounded AI assistant designed specifically for AUTOSAR Classic and Adaptive specifications."*

---

### 3. Segment 2: Document Ingestion (1:00 - 2:30)
1. In the sidebar, verify **LLM Provider** is set to `mock` (or `gemini` if API key configured).
2. Click **Document Upload** in the navigation menu.
3. Drag & drop or browse for the sample document: `data/samples/AUTOSAR_HLD_Sample_BrakingSystem.pdf`.
4. Click **🚀 Process & Index Document**.
5. Observe the live processing statistics:
   - **Total Pages:** `2`
   - **Indexed Chunks:** `3`
   - **SWCs Found:** `1` (`SWC_BrakingController`)
   - **Ports Found:** `2` (`P_BrakePressure`, `R_PedalPosition`)
   - **Interfaces Found:** `2` (`If_BrakePressure`, `If_PedalPosition`)
- **Key Talking Point:**
  > *"Notice that during ingestion, the layout extractor preserved table boundaries, extracted bounding boxes, calculated dense vector embeddings in ChromaDB, and populated our relational SQLite catalog for deterministic architectural queries."*

---

### 4. Segment 3: Grounded Engineering Q&A (2:30 - 4:00)
1. Select **Engineering Q&A** in the sidebar.
2. In the query box, enter:
   ```text
   What ports are defined for SWC_BrakingController and what are their directions?
   ```
3. Click **🔍 Submit Question**.
4. Highlight the following UI elements:
   - **Answer Display:** Generates an engineering answer with bracketed citation markers `[chunk_id]`.
   - **Grounding Badge:** Displays **✅ Grounded**.
   - **Latency Metric:** Displays sub-second response time.
   - **📚 Verified Citations:** Expand Citation 1 to reveal:
     - Document ID: `AUTOSAR_HLD_Sample_BrakingSystem`
     - Source Page: `Page 2`
     - Section: `2.0 Port Specifications`
     - Verifiable Snippet: Displays the exact Markdown table row extracted from the document.
- **Key Talking Point:**
  > *"Every claim is deterministically verified by the citation engine. If an LLM fabricates a citation ID or fails to provide an authentic chunk ID, the system flags a grounding warning immediately."*

---

### 5. Segment 4: Negative Grounding Demonstration (4:00 - 5:00)
1. In the query box, enter an unanswerable question outside the document's scope:
   ```text
   What is the quantum encryption algorithm and cryptographic seed used for satellite uplink telematics?
   ```
2. Click **🔍 Submit Question**.
3. Highlight the result:
   - **Status Badge:** Displays **⚠️ Insufficient Evidence**.
   - **Standardized Response:**
     > *"Based on the provided AUTOSAR specification, there is insufficient evidence to answer this question."*
- **Key Talking Point:**
  > *"Instead of hallucinating plausible telematics parameters, the model rigorously adheres to negative grounding constraints and admits the document lacks supporting facts."*

---

### 6. Segment 5: Architecture Explorer (5:00 - 6:30)
1. Select **Architecture Explorer** in the sidebar.
2. Ensure Document ID is set to `AUTOSAR_HLD_Sample_BrakingSystem`.
3. Walk through the tabs:
   - **Software Components Tab:** Expand `SWC_BrakingController` to see its component type (`ApplicationSwComponentType`), description, and source page (`Page 1`).
   - **Ports Tab:** Show `P_BrakePressure [Provided]` linked to `If_BrakePressure` and `R_PedalPosition [Required]` linked to `If_PedalPosition`.
   - **Interfaces Tab:** Show `If_BrakePressure` (`SenderReceiverInterface`).
- **Key Talking Point:**
  > *"Engineers can inspect components without needing to run ARXML parsers or manually scan pages."*

---

### 7. Segment 6: Wrap-up & Evaluation (6:30 - 7:00)
- Mention automated testing results:
  - 74 automated unit, integration, and evaluation tests.
  - Golden evaluation benchmark:
    - **Retrieval Success Rate:** `100%`
    - **Groundedness Rate:** `100%`
    - **Citation Validity Rate:** `100%`
    - **Negative Grounding Accuracy:** `100%`
- **Closing Statement:**
  > *"The assistant is self-contained, air-gapped ready, and provides verifiable traceability required for safety-critical automotive engineering."*
