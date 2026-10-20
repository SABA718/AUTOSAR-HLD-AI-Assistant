# Installation and Demonstration — Windows PowerShell

From repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env locally. Start with DEFAULT_LLM_PROVIDER=mock.
python -m pytest -q
```

**Terminal 1 — API:**
```powershell
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`.

**Terminal 2 — UI:**
```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run frontend/app.py
```

Open `http://localhost:8501`. Upload `data/samples/AUTOSAR_HLD_Sample_BrakingSystem.pdf`; inspect metrics, SWC/Port/Interface tables; ask a document-answerable question, open verified citations; ask an absent-fact question and inspect whether it refuses.

**Real Gemini:** store `GEMINI_API_KEY` privately in `.env`; select `DEFAULT_LLM_PROVIDER=gemini` and an available model; restart both services. Capture the provider label, factual answer and source evidence without leaking secrets.

MockLLM is an integration test double, not a measure of natural answer quality. Real LLM results and negative grounding require human inspection.
