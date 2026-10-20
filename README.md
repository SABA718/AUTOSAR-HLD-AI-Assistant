# AUTOSAR HLD AI Assistant — Case Study 1

This repository is organized for the eight required academic submission folders:

- [Synopsis](Synopsis/Project_Synopsis.pdf) — project synopsis
- [Input_Data](Input_Data/) — sample engineering HLD PDF
- [Code](Code/) — **complete runnable Python implementation** (backend, frontend, core, config, storage, tests, scripts, requirements)
- [Model_Prompts_Config](Model_Prompts_Config/) — model/prompt configuration
- [Evaluation_Results](Evaluation_Results/) — evaluation report, results worksheet and golden dataset
- [Documentation](Documentation/Technical_Report.pdf) — consolidated technical report
- [Video](Video/) — deferred demonstration
- [Declarations](Declarations/) — personalized academic integrity and AI tool usage forms

## Run from Code/

```powershell
cd Code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m pytest -q
python -m uvicorn backend.main:app --reload
```

In a second activated terminal from `Code/`:

```powershell
python -m streamlit run frontend/app.py
```

Use the sample PDF at `../Input_Data/AUTOSAR_HLD_Sample_BrakingSystem.pdf`. Never commit a real `.env`, tokens, local runtime indices, or confidential automotive design documents. Formal PDF submissions are Times New Roman, 12 pt.

**Note:** The code was relocated without modifying underlying Python algorithms. A fresh CI or local test run from `Code/` should confirm the new folder paths. If test helpers assume `data/samples` relative to the executable root, adapt only those test fixture paths to `../Input_Data` before reporting them as passed.
