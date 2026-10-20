# Current Evaluation Status — Historical vs Verified

| Check | Status / interpretation |
|---|---|
| Source code | The GitHub repository contains backend, frontend, core, storage, tests and golden dataset |
| Historical pytest status | Student previously reported 70 passed / 1 skipped after Phase 7; **not freshly rerun by this documentation commit** |
| Golden data | Ten-question synthetic braking dataset is checked in |
| Door Control PDF manual test | User reported successful upload and 16 Chroma chunks; the PDF itself is not currently in this repo |
| Positive MockLLM question | Relevant chunk and reference ID returned, but answer wording was generic |
| Negative MockLLM MCU question | Generic cited answer appeared, rather than the desired abstention: **does not demonstrate refusal success** |
| Gemini behavior | Has not been independently measured in this documentation update |
| Citation verifier | Checks referenced chunk IDs, not whether the cited text actually supports every claim |

**Do not publish fabricated accuracy numbers.** Append fresh time-stamped results to [RESULTS_TO_FILL.csv](RESULTS_TO_FILL.csv).
