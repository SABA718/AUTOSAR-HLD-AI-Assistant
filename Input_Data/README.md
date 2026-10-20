# Input Data / Knowledge Base

## Actual checked-in sample
- [Synthetic braking-system HLD PDF](../data/samples/AUTOSAR_HLD_Sample_BrakingSystem.pdf)
- [Golden evaluation question set — 10 items](../data/golden_dataset.json)

The braking-system sample is educational test data, not an official AUTOSAR release. The separate synthetic **Door Control** HLD used in prior manual testing is not currently present in this repository; add only with explicit rights to publish.

**Data flow:** text-PDF extraction → clean/section chunks → sentence embeddings → ChromaDB; separately, SWC/Port/Interface parsing → SQLite. A freshly cloned repository regenerates indexes through ingestion; local indexes and private OEM data are not inputs committed to Git.

**Data governance:** verify licensing and ownership for external datasets; never publish confidential supplier/OEM HLDs or credential-bearing logs. Scanned image-only PDFs require OCR not guaranteed by this MVP.
