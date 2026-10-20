# Input and Preprocessing

1. Ingest a digital-text HLD PDF with PyMuPDF; preserve 1-indexed page positions, text blocks and headings/layout when available.
2. Normalize line-wrap hyphenation, repetitive headers, margin page number footers and whitespace; retain necessary structural and table boundaries.
3. Split pages into hierarchical/section chunks; retain source document, page, section and deterministic chunk ID for later citations.
4. Encode chunks using pretrained `all-MiniLM-L6-v2`; persist vectors/chunk text/source metadata in ChromaDB.
5. Parse structured AUTOSAR names and common tables, validate/deduplicate SWCs/Ports/Interfaces and upsert into SQLite.

**Included test artifacts:** [synthetic Braking System PDF](../data/samples/AUTOSAR_HLD_Sample_BrakingSystem.pdf) and [ten golden questions](../data/golden_dataset.json). Door Control sample mentioned in manual tests is not currently in this GitHub repository.

**Limitations:** image-only scans need OCR, unusual multi-column layouts and diagrams can lose source relationships. No PCA/t-SNE/UMAP stage is used.
