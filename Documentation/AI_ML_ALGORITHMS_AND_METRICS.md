# AI/ML Algorithms, Dimensionality and Metrics

**ML method:** pretrained transformer encoder `all-MiniLM-L6-v2` produces semantic vectors for both retrieved chunks and questions. **Retrieval:** ChromaDB vector similarity ranks candidate evidence. **Generative model:** Gemini synthesizes a grounded-language answer; MockLLM substitutes deterministic text in tests. **Domain logic:** structured AUTOSAR extraction and citation ID verification are deterministic rule-based components.

**Dimensionality reduction:** no separately fitted PCA, t-SNE or UMAP. Learning an embedding is not equivalent to applying PCA.

**Threshold:** normal default `top_k=5`, relevance threshold `None`. Top-k controls result count; a cutoff filters weak matches. Similarity scores are not probabilities of truth. Evaluator has a 0.3 default argument under a distinct evaluation path; do not confuse this with application default.

**Evaluation available in source:** retrieval success, citation ID validity, implementation-defined groundedness rate, negative-grounding accuracy and expected-keyword overlap. They are limited: a correctly cited generic answer can be useless or unsupported. Manual comparisons with the actual PDF are essential.

**Why AI matters:** semantic retrieval beyond exact keywords and engineering-text generation based on local documentary evidence; source provenance and testing are software engineering contributions. No newly trained model claimed.
