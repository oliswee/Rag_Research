# Local retrieval evaluation

Run these commands from `backend/`. The exported corpus, labels, and reports stay in `evaluation/local/`, which Git ignores because excerpts may contain user research material.

1. Import papers, configure `OPENAI_API_KEY`, and ensure Qdrant is running with source vectors indexed. Export the current SQLite excerpts:

   ```powershell
   uv run python -m evaluation.export_for_labeling
   ```

2. Use `evaluation/local/chunks.jsonl` to write `evaluation/local/labels.jsonl`. Each line is one independently reviewed query. Use at least 40 distinct queries across `exact_terms`, `semantic_rephrase`, `cross_paper`, and `no_answer`. Relevant IDs must come from the exported corpus; `no_answer` has an empty list. Keep these test queries out of retrieval parameter tuning.

   ```json
   {"query":"What does the paper report about retrieval latency?","group":"exact_terms","relevant_chunk_ids":["chunk-id-from-export"]}
   {"query":"What evidence covers an unrelated topic?","group":"no_answer","relevant_chunk_ids":[]}
   ```

3. Run the benchmark. Only pass the confirmation flag after people have reviewed the labels and kept them separate from tuning data:

   ```powershell
   uv run python -m evaluation.evaluate evaluation/local/labels.jsonl --output evaluation/local/report.json --confirm-human-test-labels
   ```

The report contains Recall@10, nDCG@10, latency, actual retrieval mode, answer text, citation checks, and explicit reasons when the quality gate fails. No-answer queries are excluded from Recall and nDCG averages because those metrics are undefined without relevant chunks. Review their answers separately for unsupported conclusions.

For optional answer faithfulness scoring, install DeepEval in the evaluation environment and add `--judge --judge-model MODEL_ID`. Set `ANTHROPIC_API_KEY` for answer generation and `OPENAI_API_KEY` for the judge. The report records the selected judge model and DeepEval version. This sends retrieved excerpts and answers to the configured model providers. Model scores supplement citation ID checks and human review; they do not prove a citation supports its claim.
