# Ontology snapshot

This directory contains normalized, offline indexes built by
`scripts/download_ontologies.py`.

- `icd/concepts.jsonl`: CDC ICD-10-CM FY2026 code descriptions. The source
  fixed-width code is converted to the standard dotted form (`A000` → `A00.0`).
- `rxnorm/concepts.jsonl`: NLM RxNorm prescribable release dated 2026-07-06,
  extracted from English `RXNCONSO.RRF`.
- `manifest.json`: source URLs, archive SHA-256 values, release format and
  concept counts.

The BTC must confirm whether ICD-10-CM or WHO ICD-10 is the required variant.
Do not mix releases or replace these files without updating the manifest.
RxNorm full (non-prescribable) releases may require a UMLS license; the
prescribable release is used here because NLM marks it as no-license.
