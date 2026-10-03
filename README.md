# medical-jiwer

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/logo-lockup-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo/logo-lockup.svg">
    <img src="assets/logo/logo-lockup.svg" alt="medical-jiwer logo" width="480">
  </picture>
</p>

Clinical speech-recognition evaluation built on [JiWER](https://github.com/jitsi/jiwer).
Requires Python 3.11+.

Measures WER, Medical-Term WER, medical entity precision/recall/F1, and critical
value accuracy/precision/recall/F1. Values stay linked to individual entity
occurrences, so swapped medication doses count as errors.

## Quick start

Install from this checkout:

```bash
pip install .
```

```python
from medical_jiwer import Annotations, ClinicalScorer, CriticalSlotAnnotation, EntityAnnotation

annotations = Annotations(
    entities=[EntityAnnotation("E1", "MEDICATION", 0, 1, "Ramipril", "ramipril")],
    critical_slots=[CriticalSlotAnnotation(
        "C1", "E1", "DOSE", 1, 3, "5 mg", {"value": 5, "unit": "mg"}
    )],
)
result = ClinicalScorer().score("Ramipril 5 mg", "Ramipril 50 mg", annotations)
print(result.entity_f1)          # 1.0: medication preserved
print(result.critical_accuracy)  # 0.0: dose incorrect
```

Annotation spans use normalized token offsets: start inclusive, end exclusive.
Use `normalize_tokens(reference)` to check offsets before annotating.

The default rules support German doses, numeric/lab/vital values, frequency,
negation and laterality. Clinical canonicalization accepts equivalent forms such
as `fünfzig Milligramm` and `50 mg` without changing ordinary WER.

Provide a `DictionaryRecognizer` to detect additional medical terms and aliases.
Recognition is conservative; complex language can use custom recognizer and
canonicalizer adapters. Reference annotations should cover every enabled category:
unmatched recognized hypothesis entities and slots count as false positives.

## Datasets and CLI

JSON/JSONL samples contain `sample_id`, `reference`, `hypothesis`, `annotations`
and optional `metadata`. Annotation fields match the Python example above.

```bash
medical-jiwer score --dataset benchmark.jsonl --terminology terms.json \
  --group-by accent_group --output results.json
medical-jiwer inspect --dataset benchmark.jsonl --terminology terms.json \
  --sample-id conv_001
```

Terminology JSON accepts strings or entries with `surface`, `canonical` and `type`.
Use a `.csv` output path for CSV reports.

For Python corpus evaluation, use `load_dataset()` and
`scorer.score_corpus(samples, group_by="accent_group")`. Reports aggregate counts
before computing rates and support any metadata field. `bootstrap_ci(report,
seed=42)` provides conversation-level confidence intervals; `cluster_by` enables
speaker clustering.

## Development

```bash
uv sync --group dev
uv run pytest
uv build
```

Pushing a `v<version>` tag matching `pyproject.toml` runs CI and prepares a draft
GitHub release with a wheel and source archive. Local `docs/` are excluded.

Licensed under [Apache-2.0](LICENSE).
