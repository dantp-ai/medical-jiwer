"""Corpus micro metrics, metadata stratification, loading and audit output."""

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics import ClinicalASRResult, PRFCounts
from .schemas import SCHEMA_VERSION, BenchmarkSample

COUNT_FIELDS = (
    "substitutions",
    "deletions",
    "insertions",
    "hits",
    "reference_word_count",
    "medical_substitutions",
    "medical_deletions",
    "medical_insertions",
    "medical_reference_word_count",
)
VERSION_FIELDS = (
    "schema_version",
    "normalization_version",
    "terminology_version",
    "annotation_guideline_version",
)


def aggregate_results(results: Iterable[ClinicalASRResult]) -> ClinicalASRResult:
    """Sum sufficient statistics before computing rates; never average F1."""
    aggregate = ClinicalASRResult()
    for result in results:
        if aggregate.versions and aggregate.versions != result.versions:
            raise ValueError("Cannot aggregate results with different benchmark versions")
        aggregate.versions = dict(result.versions)
        for name in COUNT_FIELDS:
            setattr(aggregate, name, getattr(aggregate, name) + getattr(result, name))
        for name in ("entities", "critical"):
            target, source = getattr(aggregate, name), getattr(result, name)
            target.tp += source.tp
            target.fp += source.fp
            target.fn += source.fn
        for name in ("per_entity_type", "per_slot_type"):
            target = getattr(aggregate, name)
            for key, counts in getattr(result, name).items():
                category = target.setdefault(key, PRFCounts())
                category.tp += counts.tp
                category.fp += counts.fp
                category.fn += counts.fn
    return aggregate


@dataclass
class CorpusReport:
    metrics: ClinicalASRResult
    samples: list[BenchmarkSample]
    results: list[ClinicalASRResult]
    groups: dict[str, dict[str, ClinicalASRResult]] = field(default_factory=dict)

    @classmethod
    def from_samples(cls, scorer, samples, *, group_by: str | None = None) -> "CorpusReport":
        samples = [
            BenchmarkSample.from_dict(sample) if isinstance(sample, dict) else sample
            for sample in samples
        ]
        seen = set()
        results = []
        for sample in samples:
            if (
                not isinstance(sample.sample_id, str)
                or not sample.sample_id
                or sample.sample_id in seen
            ):
                raise ValueError(f"Sample ID must be nonempty and unique: {sample.sample_id!r}")
            seen.add(sample.sample_id)
            if sample.schema_version != SCHEMA_VERSION:
                raise ValueError(
                    f"{sample.sample_id}: unsupported schema version {sample.schema_version!r}"
                )
            for name in VERSION_FIELDS[1:]:
                expected = getattr(scorer, name)
                actual = getattr(sample, name)
                if actual is not None and actual != expected:
                    raise ValueError(
                        f"{sample.sample_id}: {name} {actual!r} does not match scorer {expected!r}"
                    )
            try:
                results.append(
                    scorer.score(sample.reference, sample.hypothesis, sample.annotations)
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{sample.sample_id}: {exc}") from exc
        report = cls(aggregate_results(results), samples, results)
        if group_by:
            report.by_group(group_by)
        return report

    def by_group(self, metadata_field: str) -> dict[str, ClinicalASRResult]:
        if metadata_field not in self.groups:
            grouped: dict[str, list[ClinicalASRResult]] = {}
            identities = {}
            for sample, result in zip(self.samples, self.results, strict=True):
                present = metadata_field in sample.metadata
                value = sample.metadata.get(metadata_field)
                identity = (
                    present,
                    json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False),
                )
                label = value if isinstance(value, str) else identity[1] if present else "<missing>"
                if label in identities and identities[label] != identity:
                    raise ValueError(
                        f"Metadata group labels collide for {metadata_field}: {label!r}"
                    )
                identities[label] = identity
                grouped.setdefault(label, []).append(result)
            self.groups[metadata_field] = {
                label: aggregate_results(rows) for label, rows in grouped.items()
            }
        return self.groups[metadata_field]

    def to_dict(self, *, include_alignment: bool = False) -> dict[str, Any]:
        return {
            "sample_count": len(self.samples),
            "metrics": self.metrics.to_dict(),
            "groups": {
                field: {key: result.to_dict() for key, result in groups.items()}
                for field, groups in self.groups.items()
            },
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "metadata": sample.metadata,
                    "metrics": result.to_dict(include_alignment=include_alignment),
                }
                for sample, result in zip(self.samples, self.results, strict=True)
            ],
        }

    def write_json(self, path: str | Path, *, include_alignment: bool = False) -> None:
        Path(path).write_text(
            json.dumps(
                self.to_dict(include_alignment=include_alignment),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def write_csv(self, path: str | Path) -> None:
        fields = [
            "scope",
            "sample_id",
            "group_by",
            "group",
            *COUNT_FIELDS,
            "wer",
            "mt_wer",
            "entity_tp",
            "entity_fp",
            "entity_fn",
            "entity_precision",
            "entity_recall",
            "entity_f1",
            "critical_tp",
            "critical_fp",
            "critical_fn",
            "critical_accuracy",
            "critical_precision",
            "critical_recall",
            "critical_f1",
        ]
        rows = [{"scope": "corpus", **self.metrics.to_dict()}]
        rows.extend(
            {"scope": "sample", "sample_id": sample.sample_id, **result.to_dict()}
            for sample, result in zip(self.samples, self.results, strict=True)
        )
        rows.extend(
            {"scope": "group", "group_by": name, "group": label, **result.to_dict()}
            for name, groups in self.groups.items()
            for label, result in groups.items()
        )
        with Path(path).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def load_dataset(path: str | Path) -> list[BenchmarkSample]:
    """Read JSONL, a JSON list, or a JSON object containing samples and versions."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(BenchmarkSample.from_dict(json.loads(line)))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    versions = {}
    if isinstance(data, dict) and "samples" in data:
        versions = {name: data[name] for name in VERSION_FIELDS if name in data}
        data = data["samples"]
    elif isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("JSON dataset must be a sample, a list, or an object with samples")
    return [BenchmarkSample.from_dict({**versions, **sample}) for sample in data]


def render_clinical_alignment(result: ClinicalASRResult) -> str:
    if result.alignment is None:
        raise ValueError("Alignment inspection requires a single-sample result")
    alignment = result.alignment
    lines = [
        "REF: " + " ".join(alignment.reference_tokens),
        "HYP: " + " ".join(alignment.hypothesis_tokens),
        f"WER: {result.wer:.6f}  MT-WER: {result.mt_wer:.6f}",
        "",
        "TOKENS:",
    ]
    for row in alignment.token_alignments:
        ref = alignment.reference_tokens[row.ref_index] if row.ref_index is not None else "∅"
        hyp = alignment.hypothesis_tokens[row.hyp_index] if row.hyp_index is not None else "∅"
        lines.append(
            f"  R{row.ref_index!s:>4} {ref} -> H{row.hyp_index!s:>4} {hyp} [{row.operation}]"
        )
    lines.append("\nMEDICAL TOKEN ERRORS:")
    lines.append(
        f"  substitutions={result.medical_substitutions} deletions={result.medical_deletions} "
        f"insertions={result.medical_insertions} "
        f"reference_tokens={result.medical_reference_word_count}"
    )
    for title, matches in [
        ("ENTITY", result.entity_matches),
        ("CRITICAL", result.critical_matches),
    ]:
        lines.append(f"\n{title}:")
        for match in matches:
            anchor = f" anchor={match['anchor_entity_id']}" if "anchor_entity_id" in match else ""
            lines.append(
                f"  {match.get('reference_id') or 'extra'} {match['type']}{anchor}: "
                f"{match.get('reference', '∅')} -> {match.get('hypothesis') or '∅'} "
                f"[{match['status']}]"
            )
            lines.append(
                f"    canonical: {match.get('reference_canonical')} "
                f"-> {match.get('hypothesis_canonical')}"
            )
    return "\n".join(lines)


def score_corpus(samples, *, group_by: str | None = None, **scorer_options) -> CorpusReport:
    from .scoring import ClinicalScorer

    return ClinicalScorer(**scorer_options).score_corpus(samples, group_by=group_by)
