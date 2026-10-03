"""Conversation-level and optional speaker-cluster bootstrap intervals."""

import json
import random
from dataclasses import asdict, dataclass

from .report import CorpusReport, aggregate_results
from .scoring import ClinicalScorer

BOOTSTRAP_METRICS = {
    "wer",
    "mt_wer",
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "critical_accuracy",
    "critical_precision",
    "critical_recall",
    "critical_f1",
    "critical_value_accuracy",
    "critical_value_recall",
}


@dataclass(frozen=True)
class BootstrapInterval:
    metric: str
    estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    seed: int | None
    cluster_by: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(values: list[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    left = int(position)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (position - left)


def _clusters(report: CorpusReport, metadata_field: str | None) -> list[list[int]]:
    if metadata_field is None:
        return [[i] for i in range(len(report.samples))]
    # Connected components keep conversations sharing ANY speaker together.
    parents = list(range(len(report.samples)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, sample in enumerate(report.samples):
        value = sample.metadata.get(metadata_field)
        if value is None or value == []:
            raise ValueError(f"{sample.sample_id}: missing cluster metadata {metadata_field!r}")
        members = value if isinstance(value, list) else [value]
        for member in members:
            key = json.dumps(member, sort_keys=True, ensure_ascii=False, allow_nan=False)
            if key in owners:
                parents[root(i)] = root(owners[key])
            else:
                owners[key] = i
    clusters = {}
    for i in range(len(parents)):
        clusters.setdefault(root(i), []).append(i)
    return list(clusters.values())


def bootstrap_ci(
    samples,
    *,
    metric: str = "mt_wer",
    n_resamples: int = 1000,
    seed: int | None = None,
    confidence: float = 0.95,
    cluster_by: str | None = None,
    scorer: ClinicalScorer | None = None,
) -> BootstrapInterval:
    """Resample complete samples/clusters and recompute corpus micro metrics.

    Accepts a CorpusReport to reuse existing scores; otherwise scores once using
    the supplied scorer. Percentile bounds use linear interpolation.
    """
    if metric not in BOOTSTRAP_METRICS:
        raise ValueError(f"Unsupported bootstrap metric: {metric!r}")
    if type(n_resamples) is not int or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    report = (
        samples
        if isinstance(samples, CorpusReport)
        else (scorer or ClinicalScorer()).score_corpus(samples)
    )
    if not report.samples:
        raise ValueError("Bootstrap requires at least one sample")
    clusters = _clusters(report, cluster_by)
    rng = random.Random(seed)
    values = []
    for _ in range(n_resamples):
        indices = [i for _ in clusters for i in rng.choice(clusters)]
        metrics = aggregate_results(report.results[i] for i in indices)
        values.append(getattr(metrics, metric))
    values.sort()
    tail = (1 - confidence) / 2
    return BootstrapInterval(
        metric,
        getattr(report.metrics, metric),
        _percentile(values, tail),
        _percentile(values, 1 - tail),
        confidence,
        n_resamples,
        seed,
        cluster_by,
    )
