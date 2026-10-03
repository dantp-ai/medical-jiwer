"""Count-based metrics with explicit empty-denominator conventions."""

from dataclasses import asdict, dataclass, field
from typing import Any

from .align import AlignmentResult


def error_rate(errors: int, reference_count: int) -> float:
    """For empty references use insertion count, matching JiWER 4 semantics."""
    return errors / reference_count if reference_count else float(errors)


@dataclass
class PRFCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float(self.fn == 0)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 1.0

    def to_dict(self) -> dict[str, int | float]:
        return {**asdict(self), "precision": self.precision, "recall": self.recall, "f1": self.f1}


@dataclass
class ClinicalASRResult:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    reference_word_count: int = 0
    medical_substitutions: int = 0
    medical_deletions: int = 0
    medical_insertions: int = 0
    medical_reference_word_count: int = 0
    entities: PRFCounts = field(default_factory=PRFCounts)
    critical: PRFCounts = field(default_factory=PRFCounts)
    per_entity_type: dict[str, PRFCounts] = field(default_factory=dict)
    per_slot_type: dict[str, PRFCounts] = field(default_factory=dict)
    alignment: AlignmentResult | None = None
    entity_matches: list[dict[str, Any]] = field(default_factory=list)
    critical_matches: list[dict[str, Any]] = field(default_factory=list)
    versions: dict[str, str | None] = field(default_factory=dict)

    @property
    def wer(self) -> float:
        return error_rate(
            self.substitutions + self.deletions + self.insertions, self.reference_word_count
        )

    @property
    def mt_wer(self) -> float:
        return error_rate(
            self.medical_substitutions + self.medical_deletions + self.medical_insertions,
            self.medical_reference_word_count,
        )

    @property
    def entity_tp(self) -> int:
        return self.entities.tp

    @property
    def entity_fp(self) -> int:
        return self.entities.fp

    @property
    def entity_fn(self) -> int:
        return self.entities.fn

    @property
    def entity_precision(self) -> float:
        return self.entities.precision

    @property
    def entity_recall(self) -> float:
        return self.entities.recall

    @property
    def entity_f1(self) -> float:
        return self.entities.f1

    @property
    def critical_tp(self) -> int:
        return self.critical.tp

    @property
    def critical_fp(self) -> int:
        return self.critical.fp

    @property
    def critical_fn(self) -> int:
        return self.critical.fn

    @property
    def critical_precision(self) -> float:
        return self.critical.precision

    @property
    def critical_recall(self) -> float:
        return self.critical.recall

    @property
    def critical_f1(self) -> float:
        return self.critical.f1

    @property
    def critical_accuracy(self) -> float:
        return self.critical.recall

    @property
    def critical_value_accuracy(self) -> float:
        return self.critical_accuracy

    @property
    def critical_value_recall(self) -> float:
        return self.critical_recall

    def to_dict(self, *, include_alignment: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if not include_alignment:
            result.pop("alignment")
        for name in (
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
            "critical_precision",
            "critical_recall",
            "critical_f1",
            "critical_accuracy",
            "critical_value_accuracy",
            "critical_value_recall",
        ):
            result[name] = getattr(self, name)
        result["entities"] = self.entities.to_dict()
        result["critical"] = self.critical.to_dict()
        result["per_entity_type"] = {
            key: value.to_dict() for key, value in self.per_entity_type.items()
        }
        result["per_slot_type"] = {
            key: value.to_dict() for key, value in self.per_slot_type.items()
        }
        return result


def count_event(
    total: PRFCounts, categories: dict[str, PRFCounts], category: str, event: str
) -> None:
    counts = categories.setdefault(category, PRFCounts())
    setattr(total, event, getattr(total, event) + 1)
    setattr(counts, event, getattr(counts, event) + 1)
