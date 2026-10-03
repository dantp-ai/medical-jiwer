"""Public annotation and benchmark schemas. Offsets are normalized token indices."""

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class EntityAnnotation:
    id: str
    type: str
    ref_start: int
    ref_end: int
    surface: str
    canonical: str
    concept_id: str | None = None


@dataclass(frozen=True)
class CriticalSlotAnnotation:
    id: str
    anchor_entity_id: str
    type: str
    ref_start: int
    ref_end: int
    surface: str
    canonical: Any


@dataclass
class Annotations:
    entities: list[EntityAnnotation] = field(default_factory=list)
    critical_slots: list[CriticalSlotAnnotation] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Annotations":
        if not isinstance(data, dict):
            raise ValueError("Annotations must be a JSON object")
        return cls(
            entities=[EntityAnnotation(**entry) for entry in data.get("entities", [])],
            critical_slots=[
                CriticalSlotAnnotation(**entry) for entry in data.get("critical_slots", [])
            ],
        )


@dataclass
class BenchmarkSample:
    sample_id: str
    reference: str
    hypothesis: str
    annotations: Annotations = field(default_factory=Annotations)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    normalization_version: str | None = None
    terminology_version: str | None = None
    annotation_guideline_version: str | None = None

    def __post_init__(self):
        if not isinstance(self.reference, str) or not isinstance(self.hypothesis, str):
            raise ValueError("Reference and hypothesis must be strings")
        if not isinstance(self.annotations, Annotations):
            raise ValueError("Sample annotations must be an Annotations object")
        if not isinstance(self.metadata, dict):
            raise ValueError("Sample metadata must be a JSON object")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkSample":
        if not isinstance(data, dict):
            raise ValueError("Each dataset sample must be a JSON object")
        data = dict(data)
        annotations = data.get("annotations", {})
        data["annotations"] = (
            Annotations.from_dict(annotations)
            if not isinstance(annotations, Annotations)
            else annotations
        )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
