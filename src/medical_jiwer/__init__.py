"""Clinical ASR metrics built on JiWER. Distribution name: medical-jiwer."""

from importlib.metadata import version

from .align import AlignmentResult, ProjectedSpan, align, project_reference_span_to_hypothesis
from .annotations import validate_annotations
from .entities import (
    CompositeRecognizer,
    DetectedEntity,
    DictionaryEntry,
    DictionaryRecognizer,
    MedicalRecognizer,
)
from .metrics import ClinicalASRResult, PRFCounts
from .normalize import NORMALIZATION_VERSION, default_asr_transform, normalize_tokens
from .report import CorpusReport, load_dataset, render_clinical_alignment, score_corpus
from .schemas import (
    SCHEMA_VERSION,
    Annotations,
    BenchmarkSample,
    CriticalSlotAnnotation,
    EntityAnnotation,
)
from .scoring import ClinicalScorer, score_sample
from .slots import (
    ClinicalCanonicalizer,
    DetectedSlot,
    GermanClinicalCanonicalizer,
    RuleBasedSlotRecognizer,
    SlotRecognizer,
)
from .statistics import BootstrapInterval, bootstrap_ci

__version__ = version("medical-jiwer")

__all__ = [
    "AlignmentResult",
    "Annotations",
    "BenchmarkSample",
    "BootstrapInterval",
    "ClinicalASRResult",
    "ClinicalCanonicalizer",
    "ClinicalScorer",
    "CompositeRecognizer",
    "CorpusReport",
    "CriticalSlotAnnotation",
    "DetectedEntity",
    "DetectedSlot",
    "DictionaryEntry",
    "DictionaryRecognizer",
    "EntityAnnotation",
    "GermanClinicalCanonicalizer",
    "MedicalRecognizer",
    "NORMALIZATION_VERSION",
    "PRFCounts",
    "ProjectedSpan",
    "RuleBasedSlotRecognizer",
    "SCHEMA_VERSION",
    "SlotRecognizer",
    "align",
    "bootstrap_ci",
    "default_asr_transform",
    "load_dataset",
    "normalize_tokens",
    "project_reference_span_to_hypothesis",
    "render_clinical_alignment",
    "score_corpus",
    "score_sample",
    "validate_annotations",
]
