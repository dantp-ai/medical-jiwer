import jiwer
import pytest

from medical_jiwer import Annotations, ClinicalScorer, CriticalSlotAnnotation, EntityAnnotation


def entity(**changes):
    fields = dict(
        id="E1", type="MEDICATION", ref_start=0, ref_end=1, surface="ramipril", canonical="ramipril"
    )
    return EntityAnnotation(**(fields | changes))


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"ref_start": -1}, "span"),
        ({"ref_end": 2}, "span"),
        ({"ref_start": True}, "span"),
        ({"ref_end": 0}, "span"),
        ({"surface": "aspirin"}, "surface"),
        ({"canonical": ""}, "canonical"),
        ({"id": ""}, "unique"),
        ({"type": ""}, "type"),
    ],
)
def test_invalid_entity_annotations_fail_fast(changes, match):
    with pytest.raises(ValueError, match=match):
        ClinicalScorer().score("ramipril", "ramipril", Annotations([entity(**changes)]))


def test_duplicate_ids_and_overlapping_entities():
    with pytest.raises(ValueError, match="unique"):
        ClinicalScorer().score("ramipril", "", Annotations([entity(), entity()]))
    with pytest.raises(ValueError, match="Overlapping"):
        ClinicalScorer().score("ramipril", "", Annotations([entity(), entity(id="E2")]))


@pytest.mark.parametrize(
    "slot,match",
    [
        (
            CriticalSlotAnnotation(
                "C1", "missing", "DOSE", 1, 3, "5 mg", {"value": 5, "unit": "mg"}
            ),
            "anchor",
        ),
        (
            CriticalSlotAnnotation(
                "C1", "E1", "DOSE", 1, 3, "5 mg", {"value": float("nan"), "unit": "mg"}
            ),
            "finite",
        ),
        (
            CriticalSlotAnnotation("C1", "E1", "DOSE", 1, 3, "5 mg", {"value": True, "unit": "mg"}),
            "Boolean",
        ),
        (
            CriticalSlotAnnotation(
                "C1", "E1", "DOSE", 1, 3, "5 mg", {"value": 5, "unit": "unknown"}
            ),
            "unit",
        ),
        (CriticalSlotAnnotation("C1", "E1", "DOSE", 1, 3, "5 mg", "5 mg"), "object"),
        (CriticalSlotAnnotation("C1", "E1", "NEGATION", 0, 1, "ramipril", "true"), "boolean"),
        (CriticalSlotAnnotation("C1", "E1", "LATERALITY", 0, 1, "ramipril", {}), "LATERALITY"),
    ],
)
def test_invalid_slots(slot, match):
    with pytest.raises(ValueError, match=match):
        ClinicalScorer().score("ramipril 5 mg", "ramipril 5 mg", Annotations([entity()], [slot]))


def test_custom_asr_transform_requires_version():
    transform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.ReduceToListOfListOfWords()])
    with pytest.raises(ValueError, match="normalization_version"):
        ClinicalScorer(asr_transform=transform)
    scorer = ClinicalScorer(asr_transform=transform, normalization_version="custom-v1")
    assert scorer.score("RAMIPRIL", "ramipril").wer == 0


def test_surface_checks_can_be_explicitly_disabled():
    annotation = Annotations([entity(surface="label only")])
    result = ClinicalScorer(check_surfaces=False).score("ramipril", "ramipril", annotation)
    assert result.wer == 0 and result.entity_f1 == 1
