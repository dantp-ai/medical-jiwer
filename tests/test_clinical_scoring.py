from decimal import Decimal

import pytest

from medical_jiwer.entities import DictionaryRecognizer
from medical_jiwer.schemas import Annotations, CriticalSlotAnnotation, EntityAnnotation
from medical_jiwer.scoring import ClinicalScorer
from medical_jiwer.slots import GermanClinicalCanonicalizer


@pytest.fixture
def scorer():
    return ClinicalScorer(
        medical_recognizer=DictionaryRecognizer(
            [
                {"surface": term, "canonical": term, "type": "MEDICATION"}
                for term in ["metoprolol", "ramipril", "aspirin", "metamizol"]
            ]
            + [
                {"surface": "thoraxschmerzen", "canonical": "chest_pain", "type": "SYMPTOM"},
                {"surface": "knie", "canonical": "knee", "type": "ANATOMY"},
                {"surface": "blutdruck", "canonical": "blood_pressure", "type": "CLINICAL_FINDING"},
                {"surface": "puls", "canonical": "pulse", "type": "CLINICAL_FINDING"},
                {
                    "surface": "arterielle hypertonie",
                    "canonical": "hypertension",
                    "type": "DIAGNOSIS",
                },
                {"surface": "hypertonie", "canonical": "hypertension", "type": "DIAGNOSIS"},
                {"surface": "hypotonie", "canonical": "hypotension", "type": "DIAGNOSIS"},
            ]
        )
    )


def medication_with_dose(value=50):
    return Annotations(
        [EntityAnnotation("E1", "MEDICATION", 0, 1, "metoprolol", "metoprolol")],
        [
            CriticalSlotAnnotation(
                "C1", "E1", "DOSE", 1, 3, f"{value} mg", {"value": value, "unit": "mg"}
            )
        ],
    )


def test_perfect_transcription(scorer):
    result = scorer.score("metoprolol 50 mg", "metoprolol 50 mg", medication_with_dose())
    assert result.wer == result.mt_wer == 0
    assert result.entity_f1 == result.critical_accuracy == result.critical_f1 == 1
    assert result.critical_tp == 1


@pytest.mark.parametrize(
    "hypothesis,counts",
    [
        ("metamizol", (0, 1, 1)),
        ("", (0, 0, 1)),
        ("ramipril und aspirin", (0, 2, 1)),
    ],
)
def test_entity_replacement_and_deletion(scorer, hypothesis, counts):
    annotations = Annotations(
        [EntityAnnotation("E1", "MEDICATION", 0, 1, "metoprolol", "metoprolol")]
    )
    result = scorer.score("metoprolol", hypothesis, annotations)
    assert (result.entity_tp, result.entity_fp, result.entity_fn) == counts


def test_hallucinated_medication(scorer):
    annotations = Annotations([EntityAnnotation("E1", "MEDICATION", 1, 2, "ramipril", "ramipril")])
    result = scorer.score("nimmt ramipril", "nimmt ramipril und aspirin", annotations)
    assert (result.entity_tp, result.entity_fp, result.entity_fn) == (1, 1, 0)
    assert result.medical_insertions == 1


@pytest.mark.parametrize(
    "hypothesis", ["metoprolol 500 mg", "metoprolol 0,5 mg", "metoprolol 50 g"]
)
def test_wrong_dose(scorer, hypothesis):
    result = scorer.score("metoprolol 50 mg", hypothesis, medication_with_dose())
    assert result.entity_tp == 1
    assert (result.critical_tp, result.critical_fp, result.critical_fn) == (0, 1, 1)
    assert result.mt_wer == 0


@pytest.mark.parametrize(
    "hypothesis", ["metoprolol fünfzig milligramm", "metoprolol 50 milligramm"]
)
def test_equivalent_dose_preserves_ordinary_wer(scorer, hypothesis):
    result = scorer.score("metoprolol 50 mg", hypothesis, medication_with_dose())
    assert result.critical_f1 == 1
    assert result.wer > 0


def test_decimal_comma(scorer):
    result = scorer.score("metoprolol 0,5 mg", "metoprolol 0.5 mg", medication_with_dose("0,5"))
    assert result.critical_tp == 1
    assert result.wer > 0


def test_missing_dose_is_only_fn(scorer):
    result = scorer.score("metoprolol 50 mg", "metoprolol", medication_with_dose())
    assert (result.critical_tp, result.critical_fp, result.critical_fn) == (0, 0, 1)


def test_deleted_negation_is_affirmative_false_positive(scorer):
    annotations = Annotations(
        [EntityAnnotation("E1", "SYMPTOM", 1, 2, "thoraxschmerzen", "chest_pain")],
        [CriticalSlotAnnotation("C1", "E1", "NEGATION", 0, 1, "keine", True)],
    )
    result = scorer.score("keine thoraxschmerzen", "thoraxschmerzen", annotations)
    assert result.entity_f1 == 1
    assert (result.critical_tp, result.critical_fp, result.critical_fn) == (0, 1, 1)
    assert result.critical_matches[0]["hypothesis_canonical"] is False


def test_laterality(scorer):
    annotations = Annotations(
        [EntityAnnotation("E1", "ANATOMY", 1, 2, "knie", "knee")],
        [CriticalSlotAnnotation("C1", "E1", "LATERALITY", 0, 1, "linkes", "left")],
    )
    result = scorer.score("linkes knie", "rechtes knie", annotations)
    assert result.entity_tp == 1
    assert (result.critical_fp, result.critical_fn) == (1, 1)


def test_repeated_medications_with_swapped_doses(scorer):
    reference = (
        "Metoprolol 50 mg absetzen. Ramipril 5 mg weiternehmen. Später Metoprolol 25 mg beginnen."
    )
    hypothesis = (
        "Metoprolol 25 mg absetzen. Ramipril 5 mg weiternehmen. Später Metoprolol 50 mg beginnen."
    )
    annotations = Annotations(
        [
            EntityAnnotation("E1", "MEDICATION", 0, 1, "Metoprolol", "metoprolol"),
            EntityAnnotation("E2", "MEDICATION", 4, 5, "Ramipril", "ramipril"),
            EntityAnnotation("E3", "MEDICATION", 9, 10, "Metoprolol", "metoprolol"),
        ],
        [
            CriticalSlotAnnotation("C1", "E1", "DOSE", 1, 3, "50 mg", {"value": 50, "unit": "mg"}),
            CriticalSlotAnnotation("C2", "E2", "DOSE", 5, 7, "5 mg", {"value": 5, "unit": "mg"}),
            CriticalSlotAnnotation(
                "C3", "E3", "DOSE", 10, 12, "25 mg", {"value": 25, "unit": "mg"}
            ),
        ],
    )
    result = scorer.score(reference, hypothesis, annotations)
    assert (result.entity_tp, result.entity_fp, result.entity_fn) == (3, 0, 0)
    assert (result.critical_tp, result.critical_fp, result.critical_fn) == (1, 2, 2)
    assert [match["status"] for match in result.critical_matches] == ["WRONG", "TP", "WRONG"]


def test_identical_entity_elsewhere_does_not_rescue_occurrence(scorer):
    annotations = Annotations(
        [
            EntityAnnotation("E1", "MEDICATION", 0, 1, "metoprolol", "metoprolol"),
            EntityAnnotation("E2", "MEDICATION", 2, 3, "metoprolol", "metoprolol"),
        ]
    )
    result = scorer.score("metoprolol dann metoprolol", "metamizol dann metoprolol", annotations)
    assert (result.entity_tp, result.entity_fp, result.entity_fn) == (1, 1, 1)


def test_partial_multiword_diagnosis_replacement():
    annotation = Annotations(
        [
            EntityAnnotation(
                "E1", "DIAGNOSIS", 0, 3, "chronisch obstruktive lungenerkrankung", "copd"
            )
        ]
    )
    result = ClinicalScorer().score(
        "chronisch obstruktive lungenerkrankung",
        "chronisch restriktive lungenerkrankung",
        annotation,
    )
    assert (result.entity_tp, result.entity_fp, result.entity_fn) == (0, 1, 1)
    assert result.mt_wer == pytest.approx(1 / 3)


def test_alias_with_shorter_surface(scorer):
    annotation = Annotations(
        [EntityAnnotation("E1", "DIAGNOSIS", 0, 2, "arterielle hypertonie", "hypertension")]
    )
    result = scorer.score("arterielle hypertonie", "hypertonie", annotation)
    assert result.entity_f1 == 1
    assert result.mt_wer == 0.5


def test_wrong_anchor_prevents_correct_dose(scorer):
    result = scorer.score("metoprolol 50 mg", "metamizol 50 mg", medication_with_dose())
    assert (result.critical_tp, result.critical_fp, result.critical_fn) == (0, 1, 1)


def test_hallucinated_dose(scorer):
    annotations = Annotations([EntityAnnotation("E1", "MEDICATION", 0, 1, "ramipril", "ramipril")])
    result = scorer.score("ramipril", "ramipril 5 mg", annotations)
    assert result.critical_fp == 1
    assert result.per_slot_type["DOSE"].fp == 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ("einmal täglich", "1/day"),
        ("zweimal täglich", "2/day"),
        ("morgens und abends", "2/day"),
        ("bei bedarf", "PRN"),
        ("alle 8 stunden", "every_8h"),
        ("alle acht stunden", "every_8h"),
    ],
)
def test_frequency_canonicalization(text, expected):
    assert GermanClinicalCanonicalizer().canonicalize_slot(text.split(), "FREQUENCY") == expected


def test_frequency_expansion_is_not_rescued_by_matching_prefix(scorer):
    annotation = Annotations(
        [EntityAnnotation("E1", "MEDICATION", 0, 1, "ramipril", "ramipril")],
        [CriticalSlotAnnotation("C1", "E1", "FREQUENCY", 1, 2, "morgens", "morning")],
    )
    result = scorer.score("ramipril morgens", "ramipril morgens und abends", annotation)
    assert (result.critical_tp, result.critical_fp, result.critical_fn) == (0, 1, 1)


def test_exact_number_canonicalization():
    canonicalizer = GermanClinicalCanonicalizer()
    assert canonicalizer.canonicalize_slot(["null", "komma", "fünf", "mg"], "DOSE") == {
        "value": Decimal("0.5"),
        "unit": "mg",
    }
    assert canonicalizer.canonicalize_slot(["fünfundzwanzig", "mg"], "DOSE")["value"] == 25
    assert canonicalizer.canonicalize_slot(["unsicher", "mg"], "DOSE") is None


def test_multiword_alias_expands_only_across_insertions():
    recognizer = DictionaryRecognizer(
        [
            {"surface": "copd", "canonical": "copd", "type": "DIAGNOSIS"},
            {
                "surface": "chronisch obstruktive lungenerkrankung",
                "canonical": "copd",
                "type": "DIAGNOSIS",
            },
        ]
    )
    annotations = Annotations([EntityAnnotation("E1", "DIAGNOSIS", 0, 1, "copd", "copd")])
    result = ClinicalScorer(medical_recognizer=recognizer).score(
        "copd", "chronisch obstruktive lungenerkrankung", annotations
    )
    assert (result.entity_tp, result.entity_fp, result.entity_fn) == (1, 0, 0)


def test_extra_vital_value(scorer):
    annotation = Annotations(
        [EntityAnnotation("E1", "CLINICAL_FINDING", 0, 1, "blutdruck", "blood_pressure")],
        [CriticalSlotAnnotation("C1", "E1", "VITAL_VALUE", 1, 2, "120/80", "120/80")],
    )
    result = scorer.score("blutdruck 120/80", "blutdruck 120/80 puls 170", annotation)
    assert (result.critical_tp, result.critical_fp, result.critical_fn) == (1, 1, 0)
    assert result.entity_fp == 1


def test_value_on_other_anchor_is_not_true_positive(scorer):
    annotations = Annotations(
        [
            EntityAnnotation("E1", "MEDICATION", 0, 1, "ramipril", "ramipril"),
            EntityAnnotation("E2", "MEDICATION", 1, 2, "metoprolol", "metoprolol"),
        ],
        [CriticalSlotAnnotation("C1", "E1", "DOSE", 2, 4, "5 mg", {"value": 5, "unit": "mg"})],
    )
    result = scorer.score("ramipril metoprolol 5 mg", "ramipril metoprolol 5 mg", annotations)
    assert (result.critical_tp, result.critical_fp, result.critical_fn) == (0, 1, 1)


def test_end_to_end_case_from_plan(scorer):
    reference = (
        "Patient nimmt Ramipril 5 mg morgens und Metoprolol 50 mg zweimal täglich. "
        "Keine Thoraxschmerzen."
    )
    hypothesis = (
        "Patient nimmt Ramipril 5 mg morgens und Metoprolol 15 mg zweimal täglich. Thoraxschmerzen."
    )
    annotations = Annotations(
        [
            EntityAnnotation("E1", "MEDICATION", 2, 3, "Ramipril", "ramipril"),
            EntityAnnotation("E2", "MEDICATION", 7, 8, "Metoprolol", "metoprolol"),
            EntityAnnotation("E3", "SYMPTOM", 13, 14, "Thoraxschmerzen", "chest_pain"),
        ],
        [
            CriticalSlotAnnotation("C1", "E1", "DOSE", 3, 5, "5 mg", {"value": 5, "unit": "mg"}),
            CriticalSlotAnnotation("C2", "E1", "FREQUENCY", 5, 6, "morgens", "morning"),
            CriticalSlotAnnotation("C3", "E2", "DOSE", 8, 10, "50 mg", {"value": 50, "unit": "mg"}),
            CriticalSlotAnnotation("C4", "E2", "FREQUENCY", 10, 12, "zweimal täglich", "2/day"),
            CriticalSlotAnnotation("C5", "E3", "NEGATION", 12, 13, "Keine", True),
        ],
    )
    result = scorer.score(reference, hypothesis, annotations)
    assert (result.substitutions, result.deletions) == (1, 1)
    assert result.wer == pytest.approx(2 / 14)
    assert result.entity_f1 == 1 and result.mt_wer == 0
    assert (result.critical_tp, result.critical_fp, result.critical_fn) == (3, 2, 2)
    assert result.critical_accuracy == result.critical_f1 == 0.6
    assert result.per_slot_type["NEGATION"].fn == 1
    assert result.per_slot_type["DOSE"].tp == 1


def test_slot_adapter_canonical_values_are_used():
    from medical_jiwer import DetectedSlot

    class TabletRecognizer:
        def extract(self, tokens, entities):
            anchor = entities[0]
            return [
                DetectedSlot(
                    1,
                    3,
                    anchor.start,
                    anchor.end,
                    "DOSE",
                    {"value": 5, "unit": "mg"},
                    "eine tablette",
                )
            ]

    result = ClinicalScorer(slot_recognizer=TabletRecognizer()).score(
        "metoprolol 5 mg", "metoprolol eine tablette", medication_with_dose(5)
    )
    assert (result.critical_tp, result.critical_fp, result.critical_fn) == (1, 0, 0)


def test_signed_numeric_values_remain_distinct(scorer):
    annotations = Annotations(
        [EntityAnnotation("E1", "CLINICAL_FINDING", 0, 1, "puls", "pulse")],
        [CriticalSlotAnnotation("C1", "E1", "NUMERIC_VALUE", 1, 2, "-5", -5)],
    )
    result = scorer.score("puls -5", "puls 5", annotations)
    assert result.wer == 0.5
    assert (result.critical_tp, result.critical_fp, result.critical_fn) == (0, 1, 1)


def test_string_dictionary_uses_annotation_types_for_known_surfaces():
    scorer = ClinicalScorer(medical_recognizer=DictionaryRecognizer({"metoprolol", "aspirin"}))
    result = scorer.score("metoprolol 50 mg", "metoprolol 50 mg", medication_with_dose())
    assert result.entity_f1 == result.critical_f1 == 1


def test_explicit_wrong_entity_type_can_be_relaxed():
    recognizer = DictionaryRecognizer(
        [{"surface": "metoprolol", "canonical": "metoprolol", "type": "SYMPTOM"}]
    )
    annotations = Annotations(
        [EntityAnnotation("E1", "MEDICATION", 0, 1, "metoprolol", "metoprolol")]
    )
    strict = ClinicalScorer(medical_recognizer=recognizer).score(
        "metoprolol", "metoprolol", annotations
    )
    assert (strict.entity_tp, strict.entity_fp, strict.entity_fn) == (0, 1, 1)
    relaxed = ClinicalScorer(medical_recognizer=recognizer, require_entity_type=False).score(
        "metoprolol", "metoprolol", annotations
    )
    assert relaxed.entity_f1 == 1
