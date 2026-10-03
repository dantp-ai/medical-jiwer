import pytest

from medical_jiwer.entities import DictionaryRecognizer
from medical_jiwer.schemas import Annotations, EntityAnnotation
from medical_jiwer.scoring import ClinicalScorer


@pytest.fixture
def scorer():
    return ClinicalScorer(
        medical_recognizer=DictionaryRecognizer(
            [
                {"surface": term, "canonical": term, "type": "MEDICATION"}
                for term in ["metoprolol", "ramipril", "aspirin", "metamizol"]
            ]
        )
    )


@pytest.mark.parametrize(
    "hypothesis,expected",
    [
        ("der patient nimmt metoprolol", (0, 0, 0)),
        ("die patient nimmt metoprolol", (0, 0, 0)),
        ("der patient nimmt metamizol", (1, 0, 0)),
        ("der patient nimmt", (0, 1, 0)),
        ("der patient nimmt metoprolol und aspirin", (0, 0, 1)),
    ],
)
def test_medical_errors(scorer, hypothesis, expected):
    annotation = Annotations(
        [EntityAnnotation("E1", "MEDICATION", 3, 4, "metoprolol", "metoprolol")]
    )
    result = scorer.score("der patient nimmt metoprolol", hypothesis, annotation)
    assert (
        result.medical_substitutions,
        result.medical_deletions,
        result.medical_insertions,
    ) == expected
    assert result.mt_wer == sum(expected)


def test_multitoken_dictionary_insertion():
    scorer = ClinicalScorer(medical_recognizer=DictionaryRecognizer({"arterielle hypertonie"}))
    result = scorer.score("patient", "patient arterielle hypertonie")
    assert result.medical_insertions == 2
    assert result.medical_reference_word_count == 0
    assert result.mt_wer == 2  # Defined insertion count when N_M == 0.


def test_overlapping_reference_masks_count_tokens_once():
    annotations = Annotations(
        [
            EntityAnnotation("E1", "DIAGNOSIS", 0, 2, "arterielle hypertonie", "hypertension"),
            EntityAnnotation("E2", "DIAGNOSIS", 1, 2, "hypertonie", "hypertension"),
        ]
    )
    result = ClinicalScorer(allow_overlapping_entities=True).score(
        "arterielle hypertonie", "arterielle hypotonie", annotations
    )
    assert result.medical_reference_word_count == 2
    assert result.mt_wer == 0.5
