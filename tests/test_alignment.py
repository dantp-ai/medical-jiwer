import jiwer
import pytest

from medical_jiwer.align import align, project_reference_span_to_hypothesis
from medical_jiwer.normalize import normalize_tokens


def test_token_mapping_matches_jiwer():
    result = align("Patient nimmt Ramipril 5 mg.", "Patient nimmt Ramipril 50 mg.")
    assert result.reference_tokens == ["patient", "nimmt", "ramipril", "5", "mg"]
    assert result.ref_to_hyp == dict(enumerate(range(5)))
    assert result.substitutions == 1
    assert result.wer == pytest.approx(0.2)
    assert result.chunks[-1].type == "equal"


def test_long_deletion_and_following_occurrence():
    result = align("nimmt metoprolol 50 mg morgens dann ramipril", "nimmt dann ramipril")
    span = project_reference_span_to_hypothesis(result, 1, 5)
    assert span.hypothesis_start is None
    assert span.deleted_reference_tokens == ["metoprolol", "50", "mg", "morgens"]
    assert project_reference_span_to_hypothesis(result, 6, 7).hypothesis_tokens == ["ramipril"]


def test_only_internal_insertions_project():
    result = align("a b c d", "x a b y c d z")
    span = project_reference_span_to_hypothesis(result, 0, 4)
    assert span.hypothesis_tokens == ["a", "b", "y", "c", "d"]
    assert span.inserted_hypothesis_tokens == ["y"]
    assert project_reference_span_to_hypothesis(result, 1, 2).hypothesis_tokens == ["b"]


@pytest.mark.parametrize("reference,hypothesis", [("", ""), ("", "a b"), ("a b", "")])
def test_empty_inputs_follow_jiwer(reference, hypothesis):
    result = align(reference, hypothesis)
    assert result.wer == jiwer.process_words(reference, hypothesis).wer
    assert len(result.ref_to_hyp) == len(result.reference_tokens)
    assert len(result.hyp_to_ref) == len(result.hypothesis_tokens)


def test_clinical_number_punctuation_is_preserved():
    assert normalize_tokens("  0,5 mg; 0.5 mg. 120/80 mmHg, 37 °C! ") == [
        "0,5",
        "mg",
        "0.5",
        "mg",
        "120/80",
        "mmhg",
        "37",
        "°c",
    ]
    assert normalize_tokens("fünfzig Milligramm") == ["fünfzig", "milligramm"]


def test_invalid_span():
    with pytest.raises(ValueError, match="Invalid reference span"):
        project_reference_span_to_hypothesis(align("a", "a"), 0, 2)
