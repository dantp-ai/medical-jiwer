import csv
import json

import pytest

from medical_jiwer import (
    Annotations,
    BenchmarkSample,
    ClinicalScorer,
    EntityAnnotation,
    bootstrap_ci,
    load_dataset,
)
from medical_jiwer.cli import main
from medical_jiwer.statistics import _clusters


def corpus():
    return [
        BenchmarkSample(
            "short",
            "ramipril",
            "wrong",
            Annotations([EntityAnnotation("E1", "MEDICATION", 0, 1, "ramipril", "ramipril")]),
            {"accent_group": "non_native", "speaker_ids": ["a", "b"]},
        ),
        BenchmarkSample(
            "long",
            "der patient nimmt ramipril",
            "der patient nimmt ramipril",
            Annotations([EntityAnnotation("E1", "MEDICATION", 3, 4, "ramipril", "ramipril")]),
            {"accent_group": "native", "speaker_ids": ["b", "c"]},
        ),
        BenchmarkSample("empty", "", "", metadata={"speaker_ids": ["d"]}),
    ]


def test_micro_aggregation_and_arbitrary_grouping():
    report = ClinicalScorer().score_corpus(corpus(), group_by="accent_group")
    assert report.metrics.wer == pytest.approx(1 / 5)
    assert report.metrics.mt_wer == 0.5
    assert report.metrics.entity_f1 == 0.5
    assert report.metrics.per_entity_type["MEDICATION"].fn == 1
    assert report.by_group("accent_group")["non_native"].wer == 1
    assert report.by_group("accent_group")["native"].wer == 0
    assert "<missing>" in report.by_group("accent_group")
    assert '["a", "b"]' in report.by_group("speaker_ids")


def test_json_csv_roundtrip(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(
        "\n".join(json.dumps(sample.to_dict()) for sample in corpus()), encoding="utf-8"
    )
    report = ClinicalScorer().score_corpus(load_dataset(path), group_by="accent_group")
    output = tmp_path / "output.json"
    report.write_json(output, include_alignment=True)
    data = json.loads(output.read_text())
    assert data["metrics"]["wer"] == 0.2
    assert data["samples"][0]["metrics"]["alignment"]["reference_tokens"] == ["ramipril"]
    report.write_csv(tmp_path / "output.csv")
    with (tmp_path / "output.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["scope"] == "corpus"
    assert float(rows[0]["entity_f1"]) == 0.5
    assert len(rows) == 7


def test_json_dataset_container_versions(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "normalization_version": "de-clinical-asr-v1",
                "samples": [
                    corpus()[0].to_dict() | {"normalization_version": "de-clinical-asr-v1"}
                ],
            }
        )
    )
    assert ClinicalScorer().score_corpus(load_dataset(path)).metrics.wer == 1


def test_bootstrap_reuses_scores_and_is_reproducible():
    report = ClinicalScorer().score_corpus(corpus())
    first = bootstrap_ci(report, metric="entity_f1", seed=42, n_resamples=100)
    assert first == bootstrap_ci(report, metric="entity_f1", seed=42, n_resamples=100)
    assert first.estimate == report.metrics.entity_f1
    assert 0 <= first.lower <= first.upper <= 1
    assert _clusters(report, "speaker_ids") == [[0, 1], [2]]
    interval = bootstrap_ci(report, cluster_by="speaker_ids", seed=3, n_resamples=20)
    assert interval.cluster_by == "speaker_ids"


@pytest.mark.parametrize(
    "options",
    [
        {"metric": "invalid"},
        {"n_resamples": 0},
        {"n_resamples": True},
        {"confidence": 1},
    ],
)
def test_invalid_bootstrap_options(options):
    with pytest.raises(ValueError):
        bootstrap_ci(corpus(), **options)


def test_bootstrap_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_ci([])


def test_missing_cluster_metadata():
    with pytest.raises(ValueError, match="cluster metadata"):
        bootstrap_ci(corpus(), cluster_by="unknown")


def test_version_mismatch_and_duplicate_ids():
    samples = corpus()
    samples[0].normalization_version = "other-version"
    with pytest.raises(ValueError, match="normalization_version"):
        ClinicalScorer().score_corpus(samples)
    with pytest.raises(ValueError, match="unique"):
        ClinicalScorer().score_corpus([corpus()[0], corpus()[0]])


def test_cli_score_inspect_csv_and_errors(tmp_path, capsys):
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([sample.to_dict() for sample in corpus()]))
    assert main(["score", "--dataset", str(dataset), "--group-by", "accent_group"]) == 0
    assert json.loads(capsys.readouterr().out)["metrics"]["wer"] == 0.2
    assert main(["inspect", "--dataset", str(dataset), "--sample-id", "short"]) == 0
    inspection = capsys.readouterr().out
    assert "REF: ramipril" in inspection and "[substitute]" in inspection and "[FN]" in inspection
    assert main(["score", "--dataset", str(dataset), "--output", str(tmp_path / "out.csv")]) == 0
    assert (tmp_path / "out.csv").read_text().startswith("scope,")
    assert main(["inspect", "--dataset", str(dataset), "--sample-id", "missing"]) == 2
    assert "found 0" in capsys.readouterr().err


def test_jsonl_error_has_line_number(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('\n{"sample_id": "x", "reference": "", "hypothesis": "", "annotations": []}\n')
    with pytest.raises(ValueError, match=r"broken.jsonl:2:"):
        load_dataset(path)
