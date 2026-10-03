"""JSON/JSONL scoring and single-sample alignment inspection."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .entities import DictionaryRecognizer
from .report import load_dataset, render_clinical_alignment
from .scoring import ClinicalScorer


def _scorer(args) -> ClinicalScorer:
    recognizer = None
    terminology_version = args.terminology_version
    if args.terminology:
        terminology = json.loads(Path(args.terminology).read_text(encoding="utf-8"))
        if isinstance(terminology, dict):
            embedded_version = terminology.get("terminology_version")
            if terminology_version and embedded_version and terminology_version != embedded_version:
                raise ValueError("CLI terminology version does not match terminology file")
            terminology_version = terminology_version or embedded_version
            terminology = terminology["entries"]
        recognizer = DictionaryRecognizer(terminology)
    return ClinicalScorer(
        medical_recognizer=recognizer,
        terminology_version=terminology_version,
        annotation_guideline_version=args.annotation_guideline_version,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate clinical ASR using JiWER alignment")
    parser.add_argument("--version", action="version", version=f"medical-jiwer {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("score", "inspect"):
        sub = commands.add_parser(command)
        sub.add_argument("--dataset", required=True, help="JSON or JSONL reference annotations")
        sub.add_argument(
            "--terminology", help="JSON dictionary entries or a versioned entries object"
        )
        sub.add_argument("--terminology-version")
        sub.add_argument("--annotation-guideline-version")
        if command == "score":
            sub.add_argument("--output", help="Output path; omit for JSON on stdout")
            sub.add_argument(
                "--format", choices=["json", "csv"], help="Default: infer from output suffix"
            )
            sub.add_argument(
                "--group-by", action="append", default=[], help="Metadata field; repeatable"
            )
            sub.add_argument("--include-alignment", action="store_true")
        else:
            sub.add_argument("--sample-id", required=True)
    args = parser.parse_args(argv)
    try:
        samples = load_dataset(args.dataset)
        scorer = _scorer(args)
        if args.command == "inspect":
            selected = [sample for sample in samples if sample.sample_id == args.sample_id]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one sample with ID {args.sample_id!r}; found {len(selected)}"
                )
            result = scorer.score_corpus(selected).results[0]
            print(render_clinical_alignment(result))
        else:
            report = scorer.score_corpus(samples)
            for field in args.group_by:
                report.by_group(field)
            output_format = args.format or (
                "csv" if args.output and Path(args.output).suffix == ".csv" else "json"
            )
            if output_format == "csv" and not args.output:
                raise ValueError("CSV output requires --output")
            if args.output:
                if output_format == "csv":
                    report.write_csv(args.output)
                else:
                    report.write_json(args.output, include_alignment=args.include_alignment)
            else:
                print(
                    json.dumps(
                        report.to_dict(include_alignment=args.include_alignment),
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )
                )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"medical-jiwer: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
