"""JiWER alignment and occurrence-preserving reference span projection."""

from dataclasses import dataclass

import jiwer

from .normalize import default_asr_transform


@dataclass(frozen=True)
class TokenAlignment:
    ref_index: int | None
    hyp_index: int | None
    operation: str
    # Insertion boundary in reference coordinates; otherwise the reference index.
    ref_boundary: int


@dataclass
class AlignmentResult:
    reference_tokens: list[str]
    hypothesis_tokens: list[str]
    chunks: list[jiwer.AlignmentChunk]
    token_alignments: list[TokenAlignment]
    ref_to_hyp: dict[int, int | None]
    hyp_to_ref: dict[int, int | None]
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int

    @property
    def reference_word_count(self) -> int:
        return len(self.reference_tokens)


@dataclass
class ProjectedSpan:
    hypothesis_start: int | None
    hypothesis_end: int | None
    hypothesis_tokens: list[str]
    deleted_reference_tokens: list[str]
    inserted_hypothesis_tokens: list[str]


def align(
    reference: str, hypothesis: str, asr_transform: jiwer.AbstractTransform | None = None
) -> AlignmentResult:
    transform = asr_transform or default_asr_transform()
    output = jiwer.process_words(
        reference, hypothesis, reference_transform=transform, hypothesis_transform=transform
    )
    rows = []
    for chunk in output.alignments[0]:
        refs = range(chunk.ref_start_idx, chunk.ref_end_idx)
        hyps = range(chunk.hyp_start_idx, chunk.hyp_end_idx)
        if chunk.type in {"equal", "substitute"}:
            # JiWER/RapidFuzz splits unequal replacement blocks into separate
            # substitution and insertion/deletion chunks, so each pair is exact.
            for r, h in zip(refs, hyps, strict=True):
                rows.append(TokenAlignment(r, h, chunk.type, r))
        elif chunk.type == "delete":
            rows.extend(TokenAlignment(r, None, "delete", r) for r in refs)
        elif chunk.type == "insert":
            rows.extend(TokenAlignment(None, h, "insert", chunk.ref_start_idx) for h in hyps)
    return AlignmentResult(
        reference_tokens=output.references[0],
        hypothesis_tokens=output.hypotheses[0],
        chunks=output.alignments[0],
        token_alignments=rows,
        ref_to_hyp={row.ref_index: row.hyp_index for row in rows if row.ref_index is not None},
        hyp_to_ref={row.hyp_index: row.ref_index for row in rows if row.hyp_index is not None},
        wer=output.wer,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
    )


def project_reference_span_to_hypothesis(
    alignment: AlignmentResult, ref_start: int, ref_end: int
) -> ProjectedSpan:
    """Project a half-open span, including only strictly internal insertions.

    Boundary insertions belong to neither adjacent span. A wholly deleted span
    has no hypothesis coordinates. This policy prevents an adjacent hallucinated
    entity or dose from becoming part of a correctly transcribed annotation.
    """
    if not 0 <= ref_start < ref_end <= len(alignment.reference_tokens):
        raise ValueError(f"Invalid reference span [{ref_start}, {ref_end})")
    indices = []
    deleted = []
    inserted = []
    for row in alignment.token_alignments:
        if row.ref_index is not None and ref_start <= row.ref_index < ref_end:
            if row.hyp_index is None:
                deleted.append(alignment.reference_tokens[row.ref_index])
            else:
                indices.append(row.hyp_index)
        elif row.operation == "insert" and ref_start < row.ref_boundary < ref_end:
            assert row.hyp_index is not None
            indices.append(row.hyp_index)
            inserted.append(alignment.hypothesis_tokens[row.hyp_index])
    start = min(indices) if indices else None
    end = max(indices) + 1 if indices else None
    return ProjectedSpan(
        start, end, alignment.hypothesis_tokens[start:end] if indices else [], deleted, inserted
    )
