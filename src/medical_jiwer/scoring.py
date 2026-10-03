"""High-level clinical scoring built exclusively on JiWER word alignment."""

from typing import TYPE_CHECKING, Any

import jiwer

from .align import align, project_reference_span_to_hypothesis
from .annotations import validate_annotations
from .entities import CompositeRecognizer, DictionaryEntry, DictionaryRecognizer, MedicalRecognizer
from .metrics import ClinicalASRResult, count_event
from .normalize import NORMALIZATION_VERSION, default_asr_transform
from .schemas import SCHEMA_VERSION, Annotations
from .slots import (
    ClinicalCanonicalizer,
    GermanClinicalCanonicalizer,
    RuleBasedSlotRecognizer,
    SlotRecognizer,
    json_value,
)

if TYPE_CHECKING:
    from .report import CorpusReport


class ClinicalScorer:
    def __init__(
        self,
        *,
        asr_transform: jiwer.AbstractTransform | None = None,
        medical_recognizer: MedicalRecognizer | None = None,
        clinical_canonicalizer: ClinicalCanonicalizer | None = None,
        slot_recognizer: SlotRecognizer | None = None,
        normalization_version: str | None = None,
        terminology_version: str | None = None,
        annotation_guideline_version: str | None = None,
        require_entity_type: bool = True,
        check_surfaces: bool = True,
        allow_overlapping_entities: bool = False,
    ):
        if asr_transform is not None and not normalization_version:
            raise ValueError("A custom ASR transform requires an explicit normalization_version")
        self.asr_transform = asr_transform or default_asr_transform()
        self.medical_recognizer = medical_recognizer
        self.clinical_canonicalizer = clinical_canonicalizer or GermanClinicalCanonicalizer()
        self.slot_recognizer = slot_recognizer or RuleBasedSlotRecognizer(
            self.clinical_canonicalizer
        )
        self.normalization_version = normalization_version or NORMALIZATION_VERSION
        self.terminology_version = terminology_version
        self.annotation_guideline_version = annotation_guideline_version
        self.require_entity_type = require_entity_type
        self.check_surfaces = check_surfaces
        self.allow_overlapping_entities = allow_overlapping_entities

    def score(
        self, reference: str, hypothesis: str, annotations: Annotations | dict | None = None
    ) -> ClinicalASRResult:
        if not isinstance(reference, str) or not isinstance(hypothesis, str):
            raise ValueError("Reference and hypothesis must be strings")
        if annotations is None:
            annotations = Annotations()
        elif isinstance(annotations, dict):
            annotations = Annotations.from_dict(annotations)
        if not isinstance(annotations, Annotations):
            raise ValueError("annotations must be an Annotations object or dictionary")
        alignment = align(reference, hypothesis, self.asr_transform)
        validate_annotations(
            annotations,
            alignment.reference_tokens,
            asr_transform=self.asr_transform,
            check_surfaces=self.check_surfaces,
            allow_overlapping_entities=self.allow_overlapping_entities,
        )
        for slot in annotations.critical_slots:
            try:
                self.clinical_canonicalizer.normalize_slot_value(slot.canonical, slot.type)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{slot.id}: invalid canonical value: {exc}") from exc
        # Exact annotated surfaces remain recognizable even without terminology.
        # The supplied recognizer takes precedence when its spans overlap these.
        fallback = DictionaryRecognizer(
            [
                DictionaryEntry(
                    " ".join(alignment.reference_tokens[entity.ref_start : entity.ref_end]),
                    entity.canonical,
                    entity.type,
                    entity.concept_id,
                )
                for entity in annotations.entities
            ],
            # These surfaces are already normalized; do not apply a custom
            # representation transform for a second time.
            asr_transform=jiwer.ReduceToListOfListOfWords(),
        )
        recognizer = (
            CompositeRecognizer(self.medical_recognizer, fallback)
            if self.medical_recognizer
            else fallback
        )
        detected = recognizer.extract(alignment.hypothesis_tokens)
        for entity in detected:
            if not 0 <= entity.start < entity.end <= len(alignment.hypothesis_tokens):
                raise ValueError(f"Medical recognizer returned invalid span: {entity}")
        medical_refs = {
            index
            for entity in annotations.entities
            for index in range(entity.ref_start, entity.ref_end)
        }
        medical_hyps = {index for entity in detected for index in range(entity.start, entity.end)}
        result = ClinicalASRResult(
            substitutions=alignment.substitutions,
            deletions=alignment.deletions,
            insertions=alignment.insertions,
            hits=alignment.hits,
            reference_word_count=alignment.reference_word_count,
            medical_reference_word_count=len(medical_refs),
            alignment=alignment,
            versions={
                "schema_version": SCHEMA_VERSION,
                "normalization_version": self.normalization_version,
                "terminology_version": self.terminology_version,
                "annotation_guideline_version": self.annotation_guideline_version,
            },
        )
        for row in alignment.token_alignments:
            if row.ref_index in medical_refs:
                result.medical_substitutions += row.operation == "substitute"
                result.medical_deletions += row.operation == "delete"
            if row.operation == "insert" and row.hyp_index in medical_hyps:
                result.medical_insertions += 1
        anchor_matches = self._score_entities(result, annotations, recognizer, detected)
        self._score_slots(result, annotations, detected, anchor_matches)
        return result

    def _score_entities(self, result, annotations, recognizer, detected):
        alignment = result.alignment
        consumed = set()
        anchors = {}
        for entity in annotations.entities:
            projected = project_reference_span_to_hypothesis(
                alignment, entity.ref_start, entity.ref_end
            )
            candidates = [
                (i, hyp)
                for i, hyp in enumerate(detected)
                if i not in consumed
                and projected.hypothesis_start is not None
                and hyp.start <= projected.hypothesis_start
                and projected.hypothesis_end <= hyp.end
                and all(
                    alignment.hyp_to_ref[h] is None
                    or entity.ref_start <= alignment.hyp_to_ref[h] < entity.ref_end
                    for h in range(hyp.start, hyp.end)
                )
            ]
            # Expand only across inserted tokens belonging to one recognized
            # occurrence. A multiword alias must not consume neighboring references.
            selected = next(
                (
                    pair
                    for pair in candidates
                    if pair[1].canonical == entity.canonical
                    and (not self.require_entity_type or pair[1].type == entity.type)
                ),
                candidates[0] if candidates else None,
            )
            hyp_tokens = projected.hypothesis_tokens
            start, end = projected.hypothesis_start, projected.hypothesis_end
            if selected:
                _, hyp = selected
                start, end = hyp.start, hyp.end
                hyp_tokens = alignment.hypothesis_tokens[start:end]
            canonical = self.clinical_canonicalizer.canonicalize_entity(
                hyp_tokens, entity.type, recognizer
            )
            if not self.require_entity_type and selected:
                canonical = selected[1].canonical
            correct = canonical == entity.canonical
            if correct and selected:
                consumed.add(selected[0])
                anchors[entity.id] = selected[1]
            elif correct:
                if any(
                    i in consumed and hyp.start == start and hyp.end == end
                    for i, hyp in enumerate(detected)
                ):
                    correct = False
            if correct and not selected:
                # A custom canonicalizer may recognize an alias not in the dictionary.
                from .entities import DetectedEntity

                anchors[entity.id] = DetectedEntity(
                    start,
                    end,
                    " ".join(hyp_tokens),
                    entity.canonical,
                    entity.type,
                )
            count_event(
                result.entities, result.per_entity_type, entity.type, "tp" if correct else "fn"
            )
            # A changed, nonempty local span is a replacement even when terminology
            # cannot classify it. Known replacements are counted by the second pass.
            overlap = any(
                projected.hypothesis_start is not None
                and hyp.start < projected.hypothesis_end
                and projected.hypothesis_start < hyp.end
                for hyp in detected
            )
            if not correct and projected.hypothesis_tokens and not overlap:
                count_event(result.entities, result.per_entity_type, entity.type, "fp")
            result.entity_matches.append(
                {
                    "reference_id": entity.id,
                    "type": entity.type,
                    "reference": entity.surface,
                    "hypothesis": " ".join(hyp_tokens),
                    "reference_canonical": entity.canonical,
                    "hypothesis_canonical": canonical,
                    "status": "TP" if correct else "FN",
                    "hypothesis_start": start,
                    "hypothesis_end": end,
                }
            )
        for i, hyp in enumerate(detected):
            if i not in consumed:
                count_event(result.entities, result.per_entity_type, hyp.type, "fp")
                result.entity_matches.append(
                    {
                        "reference_id": None,
                        "type": hyp.type,
                        "hypothesis": hyp.surface,
                        "hypothesis_canonical": hyp.canonical,
                        "status": "FP",
                        "hypothesis_start": hyp.start,
                        "hypothesis_end": hyp.end,
                    }
                )
        return anchors

    def _score_slots(self, result, annotations, detected, anchors):
        alignment = result.alignment
        # Include custom-canonicalized anchors for local deterministic extraction.
        slot_entities = list(detected)
        for anchor in anchors.values():
            if not any(e.start == anchor.start and e.end == anchor.end for e in slot_entities):
                slot_entities.append(anchor)
        hypothesis_slots = self.slot_recognizer.extract(alignment.hypothesis_tokens, slot_entities)
        for slot in hypothesis_slots:
            if not (
                0 <= slot.start < slot.end <= len(alignment.hypothesis_tokens)
                and any(
                    entity.start == slot.anchor_start and entity.end == slot.anchor_end
                    for entity in slot_entities
                )
            ):
                raise ValueError(f"Slot recognizer returned invalid span or anchor: {slot}")
            self.clinical_canonicalizer.normalize_slot_value(slot.canonical, slot.type)
        consumed = set()
        reference_entities = {entity.id: entity for entity in annotations.entities}
        numeric_types = {"DOSE", "NUMERIC_VALUE", "LAB_VALUE", "VITAL_VALUE"}
        for slot in annotations.critical_slots:
            projected = project_reference_span_to_hypothesis(
                alignment, slot.ref_start, slot.ref_end
            )
            ref_anchor = reference_entities[slot.anchor_entity_id]
            projected_anchor = project_reference_span_to_hypothesis(
                alignment, ref_anchor.ref_start, ref_anchor.ref_end
            )
            recovered_anchor = anchors.get(slot.anchor_entity_id)
            anchor_start = (
                recovered_anchor.start if recovered_anchor else projected_anchor.hypothesis_start
            )
            anchor_end = (
                recovered_anchor.end if recovered_anchor else projected_anchor.hypothesis_end
            )
            candidates = [
                (i, hyp)
                for i, hyp in enumerate(hypothesis_slots)
                if i not in consumed
                and hyp.anchor_start == anchor_start
                and hyp.anchor_end == anchor_end
                and (
                    hyp.type == slot.type
                    or hyp.type in numeric_types
                    and slot.type in numeric_types
                )
                and (
                    slot.type == "NEGATION"
                    and projected.hypothesis_start is None
                    or projected.hypothesis_start is not None
                    and hyp.start < projected.hypothesis_end
                    and projected.hypothesis_start < hyp.end
                )
            ]
            candidate = candidates[0] if candidates else None
            hyp_tokens = projected.hypothesis_tokens
            if candidate:
                i, hyp = candidate
                consumed.add(i)
                hyp_tokens = alignment.hypothesis_tokens[hyp.start : hyp.end]
            if candidate:
                try:
                    hyp_value = self.clinical_canonicalizer.normalize_slot_value(
                        hyp.canonical, slot.type
                    )
                except ValueError:
                    # A numeric value without a unit cannot reproduce a DOSE.
                    hyp_value = None
            else:
                hyp_value = self.clinical_canonicalizer.canonicalize_slot(hyp_tokens, slot.type)
            expected = self.clinical_canonicalizer.normalize_slot_value(slot.canonical, slot.type)
            anchor_correct = slot.anchor_entity_id in anchors
            assigned_elsewhere = not candidate and any(
                projected.hypothesis_start is not None
                and hyp.start < projected.hypothesis_end
                and projected.hypothesis_start < hyp.end
                and (
                    hyp.type == slot.type
                    or hyp.type in numeric_types
                    and slot.type in numeric_types
                )
                for hyp in hypothesis_slots
            )
            # Missing explicit negation on a recovered anchor means affirmative.
            present = bool(hyp_tokens) or (slot.type == "NEGATION" and anchor_correct)
            correct = (
                anchor_correct
                and not assigned_elsewhere
                and present
                and hyp_value is not None
                and hyp_value == expected
            )
            count_event(result.critical, result.per_slot_type, slot.type, "tp" if correct else "fn")
            if (
                not correct
                and present
                and not assigned_elsewhere
                and (hyp_value is not None or candidate is not None)
            ):
                count_event(result.critical, result.per_slot_type, slot.type, "fp")
            result.critical_matches.append(
                {
                    "reference_id": slot.id,
                    "anchor_entity_id": slot.anchor_entity_id,
                    "type": slot.type,
                    "reference": slot.surface,
                    "hypothesis": " ".join(hyp_tokens),
                    "reference_canonical": json_value(expected),
                    "hypothesis_canonical": json_value(hyp_value),
                    "anchor_correct": anchor_correct,
                    "status": "TP" if correct else "WRONG" if present else "FN",
                }
            )
        for i, hyp in enumerate(hypothesis_slots):
            if i not in consumed:
                count_event(result.critical, result.per_slot_type, hyp.type, "fp")
                result.critical_matches.append(
                    {
                        "reference_id": None,
                        "type": hyp.type,
                        "hypothesis": hyp.surface,
                        "hypothesis_canonical": json_value(hyp.canonical),
                        "status": "FP",
                        "anchor_start": hyp.anchor_start,
                        "anchor_end": hyp.anchor_end,
                    }
                )

    score_sample = score

    def score_corpus(self, samples, *, group_by: str | None = None) -> "CorpusReport":
        from .report import CorpusReport

        return CorpusReport.from_samples(self, samples, group_by=group_by)


def score_sample(
    reference: str,
    hypothesis: str,
    annotations: Annotations | dict | None = None,
    **scorer_options: Any,
) -> ClinicalASRResult:
    return ClinicalScorer(**scorer_options).score(reference, hypothesis, annotations)
