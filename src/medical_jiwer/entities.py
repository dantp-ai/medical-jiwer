"""Lightweight hypothesis entity recognition with a pluggable protocol."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import jiwer

from .normalize import normalize_tokens


@dataclass(frozen=True)
class DetectedEntity:
    start: int
    end: int
    surface: str
    canonical: str
    type: str = "OTHER_MEDICAL"
    concept_id: str | None = None


@dataclass(frozen=True)
class DictionaryEntry:
    surface: str
    canonical: str
    type: str = "OTHER_MEDICAL"
    concept_id: str | None = None


class MedicalRecognizer(Protocol):
    def extract(self, tokens: list[str]) -> list[DetectedEntity]: ...


class DictionaryRecognizer:
    """Deterministic longest-first, nonoverlapping dictionary matching.

    Entries may be strings or dictionaries with surface, canonical and type.
    Conflicting normalized surfaces are rejected rather than silently overwritten.
    No terminology is bundled: callers control the clinical vocabulary/version.
    """

    def __init__(
        self,
        entries: Iterable[str | dict | DictionaryEntry],
        *,
        asr_transform: jiwer.AbstractTransform | None = None,
    ):
        self.entries: dict[tuple[str, ...], DictionaryEntry] = {}
        for entry in entries:
            if isinstance(entry, str):
                entry = DictionaryEntry(entry, " ".join(normalize_tokens(entry, asr_transform)))
            elif isinstance(entry, dict):
                entry = DictionaryEntry(**entry)
            key = tuple(normalize_tokens(entry.surface, asr_transform))
            if not key or not entry.canonical or not entry.type:
                raise ValueError("Dictionary entries need nonempty surface, canonical and type")
            if key in self.entries and self.entries[key] != entry:
                previous = self.entries[key]
                if (previous.canonical, previous.type, previous.concept_id) != (
                    entry.canonical,
                    entry.type,
                    entry.concept_id,
                ):
                    raise ValueError(f"Conflicting dictionary entry: {' '.join(key)}")
            self.entries[key] = entry
        self._by_first: dict[str, list[tuple[str, ...]]] = {}
        for key in sorted(self.entries, key=lambda item: (-len(item), item)):
            self._by_first.setdefault(key[0], []).append(key)

    def extract(self, tokens: list[str]) -> list[DetectedEntity]:
        entities = []
        position = 0
        while position < len(tokens):
            for key in self._by_first.get(tokens[position], []):
                if tuple(tokens[position : position + len(key)]) == key:
                    entry = self.entries[key]
                    end = position + len(key)
                    entities.append(
                        DetectedEntity(
                            position,
                            end,
                            " ".join(tokens[position:end]),
                            entry.canonical,
                            entry.type,
                            entry.concept_id,
                        )
                    )
                    position = end
                    break
            else:
                position += 1
        return entities


class CompositeRecognizer:
    """Combine recognizers: longest spans, specific types, then recognizer order."""

    def __init__(self, *recognizers: MedicalRecognizer):
        self.recognizers = recognizers

    def extract(self, tokens: list[str]) -> list[DetectedEntity]:
        candidates = [
            entity for recognizer in self.recognizers for entity in recognizer.extract(tokens)
        ]
        selected: list[DetectedEntity] = []
        for entity in sorted(
            candidates,
            key=lambda item: (-(item.end - item.start), item.type == "OTHER_MEDICAL", item.start),
        ):
            if not any(entity.start < other.end and other.start < entity.end for other in selected):
                selected.append(entity)
        return sorted(selected, key=lambda item: item.start)
