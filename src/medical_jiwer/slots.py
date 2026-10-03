"""Conservative German clinical canonicalization and anchored slot recognition."""

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from .entities import DetectedEntity, MedicalRecognizer

UNITS = {
    "mg": "mg",
    "milligramm": "mg",
    "milligramme": "mg",
    "g": "g",
    "gramm": "g",
    "μg": "ug",
    "µg": "ug",
    "ug": "ug",
    "mikrogramm": "ug",
    "ml": "ml",
    "milliliter": "ml",
    "l": "l",
    "liter": "l",
    "mmhg": "mmHg",
    "mmol/l": "mmol/l",
    "mg/dl": "mg/dl",
    "bpm": "bpm",
    "°c": "°C",
    "%": "%",
    "prozent": "%",
}
LATERALITY = {
    **dict.fromkeys(["links", "linke", "linker", "linkes", "linken", "linkem"], "left"),
    **dict.fromkeys(["rechts", "rechte", "rechter", "rechtes", "rechten", "rechtem"], "right"),
    "beidseits": "bilateral",
    "beidseitig": "bilateral",
    "bilateral": "bilateral",
}
NEGATIONS = {"kein", "keine", "keinen", "keiner", "keinem", "keines", "nicht", "ohne", "verneint"}
FREQUENCIES = {
    "einmal täglich": "1/day",
    "zweimal täglich": "2/day",
    "dreimal täglich": "3/day",
    "viermal täglich": "4/day",
    "morgens und abends": "2/day",
    "bei bedarf": "PRN",
    "morgens": "morning",
    "mittags": "noon",
    "abends": "evening",
    "nachts": "night",
}
SMALL_NUMBERS = {
    "null": 0,
    "ein": 1,
    "eins": 1,
    "eine": 1,
    "einen": 1,
    "zwei": 2,
    "drei": 3,
    "vier": 4,
    "fünf": 5,
    "sechs": 6,
    "sieben": 7,
    "acht": 8,
    "neun": 9,
    "zehn": 10,
    "elf": 11,
    "zwölf": 12,
    "dreizehn": 13,
    "vierzehn": 14,
    "fünfzehn": 15,
    "sechzehn": 16,
    "siebzehn": 17,
    "achtzehn": 18,
    "neunzehn": 19,
    "zwanzig": 20,
    "dreißig": 30,
    "dreissig": 30,
    "vierzig": 40,
    "fünfzig": 50,
    "sechzig": 60,
    "siebzig": 70,
    "achtzig": 80,
    "neunzig": 90,
}
NUMERIC_SLOT_TYPES = {"DOSE", "NUMERIC_VALUE", "LAB_VALUE", "VITAL_VALUE"}


def _german_integer(word: str) -> int | None:
    if word in SMALL_NUMBERS:
        return SMALL_NUMBERS[word]
    for magnitude, multiplier in [("tausend", 1000), ("hundert", 100)]:
        if magnitude in word:
            left, right = word.split(magnitude, 1)
            prefix = _german_integer(left) if left else 1
            suffix = _german_integer(right) if right else 0
            if (
                prefix is not None
                and suffix is not None
                and 0 < prefix < 10
                and suffix < multiplier
            ):
                return prefix * multiplier + suffix
            return None
    if "und" in word:
        left, right = word.split("und", 1)
        if left in SMALL_NUMBERS and right in SMALL_NUMBERS:
            ones, tens = SMALL_NUMBERS[left], SMALL_NUMBERS[right]
            if 0 < ones < 10 and tens >= 20 and tens % 10 == 0:
                return ones + tens
    return None


def parse_number(text: str) -> Decimal | None:
    """Parse exact decimal digits or deterministic German number words."""
    text = text.strip().lower()
    if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", text):
        return Decimal(text.replace(",", "."))
    if " komma " in text:
        whole, fraction = text.split(" komma ", 1)
        integer = _german_integer(whole.replace(" ", ""))
        digits = [SMALL_NUMBERS.get(token) for token in fraction.split()]
        if integer is not None and digits and all(d is not None and d < 10 for d in digits):
            return Decimal(f"{integer}.{''.join(str(d) for d in digits)}")
        return None
    integer = _german_integer(text.replace(" ", ""))
    return Decimal(integer) if integer is not None else None


def _typed_number(value: Any) -> Decimal | str:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a clinical numeric value")
    if isinstance(value, str) and re.fullmatch(r"\d+/\d+", value):
        return "/".join(str(int(part)) for part in value.split("/"))
    try:
        number = Decimal(str(value).replace(",", "."))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid clinical numeric value: {value!r}") from exc
    if not number.is_finite():
        raise ValueError("Clinical numeric values must be finite")
    return number


class ClinicalCanonicalizer(Protocol):
    def canonicalize_entity(
        self, tokens: list[str], entity_type: str, recognizer: MedicalRecognizer
    ) -> str | None: ...
    def canonicalize_slot(self, tokens: list[str], slot_type: str) -> Any: ...
    def normalize_slot_value(self, value: Any, slot_type: str) -> Any: ...


class GermanClinicalCanonicalizer:
    """Exact semantic equivalences only; no fuzzy entity or numeric matching."""

    def canonicalize_entity(
        self, tokens: list[str], entity_type: str, recognizer: MedicalRecognizer
    ) -> str | None:
        for entity in recognizer.extract(tokens):
            if entity.start == 0 and entity.end == len(tokens) and entity.type == entity_type:
                return entity.canonical
        return None

    def canonicalize_slot(self, tokens: list[str], slot_type: str) -> Any:
        text = " ".join(tokens)
        if slot_type == "NEGATION":
            return any(token in NEGATIONS for token in tokens)
        if slot_type == "LATERALITY":
            values = {LATERALITY[token] for token in tokens if token in LATERALITY}
            return next(iter(values)) if len(values) == 1 else None
        if slot_type == "FREQUENCY":
            if text in FREQUENCIES:
                return FREQUENCIES[text]
            match = re.fullmatch(r"alle (.+) stunden", text)
            if match:
                number = parse_number(match[1])
                if number is not None and number > 0:
                    return f"every_{number.normalize():f}h"
            return None
        if slot_type == "UNIT":
            return UNITS.get(text)
        if slot_type in NUMERIC_SLOT_TYPES:
            if len(tokens) >= 2 and tokens[-1] in UNITS:
                numeric_text = " ".join(tokens[:-1])
                number = parse_number(numeric_text)
                if number is None and re.fullmatch(r"\d+/\d+", numeric_text):
                    number = numeric_text
                if number is not None:
                    return self.normalize_slot_value(
                        {"value": number, "unit": UNITS[tokens[-1]]}, slot_type
                    )
            if slot_type != "DOSE":
                number = parse_number(text)
                if number is None and re.fullmatch(r"\d+/\d+", text):
                    number = text
                if number is not None:
                    return self.normalize_slot_value(number, slot_type)
            return None
        # Configurable types can still use exact normalized text, or a custom adapter.
        return text if text else None

    def normalize_slot_value(self, value: Any, slot_type: str) -> Any:
        if slot_type == "NEGATION":
            if type(value) is not bool:
                raise ValueError("NEGATION canonical must be boolean")
            return value
        if slot_type == "LATERALITY":
            if value not in ("left", "right", "bilateral"):
                raise ValueError("Invalid LATERALITY canonical")
            return value
        if slot_type in NUMERIC_SLOT_TYPES:
            if isinstance(value, dict):
                if set(value) != {"value", "unit"}:
                    raise ValueError(f"{slot_type} requires exactly value and unit")
                unit = UNITS.get(str(value["unit"]).lower())
                if unit is None:
                    raise ValueError(f"Unsupported clinical unit: {value['unit']!r}")
                return {"value": _typed_number(value["value"]), "unit": unit}
            if slot_type == "DOSE":
                raise ValueError("DOSE requires a value/unit object")
            return _typed_number(value)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{slot_type} canonical must be a nonempty string")
        return value


@dataclass(frozen=True)
class DetectedSlot:
    start: int
    end: int
    anchor_start: int
    anchor_end: int
    type: str
    canonical: Any
    surface: str


class SlotRecognizer(Protocol):
    def extract(self, tokens: list[str], entities: list[DetectedEntity]) -> list[DetectedSlot]: ...


class RuleBasedSlotRecognizer:
    """Extract nearby explicit slots. Scope is bounded and intentionally simple.

    Dose/frequency attaches to a preceding entity; negation/laterality prefers a
    following entity. Numeric values require a nearby recognized clinical anchor.
    This does not implement full German grammar or complex negation scope.
    """

    def __init__(
        self, canonicalizer: ClinicalCanonicalizer | None = None, *, max_distance: int = 6
    ):
        self.canonicalizer = canonicalizer or GermanClinicalCanonicalizer()
        self.max_distance = max_distance

    def _anchor(self, entities, start, end, *, prefix=False):
        before = [e for e in entities if e.end <= start and start - e.end <= self.max_distance]
        after = [e for e in entities if e.start >= end and e.start - end <= (2 if prefix else 1)]
        if prefix and after:
            return min(after, key=lambda entity: entity.start)
        if before:
            return max(before, key=lambda entity: entity.end)
        return min(after, key=lambda entity: entity.start) if after else None

    def extract(self, tokens: list[str], entities: list[DetectedEntity]) -> list[DetectedSlot]:
        result: list[DetectedSlot] = []
        occupied_entities = {i for e in entities for i in range(e.start, e.end)}
        position = 0
        while position < len(tokens):
            if position in occupied_entities:
                position += 1
                continue
            token = tokens[position]
            candidates: list[tuple[int, str, Any]] = []
            if token in NEGATIONS:
                candidates.append((position + 1, "NEGATION", True))
            if token in LATERALITY:
                candidates.append((position + 1, "LATERALITY", LATERALITY[token]))
            for length in range(min(6, len(tokens) - position), 0, -1):
                end = position + length
                if any(i in occupied_entities for i in range(position, end)):
                    continue
                span = tokens[position:end]
                frequency = self.canonicalizer.canonicalize_slot(span, "FREQUENCY")
                if frequency is not None:
                    candidates.append((end, "FREQUENCY", frequency))
                    break
            for length in range(min(6, len(tokens) - position), 0, -1):
                end = position + length
                if any(i in occupied_entities for i in range(position, end)):
                    continue
                value = self.canonicalizer.canonicalize_slot(tokens[position:end], "NUMERIC_VALUE")
                if value is not None:
                    candidates.append((end, "NUMERIC_VALUE", value))
                    break
            for end, slot_type, value in candidates:
                anchor = self._anchor(
                    entities, position, end, prefix=slot_type in {"NEGATION", "LATERALITY"}
                )
                if anchor is None:
                    continue
                if slot_type == "NUMERIC_VALUE":
                    if (
                        anchor.type == "MEDICATION"
                        and isinstance(value, dict)
                        and value["unit"] in {"mg", "g", "ug", "ml", "l"}
                    ):
                        slot_type = "DOSE"
                    elif anchor.type == "LAB_TEST":
                        slot_type = "LAB_VALUE"
                    elif anchor.type == "CLINICAL_FINDING":
                        slot_type = "VITAL_VALUE"
                result.append(
                    DetectedSlot(
                        position,
                        end,
                        anchor.start,
                        anchor.end,
                        slot_type,
                        value,
                        " ".join(tokens[position:end]),
                    )
                )
                position = end
                break
            else:
                position += 1
        return result


def json_value(value: Any) -> Any:
    """Render canonical Decimal values as finite JSON numbers for audit output."""
    if isinstance(value, Decimal):
        number = int(value) if value == value.to_integral_value() else float(value)
        if not math.isfinite(number):
            raise ValueError("Canonical value exceeds JSON number range")
        return number
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    return value
