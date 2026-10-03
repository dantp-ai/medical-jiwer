"""Fail-fast validation against the actual normalized reference tokens."""

from .normalize import normalize_tokens
from .schemas import Annotations


def validate_annotations(
    annotations: Annotations,
    reference_tokens: list[str],
    *,
    asr_transform=None,
    check_surfaces: bool = True,
    allow_overlapping_entities: bool = False,
) -> None:
    ids: set[str] = set()
    for item in [*annotations.entities, *annotations.critical_slots]:
        if not isinstance(item.id, str) or not item.id or item.id in ids:
            raise ValueError(f"Annotation ID must be nonempty and unique: {item.id!r}")
        ids.add(item.id)
        if not isinstance(item.type, str) or not item.type:
            raise ValueError(f"{item.id}: type must be a nonempty string")
        if (
            type(item.ref_start) is not int
            or type(item.ref_end) is not int
            or not 0 <= item.ref_start < item.ref_end <= len(reference_tokens)
        ):
            raise ValueError(f"{item.id}: span must satisfy 0 <= start < end <= token count")
        if not isinstance(item.surface, str):
            raise ValueError(f"{item.id}: surface must be text")
        if (
            check_surfaces
            and normalize_tokens(item.surface, asr_transform)
            != reference_tokens[item.ref_start : item.ref_end]
        ):
            raise ValueError(f"{item.id}: surface does not match normalized reference span")
    entity_ids = {entity.id for entity in annotations.entities}
    for entity in annotations.entities:
        if not isinstance(entity.canonical, str) or not entity.canonical:
            raise ValueError(f"{entity.id}: entity canonical must be a nonempty string")
    if not allow_overlapping_entities:
        ordered = sorted(annotations.entities, key=lambda entity: entity.ref_start)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.ref_start < previous.ref_end:
                raise ValueError(f"Overlapping entities: {previous.id}, {current.id}")
    for slot in annotations.critical_slots:
        if slot.anchor_entity_id not in entity_ids:
            raise ValueError(f"{slot.id}: unknown anchor entity {slot.anchor_entity_id!r}")
        if slot.canonical is None:
            raise ValueError(f"{slot.id}: canonical value is required")
        if slot.type == "NEGATION" and type(slot.canonical) is not bool:
            raise ValueError(f"{slot.id}: NEGATION canonical must be boolean")
        if slot.type == "LATERALITY" and slot.canonical not in ("left", "right", "bilateral"):
            raise ValueError(f"{slot.id}: invalid LATERALITY canonical")
        if slot.type == "DOSE":
            if (
                not isinstance(slot.canonical, dict)
                or not {"value", "unit"} <= slot.canonical.keys()
            ):
                raise ValueError(f"{slot.id}: {slot.type} requires a value/unit object")
