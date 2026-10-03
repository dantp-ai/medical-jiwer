"""Versioned ASR normalization, deliberately separate from clinical semantics."""

import re
import unicodedata

import jiwer

NORMALIZATION_VERSION = "de-clinical-asr-v1"


class ClinicalTextNormalization(jiwer.AbstractTransform):
    """Lowercase, normalize Unicode and punctuation, preserving clinical numbers.

    Decimal commas/points, numeric ratios, unit slashes, percent and degree signs
    survive. Number words and unit names are NOT semantically canonicalized here.
    The output is text; :func:`default_asr_transform` also tokenizes it.
    """

    def process_string(self, s: str) -> str:
        s = unicodedata.normalize("NFKC", s).lower()
        chars = []
        for i, char in enumerate(s):
            numeric_separator = (
                char in ".,:"
                and i > 0
                and i + 1 < len(s)
                and s[i - 1].isdigit()
                and s[i + 1].isdigit()
            )
            numeric_sign = (
                char == "-"
                and i + 1 < len(s)
                and s[i + 1].isdigit()
                and (i == 0 or s[i - 1].isspace())
            )
            if unicodedata.category(char).startswith("P") and not (
                numeric_separator or numeric_sign or char == "/"
            ):
                chars.append(" ")
            else:
                chars.append(char)
        return re.sub(r"\s+", " ", "".join(chars)).strip()


def default_asr_transform() -> jiwer.Compose:
    return jiwer.Compose([ClinicalTextNormalization(), jiwer.ReduceToListOfListOfWords()])


def normalize_tokens(text: str, transform: jiwer.AbstractTransform | None = None) -> list[str]:
    """Return exactly the token sequence used by the scorer's JiWER transform."""
    result = (transform or default_asr_transform())([text])
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], list):
        raise ValueError("ASR transform must produce one list of tokens per input string")
    if not all(isinstance(token, str) and token for token in result[0]):
        raise ValueError("ASR transform must produce nonempty string tokens")
    return result[0]
