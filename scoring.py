from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


DEFAULT_STYLE_WEIGHTS = {
    "function_words": 0.30,
    "pos_ngrams": 0.25,
    "rhythm": 0.15,
    "punctuation_formatting": 0.15,
    "markers_stance": 0.10,
    "masked_chargrams": 0.05,
    "raw_chargrams": 0.00,
}

LOOSE_STYLE_WEIGHTS = {
    "function_words": 0.28,
    "pos_ngrams": 0.22,
    "rhythm": 0.15,
    "punctuation_formatting": 0.15,
    "markers_stance": 0.10,
    "masked_chargrams": 0.05,
    "raw_chargrams": 0.05,
}


@dataclass
class FinalScore:
    score_1000: int
    percent: float
    overall_0_1: float


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        total = 1.0
    return {k: max(v, 0.0) / total for k, v in weights.items()}


def to_score(overall_0_1: float, exact_match: bool) -> FinalScore:
    val = 1.0 if exact_match else max(0.0, min(1.0, overall_0_1))
    score_1000 = int(round(1 + 999 * val))
    percent = round(100.0 * val, 2)
    return FinalScore(score_1000=score_1000, percent=percent, overall_0_1=val)
