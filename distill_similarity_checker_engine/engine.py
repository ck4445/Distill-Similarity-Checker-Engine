from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

from features_style import StyleResult, compute_style_similarity
from preprocess import PreprocessOptions, preprocess_text
from scoring import DEFAULT_STYLE_WEIGHTS, LOOSE_STYLE_WEIGHTS, normalize_weights, to_score


@dataclass
class EngineCompareOptions:
    mode: str = "Strict Style"
    bootstrap_runs: int = 100
    custom_weights: Optional[Dict[str, float]] = None
    preprocess: PreprocessOptions = field(default_factory=PreprocessOptions)


class SimilarityEngine:
    def __init__(self, enable_cache: bool = True, cache_size: int = 128) -> None:
        self.enable_cache = enable_cache
        self.cache_size = max(cache_size, 1)
        self._cache: Dict[str, Dict[str, object]] = {}

    def _cache_get(self, key: str) -> Optional[Dict[str, object]]:
        return self._cache.get(key)

    def _cache_put(self, key: str, value: Dict[str, object]) -> None:
        self._cache[key] = value
        if len(self._cache) > self.cache_size:
            oldest_key = next(iter(self._cache.keys()))
            self._cache.pop(oldest_key, None)

    def _resolve_weights(self, mode: str, custom_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        if custom_weights:
            return normalize_weights(custom_weights)
        if mode == "Loose Style":
            return normalize_weights(LOOSE_STYLE_WEIGHTS)
        return normalize_weights(DEFAULT_STYLE_WEIGHTS)

    def _build_key(self, text_a: str, text_b: str, options: EngineCompareOptions, weights: Dict[str, float]) -> str:
        payload = {
            "text_a": text_a,
            "text_b": text_b,
            "mode": options.mode,
            "bootstrap_runs": options.bootstrap_runs,
            "weights": weights,
            "preprocess": asdict(options.preprocess),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def compare_texts(
        self,
        text_a: str,
        text_b: str,
        options: Optional[EngineCompareOptions] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, object]:
        settings = options or EngineCompareOptions()
        weights = self._resolve_weights(settings.mode, settings.custom_weights)
        key = self._build_key(text_a, text_b, settings, weights)

        if self.enable_cache:
            cached = self._cache_get(key)
            if cached is not None:
                if progress_callback:
                    progress_callback(1.0, "Loaded from engine cache")
                return cached

        a = preprocess_text(text_a, settings.preprocess)
        b = preprocess_text(text_b, settings.preprocess)
        style: StyleResult = compute_style_similarity(
            a=a,
            b=b,
            mode=settings.mode,
            weights=weights,
            ignore_numbers=settings.preprocess.ignore_numbers,
            bootstrap_runs=settings.bootstrap_runs,
            progress_callback=progress_callback,
        )

        exact_match = a.normalized_text == b.normalized_text
        final = to_score(style.overall, exact_match=exact_match)
        result = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "settings": {
                "mode": settings.mode,
                "bootstrap_runs": settings.bootstrap_runs,
                "weights": weights,
                **asdict(settings.preprocess),
            },
            "final": {
                "similarity_0_1": round(final.overall_0_1, 6),
                "score_1_1000": final.score_1000,
                "percent": final.percent,
                "stability_band_percent": style.confidence_band,
                "topic_leakage": {
                    "label": style.leakage_label,
                    "content_overlap_percent": round(100 * style.topic_leakage_overlap, 2),
                },
            },
            "sub_scores_percent": {
                "function_words": round(100 * style.family_scores["function_words"], 2),
                "pos_ngrams": round(100 * style.family_scores["pos_ngrams"], 2),
                "rhythm": round(100 * style.family_scores["rhythm"], 2),
                "punctuation_formatting": round(100 * style.family_scores["punctuation_formatting"], 2),
                "markers_stance": round(100 * style.family_scores["markers_stance"], 2),
                "masked_chargrams": round(100 * style.family_scores["masked_chargrams"], 2),
                "raw_chargrams": round(100 * style.family_scores["raw_chargrams"], 2),
            },
            "explainability": {
                "top_function_word_deltas": style.function_word_deltas,
                "rhythm_stats": style.rhythm_stats,
                "punctuation_rates": style.punctuation_rates,
                "marker_rates": style.marker_rates,
                "masked_preview": style.masked_preview,
                "short_text_warning": style.short_text_warning,
            },
        }

        if self.enable_cache:
            self._cache_put(key, result)
        return result

    def compare_files(
        self,
        file_a: str | Path,
        file_b: str | Path,
        options: Optional[EngineCompareOptions] = None,
        encoding: str = "utf-8",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, object]:
        text_a = self._read_text_file(file_a, encoding=encoding)
        text_b = self._read_text_file(file_b, encoding=encoding)
        return self.compare_texts(text_a, text_b, options=options, progress_callback=progress_callback)

    def log_result(self, result: Dict[str, object], output_file: str | Path, append: bool = True) -> None:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_text_file(path_like: str | Path, encoding: str = "utf-8") -> str:
        path = Path(path_like)
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".md"}:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: .txt, .md")
        return path.read_text(encoding=encoding)
