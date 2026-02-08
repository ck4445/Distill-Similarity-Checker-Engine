from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import ProcessedText

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None


TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b")

FUNCTION_WORDS = [
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "as",
    "that",
    "this",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "not",
    "never",
    "i",
    "you",
    "we",
    "they",
    "it",
    "he",
    "she",
    "don't",
    "can't",
    "i'm",
    "we're",
    "it's",
]

PRONOUNS = {"i", "me", "my", "mine", "you", "your", "yours", "we", "our", "ours", "they", "them", "their", "it"}
AUXILIARIES = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "am",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "can",
    "could",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
}
NEGATIONS = {"not", "never", "no", "n't"}

HEDGES = {"may", "might", "probably", "seems", "likely", "perhaps", "possibly", "roughly"}
CERTAINTY = {"clearly", "obviously", "definitely", "certainly", "undoubtedly", "always", "never"}
TRANSITIONS = {"however", "therefore", "moreover", "furthermore", "meanwhile", "thus", "example"}
DIRECTIVES = {"must", "should", "consider", "do", "need", "required"}
POLITENESS = {"please", "thanks", "thank", "appreciate", "sorry"}

PUNCT_KEYS = [",", ".", ";", ":", "(", ")", '"', "?", "!", "..."]
CONTRACTION_RE = re.compile(r"\b\w+'\w+\b")
BULLET_LINE_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+")
HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+")


@dataclass
class TokenInfo:
    text: str
    lower: str
    is_alpha: bool
    is_punct: bool
    pos: str


@dataclass
class ParsedText:
    tokens: List[TokenInfo]
    pos_sequence: List[str]
    masked_text: str
    content_words: List[str]
    words_only: List[str]


@dataclass
class StyleResult:
    overall: float
    family_scores: Dict[str, float]
    function_word_deltas: List[Dict[str, float]]
    rhythm_stats: Dict[str, Dict[str, float]]
    punctuation_rates: Dict[str, Dict[str, float]]
    marker_rates: Dict[str, Dict[str, float]]
    topic_leakage_overlap: float
    leakage_label: str
    confidence_band: Dict[str, float] | None
    short_text_warning: bool
    masked_preview: Dict[str, str]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def _rate_per_1000(count: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return 1000.0 * count / denom


def _js_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a / (np.sum(a) + 1e-12)
    b = b / (np.sum(b) + 1e-12)
    m = 0.5 * (a + b)
    kl_a = np.sum(a * np.log2((a + 1e-12) / (m + 1e-12)))
    kl_b = np.sum(b * np.log2((b + 1e-12) / (m + 1e-12)))
    js = 0.5 * (kl_a + kl_b)
    return float(np.clip(1.0 - js, 0.0, 1.0))


def _wasserstein_1d(x: np.ndarray, y: np.ndarray, n: int = 128) -> float:
    x = np.sort(x.astype(np.float64))
    y = np.sort(y.astype(np.float64))
    q = np.linspace(0.0, 1.0, n, endpoint=True)
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def _exp_sim(distance: float, k: float) -> float:
    return float(np.clip(math.exp(-distance / max(k, 1e-9)), 0.0, 1.0))


@lru_cache(maxsize=1)
def _get_spacy_model():
    if spacy is None:
        return None
    try:
        # Keep POS tagging while dropping heavy components not used by this app.
        return spacy.load("en_core_web_sm", exclude=["parser", "ner", "textcat"])
    except Exception:
        return None


def _heuristic_pos(token: str) -> str:
    t = token.lower()
    if t in PRONOUNS:
        return "PRON"
    if t in AUXILIARIES:
        return "AUX"
    if t in {"a", "an", "the", "this", "that", "these", "those"}:
        return "DET"
    if t in {"in", "on", "at", "by", "to", "from", "for", "with", "of", "about"}:
        return "ADP"
    if t in {"and", "or", "but", "yet", "nor", "so"}:
        return "CCONJ"
    if t.isdigit():
        return "NUM"
    if t.endswith("ly"):
        return "ADV"
    if t.endswith(("ing", "ed", "en", "ize", "ise")):
        return "VERB"
    if t.endswith(("ous", "ful", "able", "ible", "ive", "al", "less")):
        return "ADJ"
    return "NOUN"


def _tokenize_with_pos(text: str) -> List[TokenInfo]:
    nlp = _get_spacy_model()
    if nlp is not None:
        doc = nlp(text)
        out: List[TokenInfo] = []
        for tok in doc:
            if tok.is_space:
                continue
            pos = tok.pos_ if tok.pos_ else _heuristic_pos(tok.text)
            out.append(
                TokenInfo(
                    text=tok.text,
                    lower=tok.text.lower(),
                    is_alpha=tok.is_alpha,
                    is_punct=tok.is_punct,
                    pos=pos,
                )
            )
        return out

    out = []
    for raw in TOKEN_RE.findall(text):
        is_alpha = bool(WORD_RE.fullmatch(raw))
        is_punct = not is_alpha and bool(re.fullmatch(r"[^\w\s]+", raw))
        out.append(TokenInfo(text=raw, lower=raw.lower(), is_alpha=is_alpha, is_punct=is_punct, pos=_heuristic_pos(raw)))
    return out


def _mask_token(t: TokenInfo, ignore_numbers: bool) -> str:
    if t.is_punct:
        return t.text
    if t.lower in FUNCTION_WORDS or t.lower in PRONOUNS or t.lower in AUXILIARIES or t.lower in NEGATIONS:
        return t.text
    if t.pos in {"NOUN"}:
        return "NOUN"
    if t.pos in {"PROPN"}:
        return "PROPN"
    if t.pos in {"VERB"}:
        return "VERB"
    if t.pos in {"ADJ"}:
        return "ADJ"
    if t.pos in {"ADV"}:
        return "ADV"
    if t.pos == "NUM":
        return "" if ignore_numbers else "NUM"
    return t.pos if t.pos else "X"


def _join_tokens(tokens: Sequence[str]) -> str:
    out = ""
    for token in tokens:
        if not token:
            continue
        if re.fullmatch(r"[^\w\s]+", token):
            out += token
        else:
            if out and not out.endswith((" ", "\n", "(", "[", "{", '"')):
                out += " "
            out += token
    return out.strip()


def parse_text(processed: ProcessedText, ignore_numbers: bool) -> ParsedText:
    tokens = _tokenize_with_pos(processed.style_text)
    masked_tokens = [_mask_token(t, ignore_numbers=ignore_numbers) for t in tokens]
    pos_sequence = [t.pos for t in tokens if not t.is_punct]
    words_only = [t.lower for t in tokens if t.is_alpha]
    content_words = [t.lower for t in tokens if t.is_alpha and t.pos in {"NOUN", "PROPN"} and t.lower not in FUNCTION_WORDS]
    return ParsedText(
        tokens=tokens,
        pos_sequence=pos_sequence,
        masked_text=_join_tokens(masked_tokens),
        content_words=content_words,
        words_only=words_only,
    )


def _ngram_counts(seq: List[str], n: int) -> Dict[str, float]:
    if len(seq) < n:
        return {}
    out: Dict[str, float] = {}
    for i in range(len(seq) - n + 1):
        key = "|".join(seq[i : i + n])
        out[key] = out.get(key, 0.0) + 1.0
    return out


def _dict_similarity_js(a: Dict[str, float], b: Dict[str, float], alpha: float = 0.5) -> float:
    vocab = sorted(set(a) | set(b))
    if not vocab:
        return 0.0
    va = np.array([a.get(k, 0.0) + alpha for k in vocab], dtype=np.float64)
    vb = np.array([b.get(k, 0.0) + alpha for k in vocab], dtype=np.float64)
    return _js_similarity(va, vb)


def _function_word_similarity(pa: ParsedText, pb: ParsedText) -> Tuple[float, List[Dict[str, float]]]:
    wa, wb = max(len(pa.words_only), 1), max(len(pb.words_only), 1)
    ca = Counter(pa.words_only)
    cb = Counter(pb.words_only)
    vec_a = np.array([_rate_per_1000(ca.get(w, 0), wa) for w in FUNCTION_WORDS], dtype=np.float32)
    vec_b = np.array([_rate_per_1000(cb.get(w, 0), wb) for w in FUNCTION_WORDS], dtype=np.float32)
    sim = _cosine(vec_a, vec_b)
    diffs = []
    for idx, w in enumerate(FUNCTION_WORDS):
        diffs.append({"token": w, "a_rate": round(float(vec_a[idx]), 2), "b_rate": round(float(vec_b[idx]), 2), "delta": round(float(abs(vec_a[idx] - vec_b[idx])), 2)})
    diffs.sort(key=lambda x: x["delta"], reverse=True)
    return sim, diffs[:10]


def _pos_ngram_similarity(pa: ParsedText, pb: ParsedText) -> float:
    uni = _dict_similarity_js(_ngram_counts(pa.pos_sequence, 1), _ngram_counts(pb.pos_sequence, 1))
    bi = _dict_similarity_js(_ngram_counts(pa.pos_sequence, 2), _ngram_counts(pb.pos_sequence, 2))
    tri = _dict_similarity_js(_ngram_counts(pa.pos_sequence, 3), _ngram_counts(pb.pos_sequence, 3))
    return float(np.clip(0.4 * uni + 0.35 * bi + 0.25 * tri, 0.0, 1.0))


def _rhythm_similarity(a: ProcessedText, b: ProcessedText) -> Tuple[float, Dict[str, Dict[str, float]]]:
    def lens(items: List[str]) -> np.ndarray:
        vals = [len(WORD_RE.findall(s)) for s in items if s.strip()]
        return np.array(vals if vals else [0], dtype=np.float32)

    sent_a, sent_b = lens(a.sentences), lens(b.sentences)
    para_a, para_b = lens(a.paragraphs), lens(b.paragraphs)

    sent_dist = _exp_sim(_wasserstein_1d(sent_a, sent_b), k=10.0)
    para_dist = _exp_sim(_wasserstein_1d(para_a, para_b), k=22.0)

    smean = _exp_sim(abs(float(np.mean(sent_a) - np.mean(sent_b))), k=8.0)
    sstd = _exp_sim(abs(float(np.std(sent_a) - np.std(sent_b))), k=6.0)

    short_a = float(np.mean(sent_a <= 12))
    short_b = float(np.mean(sent_b <= 12))
    long_a = float(np.mean(sent_a >= 25))
    long_b = float(np.mean(sent_b >= 25))
    ratio_sim = _exp_sim(abs((short_a - short_b)) + abs((long_a - long_b)), k=0.9)

    rhythm = float(np.clip(0.35 * sent_dist + 0.2 * para_dist + 0.2 * smean + 0.15 * sstd + 0.1 * ratio_sim, 0.0, 1.0))
    stats = {
        "sentence_lengths": {
            "a_mean": round(float(np.mean(sent_a)), 2),
            "b_mean": round(float(np.mean(sent_b)), 2),
            "a_median": round(float(np.median(sent_a)), 2),
            "b_median": round(float(np.median(sent_b)), 2),
            "a_std": round(float(np.std(sent_a)), 2),
            "b_std": round(float(np.std(sent_b)), 2),
        },
        "paragraph_lengths": {
            "a_mean": round(float(np.mean(para_a)), 2),
            "b_mean": round(float(np.mean(para_b)), 2),
        },
    }
    return rhythm, stats


def _punct_format_rates(processed: ProcessedText) -> Dict[str, float]:
    text = processed.style_text
    chars = max(len(text), 1)
    words = max(processed.word_count, 1)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    bullet_lines = sum(1 for ln in lines if BULLET_LINE_RE.match(ln))
    heading_lines = sum(1 for ln in lines if HEADING_LINE_RE.match(ln))
    word_tokens = WORD_RE.findall(text)

    rates = {k: 1000.0 * text.count(k) / chars for k in PUNCT_KEYS if k != "..."}
    rates["..."] = 1000.0 * len(re.findall(r"\.\.\.", text)) / chars
    rates["newline_density"] = 1000.0 * text.count("\n") / chars
    rates["bullet_line_ratio"] = bullet_lines / max(len(lines), 1)
    rates["heading_line_ratio"] = heading_lines / max(len(lines), 1)
    rates["question_rate_per_1000_words"] = _rate_per_1000(text.count("?"), words)
    rates["contraction_rate_per_1000_words"] = _rate_per_1000(len(CONTRACTION_RE.findall(text)), words)
    rates["uppercase_token_ratio"] = sum(1 for t in word_tokens if len(t) > 1 and t.isupper()) / max(len(word_tokens), 1)
    return rates


def _punct_format_similarity(a: ProcessedText, b: ProcessedText) -> Tuple[float, Dict[str, Dict[str, float]]]:
    ra = _punct_format_rates(a)
    rb = _punct_format_rates(b)
    keys = sorted(set(ra) | set(rb))
    va = np.array([ra.get(k, 0.0) for k in keys], dtype=np.float32)
    vb = np.array([rb.get(k, 0.0) for k in keys], dtype=np.float32)
    table = {k: {"a": round(float(ra.get(k, 0.0)), 3), "b": round(float(rb.get(k, 0.0)), 3)} for k in keys}
    return _cosine(va, vb), table


def _marker_vector(parsed: ParsedText) -> Dict[str, float]:
    words = parsed.words_only
    denom = max(len(words), 1)
    counts = Counter(words)
    return {
        "hedges": _rate_per_1000(sum(counts.get(w, 0) for w in HEDGES), denom),
        "certainty": _rate_per_1000(sum(counts.get(w, 0) for w in CERTAINTY), denom),
        "transitions": _rate_per_1000(sum(counts.get(w, 0) for w in TRANSITIONS), denom),
        "directives": _rate_per_1000(sum(counts.get(w, 0) for w in DIRECTIVES), denom),
        "politeness": _rate_per_1000(sum(counts.get(w, 0) for w in POLITENESS), denom),
    }


def _marker_similarity(pa: ParsedText, pb: ParsedText) -> Tuple[float, Dict[str, Dict[str, float]]]:
    va = _marker_vector(pa)
    vb = _marker_vector(pb)
    keys = sorted(set(va) | set(vb))
    arr_a = np.array([va.get(k, 0.0) for k in keys], dtype=np.float32)
    arr_b = np.array([vb.get(k, 0.0) for k in keys], dtype=np.float32)
    table = {k: {"a": round(float(va.get(k, 0.0)), 2), "b": round(float(vb.get(k, 0.0)), 2)} for k in keys}
    return _cosine(arr_a, arr_b), table


def _masked_char_similarity(pa: ParsedText, pb: ParsedText) -> float:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    mat = vec.fit_transform([pa.masked_text, pb.masked_text])
    return _cosine(mat[0].toarray().ravel(), mat[1].toarray().ravel())


def _raw_char_similarity(a: ProcessedText, b: ProcessedText) -> float:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    mat = vec.fit_transform([a.style_text, b.style_text])
    return _cosine(mat[0].toarray().ravel(), mat[1].toarray().ravel())


def _topic_leakage(pa: ParsedText, pb: ParsedText) -> Tuple[float, str]:
    sa, sb = set(pa.content_words), set(pb.content_words)
    if not sa and not sb:
        return 0.0, "Low"
    jaccard = len(sa & sb) / max(len(sa | sb), 1)
    if jaccard >= 0.45:
        return jaccard, "High"
    if jaccard >= 0.22:
        return jaccard, "Medium"
    return jaccard, "Low"


def compute_style_similarity(
    a: ProcessedText,
    b: ProcessedText,
    mode: str,
    weights: Dict[str, float],
    ignore_numbers: bool,
    bootstrap_runs: int = 100,
    progress_callback: Callable[[float, str], None] | None = None,
) -> StyleResult:
    def _progress(value: float, label: str) -> None:
        if progress_callback is not None:
            progress_callback(float(np.clip(value, 0.0, 1.0)), label)

    _progress(0.05, "Parsing text style signals...")
    pa = parse_text(a, ignore_numbers=ignore_numbers)
    pb = parse_text(b, ignore_numbers=ignore_numbers)

    _progress(0.2, "Computing function-word and syntax features...")
    function_sim, top_fn = _function_word_similarity(pa, pb)
    pos_sim = _pos_ngram_similarity(pa, pb)
    _progress(0.4, "Computing rhythm and punctuation features...")
    rhythm_sim, rhythm_stats = _rhythm_similarity(a, b)
    punct_sim, punct_table = _punct_format_similarity(a, b)
    _progress(0.55, "Computing discourse marker features...")
    marker_sim, marker_table = _marker_similarity(pa, pb)
    _progress(0.65, "Computing masked character texture...")
    masked_char_sim = _masked_char_similarity(pa, pb)
    raw_char_sim = _raw_char_similarity(a, b)

    leakage_overlap, leakage_label = _topic_leakage(pa, pb)
    is_short = min(a.word_count, b.word_count) < 150

    family_scores = {
        "function_words": function_sim,
        "pos_ngrams": pos_sim,
        "rhythm": rhythm_sim,
        "punctuation_formatting": punct_sim,
        "markers_stance": marker_sim,
        "masked_chargrams": masked_char_sim,
        "raw_chargrams": raw_char_sim,
    }

    applied = dict(weights)
    if mode == "Loose Style":
        applied["raw_chargrams"] = max(applied.get("raw_chargrams", 0.0), 0.05)
        applied["masked_chargrams"] = max(applied.get("masked_chargrams", 0.0) - 0.05, 0.0)
    else:
        applied["raw_chargrams"] = 0.0

    if is_short:
        applied["rhythm"] *= 0.55
        applied["pos_ngrams"] *= 0.75
        applied["function_words"] *= 1.2
        applied["punctuation_formatting"] *= 1.15

    total = sum(max(v, 0.0) for v in applied.values())
    if total <= 0:
        total = 1.0
    applied = {k: max(v, 0.0) / total for k, v in applied.items()}

    overall = 0.0
    for key, weight in applied.items():
        overall += weight * family_scores.get(key, 0.0)
    overall = float(np.clip(overall, 0.0, 1.0))

    _progress(0.75, "Estimating stability band...")
    band = _bootstrap_style_band(
        a,
        b,
        mode,
        applied,
        ignore_numbers,
        bootstrap_runs=bootstrap_runs,
        progress_callback=_progress,
    )
    _progress(1.0, "Done")
    return StyleResult(
        overall=overall,
        family_scores=family_scores,
        function_word_deltas=top_fn,
        rhythm_stats=rhythm_stats,
        punctuation_rates=punct_table,
        marker_rates=marker_table,
        topic_leakage_overlap=leakage_overlap,
        leakage_label=leakage_label,
        confidence_band=band,
        short_text_warning=is_short,
        masked_preview={"a": pa.masked_text[:350], "b": pb.masked_text[:350]},
    )


def _bootstrap_style_band(
    a: ProcessedText,
    b: ProcessedText,
    mode: str,
    weights: Dict[str, float],
    ignore_numbers: bool,
    bootstrap_runs: int = 100,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Dict[str, float] | None:
    if bootstrap_runs <= 0 or len(a.sentences) < 3 or len(b.sentences) < 3:
        return None

    rng = np.random.default_rng(42)
    a_sent = np.array(a.sentences, dtype=object)
    b_sent = np.array(b.sentences, dtype=object)
    sims: List[float] = []
    report_every = max(1, bootstrap_runs // 8)
    for i in range(bootstrap_runs):
        sa = rng.choice(a_sent, size=len(a_sent), replace=True).tolist()
        sb = rng.choice(b_sent, size=len(b_sent), replace=True).tolist()
        joined_a = " ".join(sa)
        joined_b = " ".join(sb)
        mini_a = ProcessedText(
            raw_input=a.raw_input,
            normalized_text=a.normalized_text,
            style_text=joined_a,
            sentences=sa,
            paragraphs=[" ".join(sa)],
            word_count=len(WORD_RE.findall(joined_a)),
            line_count=1,
        )
        mini_b = ProcessedText(
            raw_input=b.raw_input,
            normalized_text=b.normalized_text,
            style_text=joined_b,
            sentences=sb,
            paragraphs=[" ".join(sb)],
            word_count=len(WORD_RE.findall(joined_b)),
            line_count=1,
        )
        pa = parse_text(mini_a, ignore_numbers=ignore_numbers)
        pb = parse_text(mini_b, ignore_numbers=ignore_numbers)
        f_sim, _ = _function_word_similarity(pa, pb)
        p_sim = _pos_ngram_similarity(pa, pb)
        r_sim, _ = _rhythm_similarity(mini_a, mini_b)
        pu_sim, _ = _punct_format_similarity(mini_a, mini_b)
        m_sim, _ = _marker_similarity(pa, pb)
        mc_sim = _masked_char_similarity(pa, pb)
        rc_sim = _raw_char_similarity(mini_a, mini_b) if mode == "Loose Style" else 0.0
        fam = {
            "function_words": f_sim,
            "pos_ngrams": p_sim,
            "rhythm": r_sim,
            "punctuation_formatting": pu_sim,
            "markers_stance": m_sim,
            "masked_chargrams": mc_sim,
            "raw_chargrams": rc_sim,
        }
        score = sum(weights.get(k, 0.0) * fam.get(k, 0.0) for k in weights)
        sims.append(float(np.clip(score, 0.0, 1.0)))

        if progress_callback is not None and (i + 1) % report_every == 0:
            frac = 0.75 + 0.22 * ((i + 1) / bootstrap_runs)
            progress_callback(frac, f"Estimating stability band... ({i + 1}/{bootstrap_runs})")

    if not sims:
        return None
    p10, p90 = np.percentile(np.array(sims), [10, 90])
    return {"p10": round(100 * float(p10), 2), "p90": round(100 * float(p90), 2)}
