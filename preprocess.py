import re
import unicodedata
from dataclasses import dataclass
from typing import List


_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)

_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_WHITESPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
_MARKDOWN_SYNTAX = re.compile(r"[*_`~>#\[\]\(\)\-]{1,}")
_CODE_BLOCK = re.compile(r"```.*?```|`[^`\n]+`", re.DOTALL)
_BLOCKQUOTE_LINE = re.compile(r"^\s*>\s?.*$", re.MULTILINE)
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.MULTILINE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


@dataclass
class PreprocessOptions:
    ignore_blockquotes: bool = False
    ignore_code_blocks: bool = False
    ignore_urls: bool = False
    ignore_numbers: bool = False
    strip_headings: bool = False
    strip_markdown: bool = False


@dataclass
class ProcessedText:
    raw_input: str
    normalized_text: str
    style_text: str
    sentences: List[str]
    paragraphs: List[str]
    word_count: int
    line_count: int


def _normalize_common(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_QUOTE_TRANSLATION)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def _apply_stripping(text: str, options: PreprocessOptions) -> str:
    value = text
    if options.ignore_code_blocks:
        value = _CODE_BLOCK.sub(" ", value)
    if options.ignore_blockquotes:
        value = _BLOCKQUOTE_LINE.sub(" ", value)
    if options.strip_headings:
        value = _HEADING_LINE.sub(" ", value)
    if options.ignore_urls:
        value = _URL_PATTERN.sub(" ", value)
    if options.ignore_numbers:
        value = _NUMBER_PATTERN.sub(" ", value)
    return value


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    rough = _SENTENCE_SPLIT.split(text)
    out: List[str] = []
    for part in rough:
        part = part.strip()
        if part:
            out.append(part)
    return out


def split_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def preprocess_text(text: str, options: PreprocessOptions) -> ProcessedText:
    normalized = _normalize_common(text)
    stripped = _apply_stripping(normalized, options)
    stripped = _normalize_common(stripped)

    style_text = stripped
    clean_text = stripped
    if options.strip_markdown:
        clean_text = _MARKDOWN_SYNTAX.sub(" ", clean_text)
        clean_text = _normalize_common(clean_text)

    sentences = split_sentences(clean_text)
    paragraphs = split_paragraphs(clean_text)
    words = re.findall(r"\b\w+\b", clean_text)
    lines = [ln for ln in style_text.split("\n") if ln.strip()]

    return ProcessedText(
        raw_input=text,
        normalized_text=normalized,
        style_text=style_text,
        sentences=sentences,
        paragraphs=paragraphs,
        word_count=len(words),
        line_count=len(lines),
    )
