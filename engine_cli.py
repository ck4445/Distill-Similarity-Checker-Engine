from __future__ import annotations

import argparse
import json

from distill_similarity_checker_engine import EngineCompareOptions, SimilarityEngine
from preprocess import PreprocessOptions


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill Similarity Checker Engine CLI")
    parser.add_argument("--file-a", required=True, help="Path to first .txt or .md file")
    parser.add_argument("--file-b", required=True, help="Path to second .txt or .md file")
    parser.add_argument("--mode", choices=["Strict Style", "Loose Style"], default="Strict Style")
    parser.add_argument("--bootstrap-runs", type=int, default=100)
    parser.add_argument("--ignore-urls", action="store_true")
    parser.add_argument("--ignore-numbers", action="store_true")
    parser.add_argument("--ignore-code-blocks", action="store_true")
    parser.add_argument("--ignore-blockquotes", action="store_true")
    parser.add_argument("--strip-headings", action="store_true")
    parser.add_argument("--strip-markdown", action="store_true")
    parser.add_argument("--log-jsonl", default="", help="Optional JSONL output path for appending results")
    args = parser.parse_args()

    preprocess = PreprocessOptions(
        ignore_blockquotes=args.ignore_blockquotes,
        ignore_code_blocks=args.ignore_code_blocks,
        ignore_urls=args.ignore_urls,
        ignore_numbers=args.ignore_numbers,
        strip_headings=args.strip_headings,
        strip_markdown=args.strip_markdown,
    )
    options = EngineCompareOptions(
        mode=args.mode,
        bootstrap_runs=args.bootstrap_runs,
        preprocess=preprocess,
    )
    engine = SimilarityEngine(enable_cache=True, cache_size=128)
    result = engine.compare_files(args.file_a, args.file_b, options=options)

    score = result["final"]["score_1_1000"]
    percent = result["final"]["percent"]
    print(f"Style similarity: {score}/1000 ({percent:.2f}%)")
    print(json.dumps(result, indent=2))

    if args.log_jsonl:
        engine.log_result(result, args.log_jsonl, append=True)
        print(f"Logged result to: {args.log_jsonl}")


if __name__ == "__main__":
    main()
