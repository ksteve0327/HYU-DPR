#!/usr/bin/env python3
"""Create a DPR paper-style Table 2 markdown from BM25 and DPR summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25", default="../data/patent/bm25_results.json")
    parser.add_argument("--dpr", default="../outputs/patent_dpr/dpr_eval_summary.json")
    parser.add_argument("--out", default="../outputs/patent_dpr/table2_patent_retrieval.md")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 20, 100])
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "summary" in payload:
        return payload["summary"]
    return payload


def format_row(method: str, summary: dict[str, object], k_values: list[int]) -> list[str]:
    accuracy = summary["accuracy"]
    return [method] + [f"{accuracy[f'top_{k}'] * 100:.2f}" for k in k_values]


def main() -> None:
    args = parse_args()
    bm25 = load_summary(Path(args.bm25))
    dpr = load_summary(Path(args.dpr))
    headers = ["Method"] + [f"Top-{k}" for k in args.k_values]
    rows = [
        format_row("BM25", bm25, args.k_values),
        format_row("DPR", dpr, args.k_values),
    ]
    lines = [
        "# Patent Retrieval Table 2",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(["", "Accuracy is gold-patent hit rate over the held-out test split."])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
