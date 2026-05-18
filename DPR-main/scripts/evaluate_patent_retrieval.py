#!/usr/bin/env python3
"""Evaluate DPR/BM25 retrieval outputs by gold patent id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Retriever output JSON file.")
    parser.add_argument("--metadata", default="../data/patent/test_metadata.jsonl", help="Test metadata JSONL.")
    parser.add_argument("--out-json", default="../data/patent/dpr_eval_summary.json")
    parser.add_argument("--out-md", default="../data/patent/table2_dpr.md")
    parser.add_argument("--method", default="DPR")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 20, 100])
    return parser.parse_args()


def patent_id_from_passage_id(passage_id: str) -> str:
    return passage_id.split("::", 1)[0]


def load_metadata(path: Path) -> dict[str, dict[str, object]]:
    by_id = {}
    by_question = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            by_id[record["query_id"]] = record
            by_question[record["question"]] = record
    return {"by_id": by_id, "by_question": by_question}


def load_results(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "test_results" in payload:
        return payload["test_results"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported retrieval results schema in {path}")


def evaluate(
    results: list[dict[str, object]],
    metadata_lookup: dict[str, dict[str, dict[str, object]]],
    k_values: list[int],
) -> dict[str, object]:
    counts = {f"top_{k}": 0 for k in k_values}
    by_type: dict[str, dict[str, int]] = {}
    type_counts: dict[str, int] = {}
    missing = []
    details = []

    for item in results:
        question = item["question"]
        query_id = item.get("query_id") or item.get("question_id")
        metadata = None
        if query_id is not None:
            metadata = metadata_lookup["by_id"].get(query_id)
        if metadata is None:
            metadata = metadata_lookup["by_question"].get(question)
        if metadata is None:
            missing.append(question)
            continue
        gold_patent_id = metadata["patent_id"]
        question_type = metadata["question_type"]
        by_type.setdefault(question_type, {f"top_{k}": 0 for k in k_values})
        type_counts[question_type] = type_counts.get(question_type, 0) + 1

        ctxs = item.get("ctxs", [])
        retrieved_patent_ids = []
        for ctx in ctxs:
            if "patent_id" in ctx:
                retrieved_patent_ids.append(ctx["patent_id"])
            else:
                retrieved_patent_ids.append(patent_id_from_passage_id(str(ctx["id"])))

        detail_hits = {}
        for k in k_values:
            hit = gold_patent_id in retrieved_patent_ids[:k]
            counts[f"top_{k}"] += int(hit)
            by_type[question_type][f"top_{k}"] += int(hit)
            detail_hits[f"top_{k}"] = hit
        details.append(
            {
                "query_id": metadata["query_id"],
                "question": question,
                "gold_patent_id": gold_patent_id,
                "question_type": question_type,
                "hits": detail_hits,
                "top_5_patent_ids": retrieved_patent_ids[:5],
            }
        )

    evaluated_count = len(details)
    summary = {
        "count": evaluated_count,
        "missing_questions": len(missing),
        "accuracy": {key: (value / evaluated_count if evaluated_count else 0.0) for key, value in counts.items()},
        "by_question_type": {},
    }
    for question_type, values in by_type.items():
        denom = type_counts[question_type]
        summary["by_question_type"][question_type] = {
            key: (value / denom if denom else 0.0) for key, value in values.items()
        }
    return {"summary": summary, "details": details, "missing_question_examples": missing[:20]}


def write_table(path: Path, method: str, summary: dict[str, object], k_values: list[int]) -> None:
    accuracy = summary["accuracy"]
    headers = ["Method"] + [f"Top-{k}" for k in k_values]
    row = [method] + [f"{accuracy[f'top_{k}'] * 100:.2f}" for k in k_values]
    lines = [
        f"# Patent Retrieval Table 2 - {method}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(row) + " |",
        "",
        "Accuracy is gold-patent hit rate over the held-out test split.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metadata = load_metadata(Path(args.metadata))
    results = load_results(Path(args.results))
    evaluation = evaluate(results, metadata, args.k_values)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_table(out_md, args.method, evaluation["summary"], args.k_values)
    print(json.dumps(evaluation["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
