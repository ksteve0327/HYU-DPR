#!/usr/bin/env python3
"""Prepare AI semiconductor patent data for DPR and BM25 retrieval experiments."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SEED = 12345
DEFAULT_TOP_K = 100
QUESTION_TYPES = ("purpose", "solution", "purpose_solution")
TOKEN_RE = re.compile(r"[0-9A-Za-z]+|[가-힣]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "method",
    "of",
    "the",
    "to",
    "with",
    "그",
    "기술",
    "대한",
    "무엇인가",
    "발명",
    "설명하라",
    "어떤",
    "위한",
    "이",
    "장치",
    "특허",
    "해결수단",
}


@dataclass(frozen=True)
class PatentRow:
    patent_id: str
    title: str
    text: str
    purpose: str
    solution: str
    purpose_solution: str
    middle_category: str
    small_category: str


@dataclass(frozen=True)
class Passage:
    passage_id: str
    patent_id: str
    chunk_id: int
    title: str
    text: str


@dataclass(frozen=True)
class QuerySample:
    query_id: str
    patent_id: str
    question_type: str
    question: str
    answers: list[str]
    positive_passage_id: str
    split: str


class BM25Index:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1 = k1
        self.b = b
        self.doc_lens: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self._build()

    def _build(self) -> None:
        doc_freq: Counter[str] = Counter()
        for doc_idx, passage in enumerate(self.passages):
            tokens = tokenize(f"{passage.title} {passage.text}")
            counts = Counter(tokens)
            self.doc_lens.append(sum(counts.values()))
            for token, tf in counts.items():
                self.postings[token].append((doc_idx, tf))
                doc_freq[token] += 1

        total_docs = len(self.passages)
        self.avgdl = sum(self.doc_lens) / total_docs if total_docs else 0.0
        for token, df in doc_freq.items():
            self.idf[token] = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int,
        max_query_terms: int | None = None,
        min_query_idf: float = 0.0,
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        query_terms = Counter(tokenize(query))
        ranked_terms = [
            (token, qtf, self.idf.get(token, 0.0))
            for token, qtf in query_terms.items()
            if token in self.postings and self.idf.get(token, 0.0) >= min_query_idf
        ]
        if not ranked_terms:
            ranked_terms = [
                (token, qtf, self.idf.get(token, 0.0))
                for token, qtf in query_terms.items()
                if token in self.postings
            ]
        ranked_terms.sort(key=lambda item: item[2], reverse=True)
        if max_query_terms:
            ranked_terms = ranked_terms[:max_query_terms]

        for token, qtf, idf in ranked_terms:
            postings = self.postings.get(token)
            if not postings:
                continue
            for doc_idx, tf in postings:
                dl = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf * (tf * (self.k1 + 1.0) / denom) * qtf
        return heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../patent_rawdata.csv", help="Source patent CSV path.")
    parser.add_argument("--output-dir", default="../data/patent", help="Output directory for DPR-ready data.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--chunk-tokens", type=int, default=140)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-query-terms", type=int, default=16)
    parser.add_argument("--min-query-idf", type=float, default=0.2)
    parser.add_argument("--sample-size", type=int, default=20, help="Number of test queries to save for manual review.")
    parser.add_argument(
        "--positive-mode",
        choices=["patent_first", "answer_string"],
        default="patent_first",
        help="patent_first keeps the original gold-patent positive; answer_string requires the positive passage to contain an answer string.",
    )
    parser.add_argument(
        "--hard-negative-mode",
        choices=["different_patent", "bm25_answer_free"],
        default="different_patent",
        help="bm25_answer_free follows DPR-style hard negatives: high BM25 score, different patent, and no answer string.",
    )
    return parser.parse_args()


def clean_text(value: str | None) -> str:
    value = value or ""
    value = value.replace("\ufeff", "")
    value = value.replace("\t", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenize(text: str) -> list[str]:
    tokens = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        if len(token) <= 1:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def chunk_words(text: str, chunk_tokens: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= chunk_tokens:
        return [text]

    chunks = []
    step = max(1, chunk_tokens - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_tokens]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_tokens >= len(words):
            break
    return chunks


def read_patents(path: Path) -> list[PatentRow]:
    rows: list[PatentRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            patent_id = clean_text(raw.get("patent_id") or raw.get("출원번호"))
            title = clean_text(raw.get("발명의 명칭"))
            text = clean_text(raw.get("graphrag_text"))
            purpose = clean_text(raw.get("AI요약(목적)"))
            solution = clean_text(raw.get("AI요약(솔루션)"))
            purpose_solution = clean_text(raw.get("AI요약(목적+솔루션)"))
            if not all((patent_id, title, text, purpose, solution, purpose_solution)):
                continue
            rows.append(
                PatentRow(
                    patent_id=patent_id,
                    title=title,
                    text=text,
                    purpose=purpose,
                    solution=solution,
                    purpose_solution=purpose_solution,
                    middle_category=clean_text(raw.get("중분류명")),
                    small_category=clean_text(raw.get("소분류명")),
                )
            )
    return rows


def build_passages(rows: list[PatentRow], chunk_tokens: int, overlap: int) -> list[Passage]:
    passages: list[Passage] = []
    for row in rows:
        for chunk_id, chunk in enumerate(chunk_words(row.text, chunk_tokens, overlap)):
            passage_id = f"{row.patent_id}::chunk_{chunk_id:03d}"
            passages.append(
                Passage(
                    passage_id=passage_id,
                    patent_id=row.patent_id,
                    chunk_id=chunk_id,
                    title=row.title,
                    text=chunk,
                )
            )
    return passages


def query_templates(row: PatentRow) -> dict[str, tuple[str, list[str]]]:
    return {
        "purpose": (
            f"'{row.title}' 특허의 기술 목적은 무엇인가?",
            [row.purpose],
        ),
        "solution": (
            f"'{row.title}' 특허는 어떤 해결수단을 제안하는가?",
            [row.solution],
        ),
        "purpose_solution": (
            f"'{row.title}' 특허의 목적과 솔루션을 설명하라.",
            [row.purpose_solution, row.purpose, row.solution],
        ),
    }


def contains_answer(text: str, answers: Iterable[str]) -> bool:
    return any(answer and answer in text for answer in answers)


def split_patents(rows: list[PatentRow], seed: int, train_ratio: float, dev_ratio: float) -> dict[str, str]:
    patent_ids = [row.patent_id for row in rows]
    rng = random.Random(seed)
    rng.shuffle(patent_ids)
    train_end = int(len(patent_ids) * train_ratio)
    dev_end = train_end + int(len(patent_ids) * dev_ratio)
    split_by_id = {}
    for patent_id in patent_ids[:train_end]:
        split_by_id[patent_id] = "train"
    for patent_id in patent_ids[train_end:dev_end]:
        split_by_id[patent_id] = "dev"
    for patent_id in patent_ids[dev_end:]:
        split_by_id[patent_id] = "test"
    return split_by_id


def make_queries(
    rows: list[PatentRow],
    split_by_id: dict[str, str],
    first_positive_by_patent: dict[str, Passage],
    positive_by_query: dict[tuple[str, str], Passage] | None = None,
) -> list[QuerySample]:
    queries: list[QuerySample] = []
    for row in rows:
        templates = query_templates(row)
        for question_type in QUESTION_TYPES:
            question, answers = templates[question_type]
            if positive_by_query is not None:
                positive = positive_by_query.get((row.patent_id, question_type))
                if positive is None:
                    continue
            else:
                positive = first_positive_by_patent[row.patent_id]
            queries.append(
                QuerySample(
                    query_id=f"{row.patent_id}::{question_type}",
                    patent_id=row.patent_id,
                    question_type=question_type,
                    question=question,
                    answers=answers,
                    positive_passage_id=positive.passage_id,
                    split=split_by_id[row.patent_id],
                )
            )
    return queries


def choose_positive_passages(rows: list[PatentRow], passages_by_patent: dict[str, list[Passage]]) -> dict[str, Passage]:
    positive_by_patent = {}
    for row in rows:
        candidates = passages_by_patent[row.patent_id]
        answer_texts = (row.purpose_solution, row.purpose, row.solution)
        chosen = candidates[0]
        for passage in candidates:
            if any(answer and answer in passage.text for answer in answer_texts):
                chosen = passage
                break
        positive_by_patent[row.patent_id] = chosen
    return positive_by_patent


def choose_answer_string_positive_passages(
    rows: list[PatentRow],
    passages_by_patent: dict[str, list[Passage]],
) -> tuple[dict[tuple[str, str], Passage], list[dict[str, str]]]:
    positive_by_query = {}
    missing = []
    for row in rows:
        candidates = passages_by_patent[row.patent_id]
        for question_type, (_question, answers) in query_templates(row).items():
            chosen = None
            for passage in candidates:
                if contains_answer(passage.text, answers):
                    chosen = passage
                    break
            if chosen is None:
                missing.append({"patent_id": row.patent_id, "question_type": question_type})
                continue
            positive_by_query[(row.patent_id, question_type)] = chosen
    return positive_by_query, missing


def passage_to_ctx(passage: Passage) -> dict[str, str]:
    return {"title": passage.title, "text": passage.text, "passage_id": passage.passage_id, "patent_id": passage.patent_id}


def make_dpr_samples(
    queries: Iterable[QuerySample],
    passages_by_id: dict[str, Passage],
    bm25_by_query: dict[str, list[tuple[str, float]]],
    fallback_negative_by_patent: dict[str, Passage],
    fallback_passages: list[Passage] | None = None,
    hard_negative_mode: str = "different_patent",
) -> list[dict[str, object]]:
    samples = []
    for query in queries:
        positive = passages_by_id[query.positive_passage_id]
        hard_negative = None
        for passage_id, _score in bm25_by_query[query.query_id]:
            candidate = passages_by_id[passage_id]
            if candidate.patent_id == query.patent_id:
                continue
            if hard_negative_mode == "bm25_answer_free" and contains_answer(candidate.text, query.answers):
                continue
            hard_negative = candidate
            break
        if hard_negative is None:
            hard_negative = fallback_negative_by_patent[query.patent_id]
            if hard_negative_mode == "bm25_answer_free" and contains_answer(hard_negative.text, query.answers):
                hard_negative = None
                for candidate in fallback_passages or passages_by_id.values():
                    if candidate.patent_id != query.patent_id and not contains_answer(candidate.text, query.answers):
                        hard_negative = candidate
                        break
            if hard_negative is None:
                raise ValueError(f"No hard negative found for {query.query_id}")

        samples.append(
            {
                "question": query.question,
                "answers": query.answers,
                "positive_ctxs": [passage_to_ctx(positive)],
                "negative_ctxs": [],
                "hard_negative_ctxs": [passage_to_ctx(hard_negative)],
                "query_id": query.query_id,
                "patent_id": query.patent_id,
                "question_type": query.question_type,
            }
        )
    return samples


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_passages(path: Path, passages: Iterable[Passage]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "text", "title"])
        for passage in passages:
            writer.writerow([passage.passage_id, passage.text, passage.title])


def write_test_tsv(path: Path, queries: Iterable[QuerySample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for query in queries:
            writer.writerow([query.question, repr(query.answers), query.query_id])


def evaluate_ranked(
    queries: Iterable[QuerySample],
    bm25_by_query: dict[str, list[tuple[str, float]]],
    passages_by_id: dict[str, Passage],
    k_values: tuple[int, ...],
) -> dict[str, object]:
    per_type = {question_type: {f"top_{k}": 0 for k in k_values} for question_type in QUESTION_TYPES}
    per_type_counts = Counter()
    totals = {f"top_{k}": 0 for k in k_values}
    total_queries = 0

    for query in queries:
        total_queries += 1
        per_type_counts[query.question_type] += 1
        ranked = bm25_by_query[query.query_id]
        retrieved_patents = [passages_by_id[passage_id].patent_id for passage_id, _score in ranked]
        for k in k_values:
            hit = query.patent_id in retrieved_patents[:k]
            totals[f"top_{k}"] += int(hit)
            per_type[query.question_type][f"top_{k}"] += int(hit)

    table = {key: (value / total_queries if total_queries else 0.0) for key, value in totals.items()}
    type_table = {}
    for question_type, counts in per_type.items():
        denom = per_type_counts[question_type]
        type_table[question_type] = {key: (value / denom if denom else 0.0) for key, value in counts.items()}
    return {"count": total_queries, "accuracy": table, "by_question_type": type_table}


def build_bm25_results(
    queries: list[QuerySample],
    bm25_by_query: dict[str, list[tuple[str, float]]],
    passages_by_id: dict[str, Passage],
    k_values: tuple[int, ...],
    sample_size: int,
    seed: int,
) -> dict[str, object]:
    test_queries = [query for query in queries if query.split == "test"]
    summary = evaluate_ranked(test_queries, bm25_by_query, passages_by_id, k_values)
    rng = random.Random(seed)
    sample_queries = rng.sample(test_queries, min(sample_size, len(test_queries)))
    samples = []
    for query in sample_queries:
        ctxs = []
        for rank, (passage_id, score) in enumerate(bm25_by_query[query.query_id][:5], start=1):
            passage = passages_by_id[passage_id]
            ctxs.append(
                {
                    "rank": rank,
                    "passage_id": passage_id,
                    "patent_id": passage.patent_id,
                    "title": passage.title,
                    "score": score,
                    "is_gold_patent": passage.patent_id == query.patent_id,
                    "text_preview": passage.text[:300],
                }
            )
        samples.append(
            {
                "query_id": query.query_id,
                "question": query.question,
                "gold_patent_id": query.patent_id,
                "question_type": query.question_type,
                "top_5": ctxs,
            }
        )

    compact_results = []
    for query in test_queries:
        compact_results.append(
            {
                "query_id": query.query_id,
                "question": query.question,
                "gold_patent_id": query.patent_id,
                "question_type": query.question_type,
                "ctxs": [
                    {
                        "id": passage_id,
                        "score": score,
                        "patent_id": passages_by_id[passage_id].patent_id,
                        "has_gold_patent": passages_by_id[passage_id].patent_id == query.patent_id,
                    }
                    for passage_id, score in bm25_by_query[query.query_id]
                ],
            }
        )
    return {"summary": summary, "sample_top5": samples, "test_results": compact_results}


def write_table2(path: Path, bm25_summary: dict[str, object], k_values: tuple[int, ...]) -> None:
    accuracy = bm25_summary["accuracy"]
    headers = ["Method"] + [f"Top-{k}" for k in k_values]
    row = ["BM25"] + [f"{accuracy[f'top_{k}'] * 100:.2f}" for k in k_values]
    lines = [
        "# Patent Retrieval Table 2 Baseline",
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
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_patents(input_path)
    if not rows:
        raise SystemExit(f"No usable patent rows found in {input_path}")

    passages = build_passages(rows, args.chunk_tokens, args.chunk_overlap)
    passages_by_id = {passage.passage_id: passage for passage in passages}
    passages_by_patent: dict[str, list[Passage]] = defaultdict(list)
    for passage in passages:
        passages_by_patent[passage.patent_id].append(passage)

    split_by_id = split_patents(rows, args.seed, args.train_ratio, args.dev_ratio)
    positive_by_patent = choose_positive_passages(rows, passages_by_patent)
    missing_answer_positive: list[dict[str, str]] = []
    if args.positive_mode == "answer_string":
        positive_by_query, missing_answer_positive = choose_answer_string_positive_passages(rows, passages_by_patent)
        queries = make_queries(rows, split_by_id, positive_by_patent, positive_by_query)
    else:
        queries = make_queries(rows, split_by_id, positive_by_patent)

    bm25 = BM25Index(passages)
    bm25_by_query: dict[str, list[tuple[str, float]]] = {}
    for idx, query in enumerate(queries, start=1):
        ranked = bm25.search(
            query.question,
            args.top_k,
            max_query_terms=args.max_query_terms,
            min_query_idf=args.min_query_idf,
        )
        bm25_by_query[query.query_id] = [(passages[doc_idx].passage_id, float(score)) for doc_idx, score in ranked]
        if idx % 1000 == 0:
            print(f"BM25 ranked {idx}/{len(queries)} queries", flush=True)

    all_patent_ids = [row.patent_id for row in rows]
    fallback_negative_by_patent = {}
    for idx, patent_id in enumerate(all_patent_ids):
        fallback_id = all_patent_ids[(idx + 1) % len(all_patent_ids)]
        fallback_negative_by_patent[patent_id] = passages_by_patent[fallback_id][0]

    train_queries = [query for query in queries if query.split == "train"]
    dev_queries = [query for query in queries if query.split == "dev"]
    test_queries = [query for query in queries if query.split == "test"]

    write_passages(output_dir / "passages.tsv", passages)
    write_json(
        output_dir / "train.json",
        make_dpr_samples(
            train_queries,
            passages_by_id,
            bm25_by_query,
            fallback_negative_by_patent,
            passages,
            args.hard_negative_mode,
        ),
    )
    write_json(
        output_dir / "dev.json",
        make_dpr_samples(
            dev_queries,
            passages_by_id,
            bm25_by_query,
            fallback_negative_by_patent,
            passages,
            args.hard_negative_mode,
        ),
    )
    write_test_tsv(output_dir / "test.tsv", test_queries)

    query_metadata = [
        {
            "query_id": query.query_id,
            "patent_id": query.patent_id,
            "question_type": query.question_type,
            "question": query.question,
            "answers": query.answers,
            "positive_passage_id": query.positive_passage_id,
            "split": query.split,
        }
        for query in queries
    ]
    write_jsonl(output_dir / "queries.jsonl", query_metadata)
    write_jsonl(output_dir / "test_metadata.jsonl", (record for record in query_metadata if record["split"] == "test"))

    split_counts = Counter(split_by_id.values())
    manifest = {
        "input": str(input_path),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "dev_ratio": args.dev_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.dev_ratio, 6),
        "patent_count": len(rows),
        "passage_count": len(passages),
        "query_count": len(queries),
        "split_patent_counts": dict(split_counts),
        "split_query_counts": Counter(query.split for query in queries),
        "chunk_tokens": args.chunk_tokens,
        "chunk_overlap": args.chunk_overlap,
        "bm25_top_k": args.top_k,
        "bm25_max_query_terms": args.max_query_terms,
        "bm25_min_query_idf": args.min_query_idf,
        "question_types": QUESTION_TYPES,
        "positive_mode": args.positive_mode,
        "hard_negative_mode": args.hard_negative_mode,
        "evaluation_basis": "answer_string_containment" if args.positive_mode == "answer_string" else "gold_patent_id",
        "missing_answer_positive_count": len(missing_answer_positive),
        "missing_answer_positive_by_type": dict(Counter(item["question_type"] for item in missing_answer_positive)),
    }
    manifest["split_query_counts"] = dict(manifest["split_query_counts"])
    write_json(output_dir / "split_manifest.json", manifest)

    bm25_results = build_bm25_results(queries, bm25_by_query, passages_by_id, (5, 20, 100), args.sample_size, args.seed)
    write_json(output_dir / "bm25_results.json", bm25_results)
    write_table2(output_dir / "table2_bm25.md", bm25_results["summary"], (5, 20, 100))

    print(json.dumps({"output_dir": str(output_dir), **manifest, "bm25": bm25_results["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
