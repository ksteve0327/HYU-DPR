#!/usr/bin/env python3
"""Build local DPR vector-space SVGs for the patent report."""

from __future__ import annotations

import argparse
import html
import json
import pickle
from pathlib import Path

import numpy as np


CASES = {
    "case_a": {
        "question_id": "2025-0016785::purpose",
        "label": "예시 A",
        "title": "인공지능 코어 특허",
        "output": "patent_vector_space_case_a.svg",
    },
    "case_b": {
        "question_id": "19-007055::purpose",
        "label": "예시 B",
        "title": "High performance computing system for deep learning",
        "output": "patent_vector_space_case_b.svg",
    },
    "case_c": {
        "question_id": "2025-0016785::solution",
        "label": "예시 C",
        "title": "인공지능 코어 특허 solution",
        "output": "patent_vector_space_case_c.svg",
    },
    "case_d": {
        "question_id": "19-007055::purpose_solution",
        "label": "예시 D",
        "title": "High performance computing purpose+solution",
        "output": "patent_vector_space_case_d.svg",
    },
}

COLORS = {
    "query": "#7c3aed",
    "gold": "#16a34a",
    "top5": "#ea580c",
    "top20": "#2563eb",
    "top100": "#9ca3af",
}


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as fh:
        rows = pickle.load(fh)
    return {str(pid): vec.astype("float64", copy=False) for pid, vec in rows}


def load_results(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["question_id"]: row for row in rows}


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def reconstruct_query_direction(passage_vectors: np.ndarray, scores: np.ndarray) -> np.ndarray:
    # DPR uses dot products. Given top-k passage vectors P and scores s = Pq,
    # this recovers q's projection onto the span of the shown neighborhood.
    gram = passage_vectors @ passage_vectors.T
    ridge = max(float(np.trace(gram) / max(len(gram), 1)) * 1e-4, 1e-6)
    weights = np.linalg.solve(gram + np.eye(len(gram)) * ridge, scores)
    query = passage_vectors.T @ weights
    norm = np.linalg.norm(query)
    if norm == 0:
        return passage_vectors.mean(axis=0)
    median_norm = np.median(np.linalg.norm(passage_vectors, axis=1))
    return query / norm * median_norm


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(matrix)
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def scale_points(coords: np.ndarray, width: int, height: int, margin: int) -> np.ndarray:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = maxs - mins
    span[span == 0] = 1.0
    pad = span * 0.08
    mins -= pad
    maxs += pad
    span = maxs - mins
    x = margin + (coords[:, 0] - mins[0]) / span[0] * (width - margin * 2)
    y = height - margin - (coords[:, 1] - mins[1]) / span[1] * (height - margin * 2)
    return np.column_stack([x, y])


def point_category(pid: str, rank: int, gold_patent_id: str) -> str:
    if pid.startswith(f"{gold_patent_id}::"):
        return "gold"
    if rank <= 5:
        return "top5"
    if rank <= 20:
        return "top20"
    return "top100"


def short_title(value: str, limit: int = 58) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def star_points(cx: float, cy: float, r_outer: float = 10, r_inner: float = 4.5) -> str:
    points = []
    for i in range(10):
        angle = -np.pi / 2 + i * np.pi / 5
        radius = r_outer if i % 2 == 0 else r_inner
        points.append((cx + np.cos(angle) * radius, cy + np.sin(angle) * radius))
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def build_svg(case: dict, embeddings: dict[str, np.ndarray], results: dict[str, dict]) -> str:
    row = results[case["question_id"]]
    ctxs = row["ctxs"][:100]
    gold_patent_id = case["question_id"].split("::", 1)[0]
    top_ids = [ctx["id"] for ctx in ctxs if ctx["id"] in embeddings]
    gold_ids = sorted(pid for pid in embeddings if pid.startswith(f"{gold_patent_id}::"))
    selected_ids = list(dict.fromkeys(top_ids + gold_ids))
    rank_by_id = {ctx["id"]: i + 1 for i, ctx in enumerate(ctxs)}
    score_by_id = {ctx["id"]: float(ctx["score"]) for ctx in ctxs}

    top_vectors = np.vstack([embeddings[pid] for pid in top_ids])
    top_scores = np.array([score_by_id[pid] for pid in top_ids], dtype="float64")
    query_direction = reconstruct_query_direction(top_vectors, top_scores)

    vectors = np.vstack([embeddings[pid] for pid in selected_ids] + [query_direction])
    coords = pca_2d(vectors)
    width, height, margin = 920, 500, 56
    points = scale_points(coords, width, height, margin)
    query_point = points[-1]
    passage_points = points[:-1]

    first_gold_rank = min((rank_by_id.get(pid, 10**9) for pid in selected_ids if pid.startswith(f"{gold_patent_id}::")), default=None)
    gold_count_top100 = sum(1 for ctx in ctxs if ctx["id"].startswith(f"{gold_patent_id}::"))

    items = []
    for pid, (x, y) in zip(selected_ids, passage_points):
        rank = rank_by_id.get(pid, 999)
        category = point_category(pid, rank, gold_patent_id)
        radius = 6.5 if category == "gold" else 4.5 if category == "top5" else 3.8
        opacity = "0.95" if category in {"gold", "top5"} else "0.72"
        stroke = "#0f172a" if category == "gold" else "#ffffff"
        title = row["ctxs"][rank - 1]["title"] if rank <= len(row["ctxs"]) else pid
        tooltip = f"{pid} | rank {rank if rank != 999 else 'outside top100'} | {title}"
        items.append(
            (
                {"top100": 0, "top20": 1, "top5": 2, "gold": 3}[category],
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{COLORS[category]}" '
                f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="1.1">'
                f"<title>{html.escape(tooltip)}</title></circle>",
            )
        )

    items.sort(key=lambda pair: pair[0])
    first_rank_id = ctxs[0]["id"]
    label_rows = []
    for label, pid in [("rank 1", first_rank_id)]:
        if pid in selected_ids:
            idx = selected_ids.index(pid)
            x, y = passage_points[idx]
            label_rows.append(
                f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" class="point-label">{html.escape(label)}</text>'
            )
    if first_gold_rank is not None and first_gold_rank < 10**9:
        first_gold_id = next(pid for pid in selected_ids if pid.startswith(f"{gold_patent_id}::") and rank_by_id.get(pid) == first_gold_rank)
        idx = selected_ids.index(first_gold_id)
        x, y = passage_points[idx]
        label_rows.append(
            f'<text x="{x + 8:.1f}" y="{y + 16:.1f}" class="point-label gold-label">first gold r{first_gold_rank}</text>'
        )

    qx, qy = query_point
    legend = [
        ("query direction", COLORS["query"], "star"),
        ("gold patent chunks", COLORS["gold"], "circle"),
        ("DPR top-5 non-gold", COLORS["top5"], "circle"),
        ("DPR top-20 non-gold", COLORS["top20"], "circle"),
        ("DPR top-100 non-gold", COLORS["top100"], "circle"),
    ]
    legend_svg = [
        '<rect x="598" y="96" width="246" height="124" rx="8" fill="#ffffff" '
        'fill-opacity="0.94" stroke="#d7dce2" stroke-width="1"></rect>'
    ]
    lx, ly = 618, 120
    for i, (name, color, shape) in enumerate(legend):
        y = ly + i * 22
        if shape == "star":
            legend_svg.append(f'<polygon points="{star_points(lx + 7, y - 4, 7, 3.2)}" fill="{color}"></polygon>')
        else:
            legend_svg.append(f'<circle cx="{lx + 7}" cy="{y - 4}" r="5" fill="{color}"></circle>')
        legend_svg.append(f'<text x="{lx + 20}" y="{y}" class="legend-text">{html.escape(name)}</text>')

    title = f'{case["label"]}: DPR Colab vector space'
    subtitle = f'{short_title(row["question"], 95)}'
    stats = f'Top-100 local PCA, selected passages={len(selected_ids)}, gold chunks in top100={gold_count_top100}'

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
  <style>
    .bg {{ fill: #ffffff; }}
    .frame {{ fill: #fbfcfe; stroke: #d7dce2; stroke-width: 1; }}
    .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
    .title {{ font: 700 20px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #202124; }}
    .subtitle {{ font: 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #5f6368; }}
    .axis {{ font: 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #6b7280; }}
    .legend-text {{ font: 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #374151; }}
    .point-label {{ font: 700 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #111827; paint-order: stroke; stroke: #ffffff; stroke-width: 3px; }}
    .gold-label {{ fill: #14532d; }}
    .caption {{ font: 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #6b7280; }}
  </style>
  <rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="12"></rect>
  <text x="28" y="32" class="title">{html.escape(title)}</text>
  <text x="28" y="54" class="subtitle">{html.escape(subtitle)}</text>
  <rect class="frame" x="{margin}" y="82" width="{width - margin * 2}" height="{height - 144}" rx="8"></rect>
  <line class="grid" x1="{margin}" y1="{height/2:.1f}" x2="{width-margin}" y2="{height/2:.1f}"></line>
  <line class="grid" x1="{width/2:.1f}" y1="82" x2="{width/2:.1f}" y2="{height-62}"></line>
  <text x="{width - margin - 74}" y="{height - 72}" class="axis">PCA dim 1</text>
  <text x="{margin + 10}" y="101" class="axis">PCA dim 2</text>
  {''.join(svg for _, svg in items)}
  <polygon points="{star_points(qx, qy)}" fill="{COLORS["query"]}" stroke="#ffffff" stroke-width="1.4"><title>score-derived local query direction</title></polygon>
  <text x="{qx + 12:.1f}" y="{qy - 10:.1f}" class="point-label">query</text>
  {''.join(label_rows)}
  {''.join(legend_svg)}
  <text x="28" y="{height - 24}" class="caption">{html.escape(stats)} · passage points are actual Colab checkpoint embeddings; query marker is reconstructed from shown DPR scores.</text>
</svg>
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=Path("outputs/patent_dpr_colab_pro/patent_passages_colab_0"))
    parser.add_argument("--results", type=Path, default=Path("outputs/patent_dpr_colab_pro/dpr_results.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings)
    results = load_results(args.results)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES.values():
        svg = build_svg(case, embeddings, results)
        out_path = args.out_dir / case["output"]
        out_path.write_text(svg, encoding="utf-8")
        print(out_path)


if __name__ == "__main__":
    main()
