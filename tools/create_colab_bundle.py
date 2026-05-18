#!/usr/bin/env python3
"""Create a clean Colab input bundle for the patent DPR experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "colab" / "HYU-DPR_colab_input.tar.gz"

EXCLUDED_DIR_NAMES = {
    ".git",
    ".hydra",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".venv",
    ".venv-local",
    "__pycache__",
    "outputs",
}
EXCLUDED_FILE_NAMES = {
    ".DS_Store",
    "dpr_biencoder.0",
    "patent_passages_0",
    "dpr_results.json",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".swp"}
PATENT_DATA_SUFFIXES = {".json", ".jsonl", ".tsv", ".md"}


@dataclass(frozen=True)
class BundlePlan:
    files: list[Path]
    output_path: Path


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _excluded_by_parts(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def _should_include(path: Path, output_path: Path) -> bool:
    if path == output_path:
        return False
    rel = path.relative_to(REPO_ROOT)
    if _excluded_by_parts(rel):
        return False
    if path.name in EXCLUDED_FILE_NAMES:
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False

    dpr_root = REPO_ROOT / "DPR-main"
    patent_root = REPO_ROOT / "data" / "patent"
    colab_root = REPO_ROOT / "colab"
    report_file = REPO_ROOT / "reports" / "patent_dpr_report.html"

    if _is_under(path, dpr_root):
        return True
    if _is_under(path, patent_root):
        return path.suffix in PATENT_DATA_SUFFIXES
    if _is_under(path, colab_root):
        return path.suffix in {".py", ".md", ".ipynb"}
    if path == report_file:
        return True
    return False


def iter_bundle_files(output_path: Path) -> Iterable[Path]:
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if _should_include(path, output_path):
            yield path


def make_plan(output_path: Path) -> BundlePlan:
    files = sorted(iter_bundle_files(output_path), key=lambda p: p.relative_to(REPO_ROOT).as_posix())
    return BundlePlan(files=files, output_path=output_path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def create_bundle(plan: BundlePlan) -> dict:
    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(plan.output_path, "w:gz") as tar:
        for file_path in plan.files:
            tar.add(file_path, arcname=file_path.relative_to(REPO_ROOT).as_posix())
    return summarize(plan, include_sha=True)


def summarize(plan: BundlePlan, include_sha: bool = False) -> dict:
    total_bytes = sum(p.stat().st_size for p in plan.files)
    summary = {
        "bundle_path": str(plan.output_path),
        "file_count": len(plan.files),
        "uncompressed_bytes": total_bytes,
        "uncompressed_mb": round(total_bytes / 1024 / 1024, 2),
        "included_top_levels": sorted({p.relative_to(REPO_ROOT).parts[0] for p in plan.files}),
    }
    if include_sha and plan.output_path.exists():
        summary.update(
            {
                "bundle_bytes": plan.output_path.stat().st_size,
                "bundle_mb": round(plan.output_path.stat().st_size / 1024 / 1024, 2),
                "sha256": sha256_file(plan.output_path),
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="Output tar.gz path. Default: colab/HYU-DPR_colab_input.tar.gz",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the manifest summary without writing the bundle.")
    parser.add_argument(
        "--manifest",
        help="Optional JSON manifest path to write next to the bundle or in another location.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.out).expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    plan = make_plan(output_path.resolve())
    summary = summarize(plan)

    if not args.dry_run:
        summary = create_bundle(plan)

    if args.manifest:
        manifest_path = Path(args.manifest).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
