#!/usr/bin/env python3
"""Verify Colab-produced DPR training artifacts before local evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METADATA_KEYS = {
    "gpu_name",
    "gpu_total_memory_mb",
    "actual_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "num_train_epochs",
    "sequence_length",
    "hard_negatives",
    "started_at",
    "finished_at",
    "fallback",
}


def _resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify(run_dir: Path, mode: str, check_checkpoint: bool) -> dict:
    metadata_path = run_dir / "run_metadata.json"
    log_name = "train_smoke.log" if mode == "smoke" else "train_dense_encoder.log"
    log_path = run_dir / log_name
    checkpoint_path = run_dir / "dpr_biencoder.0"

    missing = [str(p) for p in (metadata_path, log_path, checkpoint_path) if not p.exists()]
    if missing:
        raise SystemExit("Missing required Colab artifact(s): " + ", ".join(missing))

    metadata = _read_json(metadata_path)
    missing_keys = sorted(REQUIRED_METADATA_KEYS - set(metadata))
    if missing_keys:
        raise SystemExit("run_metadata.json missing key(s): " + ", ".join(missing_keys))

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Traceback" in log_text or "RuntimeError: CUDA out of memory" in log_text:
        raise SystemExit(f"{log_path} contains a failure marker; inspect the Colab log before evaluation.")

    checkpoint_loadable = None
    if check_checkpoint:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_loadable = isinstance(checkpoint, dict)

    summary = {
        "run_dir": str(run_dir),
        "mode": mode,
        "gpu_name": metadata["gpu_name"],
        "gpu_total_memory_mb": metadata["gpu_total_memory_mb"],
        "actual_batch_size": metadata["actual_batch_size"],
        "gradient_accumulation_steps": metadata["gradient_accumulation_steps"],
        "effective_batch_size": metadata["effective_batch_size"],
        "num_train_epochs": metadata["num_train_epochs"],
        "sequence_length": metadata["sequence_length"],
        "hard_negatives": metadata["hard_negatives"],
        "fallback": metadata["fallback"],
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_mb": round(checkpoint_path.stat().st_size / 1024 / 1024, 2),
        "log_bytes": log_path.stat().st_size,
    }
    if checkpoint_loadable is not None:
        summary["checkpoint_loadable"] = checkpoint_loadable
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/patent_dpr_colab_pro", help="Colab output directory to verify.")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument(
        "--check-checkpoint",
        action="store_true",
        help="Load dpr_biencoder.0 with torch on CPU. This is slower and memory-heavy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = verify(_resolve(args.run_dir), args.mode, args.check_checkpoint)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
