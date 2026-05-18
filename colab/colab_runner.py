#!/usr/bin/env python3
"""Colab runner for the patent DPR training workflow.

Run this from the extracted bundle root, for example:

    python colab/colab_runner.py smoke
    python colab/colab_runner.py profile-check --profile auto
    python colab/colab_runner.py train --profile auto
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path("/content/HYU-DPR")
DRIVE_ROOT = Path("/content/drive/MyDrive/HYU-DPR")
PROFILE_CONFIGS = {
    "128": {"config": "biencoder_patent_colab_128", "batch": 128, "grad_accum": 1},
    "64": {"config": "biencoder_patent_colab_64", "batch": 64, "grad_accum": 2},
    "32": {"config": "biencoder_patent_colab_32", "batch": 32, "grad_accum": 4},
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_cmd(cmd: list[str], cwd: Path, log_path: Path | None = None) -> int:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONPATH", ".")
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"$ {rendered}", flush=True)

    with subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        log_fh = log_path.open("a", encoding="utf-8") if log_path else None
        try:
            if log_fh:
                log_fh.write(f"$ {rendered}\n")
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                if log_fh:
                    log_fh.write(line)
            return proc.wait()
        finally:
            if log_fh:
                log_fh.close()


def detect_gpu() -> dict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True).strip().splitlines()
    except Exception as exc:  # noqa: BLE001 - Colab CPU/no driver path should be explicit in metadata.
        return {"gpu_name": "UNAVAILABLE", "gpu_total_memory_mb": 0, "nvidia_smi_error": str(exc)}

    if not output:
        return {"gpu_name": "UNAVAILABLE", "gpu_total_memory_mb": 0}
    first = output[0]
    parts = [part.strip() for part in first.split(",")]
    gpu_name = parts[0]
    try:
        memory_mb = int(parts[1])
    except (IndexError, ValueError):
        memory_mb = 0
    return {"gpu_name": gpu_name, "gpu_total_memory_mb": memory_mb}


def choose_profile(profile: str, gpu_name: str) -> str:
    if profile != "auto":
        return profile
    normalized = gpu_name.upper()
    if "A100" in normalized or "L4" in normalized:
        return "128"
    return "64"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_subset(src: Path, dst: Path, rows: int) -> int:
    data = read_json(src)
    selected = data[: min(rows, len(data))]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(selected)


def ensure_workspace(workspace: Path) -> None:
    if not (workspace / "DPR-main" / "train_dense_encoder.py").exists():
        raise SystemExit(f"DPR workspace not found at {workspace}. Extract the bundle first.")
    if not (workspace / "data" / "patent" / "train.json").exists():
        raise SystemExit(f"Patent train.json not found under {workspace / 'data' / 'patent'}.")


def ensure_data_subdir(workspace: Path, data_subdir: str) -> Path:
    data_dir = workspace / "data" / data_subdir
    if not (data_dir / "train.json").exists():
        raise SystemExit(f"Train data not found: {data_dir / 'train.json'}")
    if not (data_dir / "dev.json").exists():
        raise SystemExit(f"Dev data not found: {data_dir / 'dev.json'}")
    return data_dir


def base_train_cmd(
    workspace: Path,
    output_dir: Path,
    train_file: Path,
    dev_file: Path,
    train_config: str,
    fp16: bool,
    extra_overrides: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "train_dense_encoder.py",
        "train_datasets=[patent_train]",
        "dev_datasets=[patent_dev]",
        f"datasets.patent_train.file={train_file}",
        f"datasets.patent_dev.file={dev_file}",
        f"train={train_config}",
        "encoder=hf_mbert",
        "do_lower_case=False",
        f"output_dir={output_dir}",
        f"fp16={str(fp16)}",
        "hydra.run.dir=.",
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)
    return cmd


def metadata_for_run(
    command_name: str,
    workspace: Path,
    output_dir: Path,
    selected_profile: str,
    fp16: bool,
    started_at: str,
    finished_at: str | None,
    status: str,
    fallback: bool,
    fallback_reason: str | None,
    return_code: int | None,
    extra: dict | None = None,
) -> dict:
    gpu = detect_gpu()
    if selected_profile in PROFILE_CONFIGS:
        profile = PROFILE_CONFIGS[selected_profile]
        batch = profile["batch"]
        grad_accum = profile["grad_accum"]
        epochs = 40
    else:
        batch = 4
        grad_accum = 1
        epochs = 1

    data = {
        "command": command_name,
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "gpu_name": gpu["gpu_name"],
        "gpu_total_memory_mb": gpu["gpu_total_memory_mb"],
        "actual_batch_size": batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": batch * grad_accum,
        "num_train_epochs": epochs,
        "sequence_length": 256,
        "hard_negatives": 1,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "fp16": fp16,
        "fp16_note": "DPR requires NVIDIA Apex when fp16=True.",
        "return_code": return_code,
    }
    if "nvidia_smi_error" in gpu:
        data["nvidia_smi_error"] = gpu["nvidia_smi_error"]
    if extra:
        data.update(extra)
    return data


def prepare_answer_string(args: argparse.Namespace) -> None:
    workspace = args.workspace
    ensure_workspace(workspace)
    raw_csv = workspace / "patent_rawdata.csv"
    if not raw_csv.exists():
        raise SystemExit(
            f"{raw_csv} not found. Recreate and upload the latest Colab bundle because paper-style preparation requires the raw CSV."
        )
    output_dir = workspace / "data" / args.output_subdir
    cmd = [
        sys.executable,
        "DPR-main/scripts/prepare_patent_dpr.py",
        "--input",
        str(raw_csv),
        "--output-dir",
        str(output_dir),
        "--positive-mode",
        "answer_string",
        "--hard-negative-mode",
        "bm25_answer_free",
        "--top-k",
        str(args.top_k),
    ]
    rc = run_cmd(cmd, workspace)
    if rc != 0:
        raise SystemExit(rc)
    manifest = output_dir / "split_manifest.json"
    if manifest.exists():
        print(manifest.read_text(encoding="utf-8"))


def smoke(args: argparse.Namespace) -> None:
    workspace = args.workspace
    ensure_workspace(workspace)
    data_dir = ensure_data_subdir(workspace, args.data_subdir)
    smoke_dir = workspace / "data" / "patent_colab_smoke"
    train_rows = make_subset(data_dir / "train.json", smoke_dir / "train.json", args.train_rows)
    dev_rows = make_subset(data_dir / "dev.json", smoke_dir / "dev.json", args.dev_rows)

    output_dir = args.drive_root / "outputs" / "patent_dpr_colab_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_smoke.log"
    metadata_path = output_dir / "run_metadata.json"
    started_at = utc_now()
    write_json(
        metadata_path,
        metadata_for_run(
            "smoke",
            workspace,
            output_dir,
            "smoke",
            args.fp16,
            started_at,
            None,
            "running",
            False,
            None,
            None,
            {"train_rows": train_rows, "dev_rows": dev_rows, "data_subdir": args.data_subdir},
        ),
    )
    cmd = base_train_cmd(
        workspace,
        output_dir,
        smoke_dir / "train.json",
        smoke_dir / "dev.json",
        "biencoder_patent_colab_smoke",
        args.fp16,
        ["val_av_rank_start_epoch=100"],
    )
    rc = run_cmd(cmd, workspace / "DPR-main", log_path)
    status = "success" if rc == 0 else "failed"
    write_json(
        metadata_path,
        metadata_for_run(
            "smoke",
            workspace,
            output_dir,
            "smoke",
            args.fp16,
            started_at,
            utc_now(),
            status,
            False,
            None,
            rc,
            {"train_rows": train_rows, "dev_rows": dev_rows, "data_subdir": args.data_subdir},
        ),
    )
    raise SystemExit(rc)


def profile_check(args: argparse.Namespace) -> None:
    workspace = args.workspace
    ensure_workspace(workspace)
    gpu = detect_gpu()
    selected = choose_profile(args.profile, gpu["gpu_name"])
    profile = PROFILE_CONFIGS[selected]

    data_dir = ensure_data_subdir(workspace, args.data_subdir)
    check_dir = workspace / "data" / f"patent_colab_profile_{selected}"
    train_rows = make_subset(data_dir / "train.json", check_dir / "train.json", profile["batch"] * args.steps)
    dev_rows = make_subset(data_dir / "dev.json", check_dir / "dev.json", max(profile["batch"], args.dev_rows))

    output_dir = args.drive_root / "outputs" / f"patent_dpr_colab_profile_{selected}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_profile_check.log"
    metadata_path = output_dir / "run_metadata.json"
    started_at = utc_now()
    cmd = base_train_cmd(
        workspace,
        output_dir,
        check_dir / "train.json",
        check_dir / "dev.json",
        profile["config"],
        args.fp16,
        [
            "train.num_train_epochs=1",
            "train.warmup_steps=1",
            "val_av_rank_start_epoch=100",
        ],
    )
    write_json(
        metadata_path,
        metadata_for_run(
            "profile-check",
            workspace,
            output_dir,
            selected,
            args.fp16,
            started_at,
            None,
            "running",
            False,
            None,
            None,
            {"train_rows": train_rows, "dev_rows": dev_rows, "profile_check_steps": args.steps, "data_subdir": args.data_subdir},
        ),
    )
    rc = run_cmd(cmd, workspace / "DPR-main", log_path)
    status = "success" if rc == 0 else "failed"
    write_json(
        metadata_path,
        metadata_for_run(
            "profile-check",
            workspace,
            output_dir,
            selected,
            args.fp16,
            started_at,
            utc_now(),
            status,
            False,
            None,
            rc,
            {"train_rows": train_rows, "dev_rows": dev_rows, "profile_check_steps": args.steps, "data_subdir": args.data_subdir},
        ),
    )
    raise SystemExit(rc)


def train(args: argparse.Namespace) -> None:
    workspace = args.workspace
    ensure_workspace(workspace)
    gpu = detect_gpu()
    selected = choose_profile(args.profile, gpu["gpu_name"])
    profile_order = [selected]
    if args.auto_fallback and selected == "128":
        profile_order.extend(["64", "32"])
    elif args.auto_fallback and selected == "64":
        profile_order.append("32")

    data_dir = ensure_data_subdir(workspace, args.data_subdir)
    output_dir = args.drive_root / "outputs" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_dense_encoder.log"
    metadata_path = output_dir / "run_metadata.json"
    started_at = utc_now()

    fallback = False
    fallback_reason = None
    final_rc = 1
    final_profile = selected

    for idx, candidate in enumerate(profile_order):
        final_profile = candidate
        profile = PROFILE_CONFIGS[candidate]
        if idx > 0:
            fallback = True
            fallback_reason = "Previous profile failed, most likely due to GPU memory or runtime limits."
        write_json(
            metadata_path,
            metadata_for_run(
                "train",
                workspace,
                output_dir,
                candidate,
                args.fp16,
                started_at,
                None,
                "running",
                fallback,
                fallback_reason,
                None,
                {"profile_config": profile["config"], "data_subdir": args.data_subdir, "num_train_epochs": args.epochs},
            ),
        )
        cmd = base_train_cmd(
            workspace,
            output_dir,
            data_dir / "train.json",
            data_dir / "dev.json",
            profile["config"],
            args.fp16,
            [f"train.num_train_epochs={args.epochs}", "val_av_rank_start_epoch=100"],
        )
        final_rc = run_cmd(cmd, workspace / "DPR-main", log_path)
        if final_rc == 0:
            break
        if not args.auto_fallback:
            break

    status = "success" if final_rc == 0 else "failed"
    write_json(
        metadata_path,
        metadata_for_run(
            "train",
            workspace,
            output_dir,
            final_profile,
            args.fp16,
            started_at,
            utc_now(),
            status,
            fallback,
            fallback_reason,
            final_rc,
            {"profile_config": PROFILE_CONFIGS[final_profile]["config"], "data_subdir": args.data_subdir, "num_train_epochs": args.epochs},
        ),
    )
    raise SystemExit(final_rc)


def evaluate(args: argparse.Namespace) -> None:
    workspace = args.workspace
    ensure_workspace(workspace)
    output_dir = args.drive_root / "outputs" / "patent_dpr_colab_pro"
    checkpoint = output_dir / args.checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    data_dir = workspace / "data" / "patent"
    log_path = output_dir / "postprocess_eval.log"
    encoded_prefix = output_dir / "patent_passages_colab_best"
    results_path = output_dir / "dpr_results.json"
    eval_json = output_dir / "dpr_eval_summary.json"
    eval_md = output_dir / "table2_dpr_colab.md"

    commands = [
        [
            sys.executable,
            "generate_dense_embeddings.py",
            f"model_file={checkpoint}",
            "ctx_src=patent_passages",
            f"ctx_sources.patent_passages.file={data_dir / 'passages.tsv'}",
            "encoder=hf_mbert",
            "do_lower_case=False",
            f"out_file={encoded_prefix}",
            f"batch_size={args.embedding_batch_size}",
            "hydra.run.dir=.",
        ],
        [
            sys.executable,
            "dense_retriever.py",
            f"model_file={checkpoint}",
            "qa_dataset=patent_test",
            "ctx_datatsets=[patent_passages]",
            f"datasets.patent_test.file={data_dir / 'test.tsv'}",
            f"ctx_sources.patent_passages.file={data_dir / 'passages.tsv'}",
            f"encoded_ctx_files=[\"{encoded_prefix}_*\"]",
            "encoder=hf_mbert",
            "do_lower_case=False",
            "n_docs=100",
            f"batch_size={args.retrieval_batch_size}",
            f"out_file={results_path}",
            "hydra.run.dir=.",
        ],
        [
            sys.executable,
            "scripts/evaluate_patent_retrieval.py",
            "--results",
            str(results_path),
            "--metadata",
            str(data_dir / "test_metadata.jsonl"),
            "--out-json",
            str(eval_json),
            "--out-md",
            str(eval_md),
            "--method",
            args.method,
        ],
    ]

    for cmd in commands:
        rc = run_cmd(cmd, workspace / "DPR-main", log_path)
        if rc != 0:
            raise SystemExit(rc)


def print_gpu(args: argparse.Namespace) -> None:
    print(json.dumps(detect_gpu(), indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--drive-root", type=Path, default=DRIVE_ROOT)
    parser.add_argument("--fp16", action="store_true", help="Pass fp16=True to DPR. Requires NVIDIA Apex.")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("gpu").set_defaults(func=print_gpu)

    prepare_parser = sub.add_parser("prepare-answer-string", help="Create DPR paper-style answer-string positive/hard-negative data.")
    prepare_parser.add_argument("--output-subdir", default="patent_answer_string")
    prepare_parser.add_argument("--top-k", type=int, default=100)
    prepare_parser.set_defaults(func=prepare_answer_string)

    smoke_parser = sub.add_parser("smoke", help="Run a tiny end-to-end training/checkpoint test.")
    smoke_parser.add_argument("--data-subdir", default="patent")
    smoke_parser.add_argument("--train-rows", type=int, default=64)
    smoke_parser.add_argument("--dev-rows", type=int, default=8)
    smoke_parser.set_defaults(func=smoke)

    check_parser = sub.add_parser("profile-check", help="Run a short batch-size feasibility test.")
    check_parser.add_argument("--data-subdir", default="patent")
    check_parser.add_argument("--profile", choices=["auto", "128", "64", "32"], default="auto")
    check_parser.add_argument("--steps", type=int, default=20)
    check_parser.add_argument("--dev-rows", type=int, default=128)
    check_parser.set_defaults(func=profile_check)

    train_parser = sub.add_parser("train", help="Run Colab Pro DPR training.")
    train_parser.add_argument("--data-subdir", default="patent")
    train_parser.add_argument("--output-name", default="patent_dpr_colab_pro")
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--profile", choices=["auto", "128", "64", "32"], default="auto")
    train_parser.add_argument("--auto-fallback", action="store_true")
    train_parser.set_defaults(func=train)

    eval_parser = sub.add_parser("evaluate", help="Generate embeddings, retrieve, and evaluate a Colab checkpoint.")
    eval_parser.add_argument("--checkpoint", default="dpr_biencoder.5")
    eval_parser.add_argument("--embedding-batch-size", type=int, default=128)
    eval_parser.add_argument("--retrieval-batch-size", type=int, default=128)
    eval_parser.add_argument("--method", default="DPR Colab Pro fine-tuned")
    eval_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
