# Patent DPR Retrieval Experiment

This repo extension prepares the AI semiconductor patent CSV for a DPR-style
retrieval experiment and compares BM25 with a fine-tuned DPR bi-encoder.

## Data

Source data:

```text
../patent_rawdata.csv
```

Generated artifacts:

```text
../data/patent/passages.tsv
../data/patent/train.json
../data/patent/dev.json
../data/patent/test.tsv
../data/patent/test_metadata.jsonl
../data/patent/bm25_results.json
../data/patent/table2_bm25.md
```

`passages.tsv` uses DPR's `id<TAB>text<TAB>title` context format. Train/dev
JSON files follow DPR's bi-encoder format with one positive context and one
BM25-mined hard negative per question. The test split is evaluated by
gold-patent hit rate: a query is correct at Top-k when any retrieved passage
belongs to the same `patent_id` as the held-out query.

## Prepare Data and BM25

Run from `DPR-main`:

```bash
python scripts/prepare_patent_dpr.py \
  --input ../patent_rawdata.csv \
  --output-dir ../data/patent \
  --seed 12345 \
  --chunk-tokens 140 \
  --chunk-overlap 20 \
  --top-k 100
```

This creates the patent passage corpus, train/dev/test splits, DPR training
JSON, BM25 hard negatives, and the BM25 Table 2 baseline.

## Fine-Tune DPR

Use a CUDA GPU environment with Python 3.10. Install DPR from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Train the bi-encoder:

```bash
DATA_DIR="$(cd ../data/patent && pwd)"

python train_dense_encoder.py \
  train_datasets=[patent_train] \
  dev_datasets=[patent_dev] \
  datasets.patent_train.file="$DATA_DIR/train.json" \
  datasets.patent_dev.file="$DATA_DIR/dev.json" \
  train=biencoder_patent \
  encoder=hf_mbert \
  do_lower_case=False \
  output_dir=../outputs/patent_dpr
```

The patent training profile uses `batch_size=16` and
`gradient_accumulation_steps=8`, matching an effective batch size of 128.

For local Apple Silicon training without CUDA, use the local profile. It uses
MPS when available and a smaller batch size so it can run on an M2-class Mac:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
  .venv-local/bin/python train_dense_encoder.py \
  'train_datasets=[patent_train]' \
  'dev_datasets=[patent_dev]' \
  'train_sampling_rates=[0.01]' \
  datasets.patent_train.file="$(cd ../data/patent && pwd)/train.json" \
  datasets.patent_dev.file="$(cd ../data/patent && pwd)/local_dev.json" \
  train=biencoder_patent_local \
  encoder=hf_mbert \
  do_lower_case=False \
  output_dir=../outputs/patent_dpr_local_sampled
```

The sampled local run writes:

```text
../outputs/patent_dpr_local_sampled/dpr_biencoder.0
```

## Colab Pro Re-Training

Use the ready notebook from the repository:

```text
../colab/patent_dpr_colab_pro.ipynb
```

Open it in VSCode and attach a Colab GPU kernel, or open/upload it in Colab.
Before running it, create the clean input bundle locally:

```bash
python3 ../tools/create_colab_bundle.py \
  --out ../colab/HYU-DPR_colab_input.tar.gz \
  --manifest ../colab/HYU-DPR_colab_input.manifest.json
```

Upload the bundle to Google Drive:

```text
/content/drive/MyDrive/HYU-DPR/HYU-DPR_colab_input.tar.gz
```

The notebook runs this sequence:

```text
Drive mount -> bundle extract -> dependency install -> import/GPU check
-> smoke training -> batch feasibility check -> full 40-epoch training
```

A100/L4 profile:

```text
batch_size=128
gradient_accumulation_steps=1
effective_batch_size=128
num_train_epochs=40
sequence_length=256
hard_negatives=1
```

T4 fallback profiles:

```text
batch_size=64, gradient_accumulation_steps=2
batch_size=32, gradient_accumulation_steps=4
```

This DPR clone uses NVIDIA Apex when `fp16=True`; the notebook defaults to
`fp16=False` to avoid Apex install failures. If `batch_size=128` fails only from
memory pressure, install Apex in Colab and rerun the same profile with `--fp16`.

Required Colab output:

```text
/content/drive/MyDrive/HYU-DPR/outputs/patent_dpr_colab_pro/dpr_biencoder.0
/content/drive/MyDrive/HYU-DPR/outputs/patent_dpr_colab_pro/train_dense_encoder.log
/content/drive/MyDrive/HYU-DPR/outputs/patent_dpr_colab_pro/run_metadata.json
```

After copying that output folder back to the local repo, verify it:

```bash
python3 ../tools/verify_colab_artifacts.py \
  --run-dir ../outputs/patent_dpr_colab_pro
```

Current Colab A100 result note:

```text
GPU: NVIDIA A100-SXM4-80GB
actual batch_size: 128
gradient_accumulation_steps: 1
fp16: False
target epochs: 40
metadata status: failed at epoch 30 average-rank validation
evaluated local checkpoint: ../outputs/patent_dpr_colab_pro/dpr_biencoder.1
reason: best dev NLL among locally available checkpoints 0-4
Top-5 / Top-20 / Top-100: 98.30 / 99.55 / 100.00
```

## Dense Retrieval Evaluation

Generate passage embeddings with the best checkpoint:

```bash
DATA_DIR="$(cd ../data/patent && pwd)"

python generate_dense_embeddings.py \
  model_file=../outputs/patent_dpr/dpr_biencoder.best \
  ctx_src=patent_passages \
  ctx_sources.patent_passages.file="$DATA_DIR/passages.tsv" \
  encoder=hf_mbert \
  do_lower_case=False \
  out_file=../outputs/patent_dpr/patent_passages
```

Run DPR retrieval:

```bash
DATA_DIR="$(cd ../data/patent && pwd)"

python dense_retriever.py \
  model_file=../outputs/patent_dpr/dpr_biencoder.best \
  qa_dataset=patent_test \
  ctx_datatsets=[patent_passages] \
  datasets.patent_test.file="$DATA_DIR/test.tsv" \
  ctx_sources.patent_passages.file="$DATA_DIR/passages.tsv" \
  encoded_ctx_files=[\"../outputs/patent_dpr/patent_passages_*\"] \
  encoder=hf_mbert \
  do_lower_case=False \
  n_docs=100 \
  out_file=../outputs/patent_dpr/dpr_results.json
```

Evaluate DPR by gold patent id:

```bash
python scripts/evaluate_patent_retrieval.py \
  --results ../outputs/patent_dpr/dpr_results.json \
  --metadata ../data/patent/test_metadata.jsonl \
  --out-json ../outputs/patent_dpr/dpr_eval_summary.json \
  --out-md ../outputs/patent_dpr/table2_dpr.md \
  --method DPR
```

Create the combined BM25 vs DPR table:

```bash
python scripts/make_patent_table2.py \
  --bm25 ../data/patent/bm25_results.json \
  --dpr ../outputs/patent_dpr/dpr_eval_summary.json \
  --out ../outputs/patent_dpr/table2_patent_retrieval.md
```

## Method Note: In-Batch Negative Matrix

For batch size `B` and embedding dimension `d`, DPR encodes questions and
positive passages as:

```text
Q = [q1, q2, ..., qB] in R^(B x d)
P = [p1+, p2+, ..., pB+] in R^(B x d)
S = QP^T in R^(B x B)
```

The diagonal entries are positive pairs. Off-diagonal entries are in-batch
negatives. With one BM25 hard negative per question:

```text
P = [p1+, h1, p2+, h2, ..., pB+, hB] in R^(2B x d)
S = QP^T in R^(B x 2B)
```

Training minimizes cross entropy over each row of `S`, selecting the matching
positive passage.
