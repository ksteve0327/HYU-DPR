# Patent DPR Retrieval

This project was built to read and review the paper **Dense Passage Retrieval for Open-Domain Question Answering**, then directly implement its retriever training and Top-k retrieval evaluation pipeline on a custom Korean & United States of America patent dataset.

> 원본 특허 CSV, 생성된 DPR train/dev/test 데이터, passage corpus, checkpoint, embedding, 대용량 retrieval dump는 GitHub에 포함하지 않습니다.

## Project Objective

The main objective is not only to run an existing DPR repository, but to reproduce the core retriever idea from **Dense Passage Retrieval for Open-Domain Question Answering** after reviewing the paper:

- understand the DPR bi-encoder architecture;
- implement patent-specific preprocessing for passage retrieval;
- reproduce in-batch negative training with BM25 hard negatives;
- compare BM25 and fine-tuned DPR under the same Top-k retrieval evaluation setting;
- document where this patent setup differs from the original open-domain QA benchmark.

## What This Project Does

- 특허 1건을 passage retrieval corpus로 변환합니다.
- `purpose`, `solution`, `purpose_solution` 형태의 query를 생성합니다.
- BM25 baseline과 DPR fine-tuned retriever를 같은 test split에서 비교합니다.
- DPR 논문의 in-batch negative 구조를 특허 데이터 실험에 맞춰 재현합니다.
- 결과를 HTML 보고서와 상세 검색 예시 페이지로 정리합니다.

## Reports

- [Main HTML report](reports/patent_dpr_report.html)
- [Detailed retrieval examples](reports/patent_dpr_examples_detail.html)

두 보고서에는 왼쪽 sticky 목차, Table 2 형태 결과, line plot, 실제 query 예시, vector space 시각화, Q/P/S score matrix 설명이 포함되어 있습니다.

## Result Summary

Accuracy is measured as gold `patent_id` hit rate on the held-out test split. A query is counted as a hit when at least one retrieved passage in Top-k has the same `patent_id` as the query's source patent.

| Method | Top-5 | Top-10 | Top-20 | Top-50 | Top-100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 92.18 | 94.94 | 96.24 | 97.74 | 98.65 |
| DPR zero-shot | 0.10 | 0.25 | 0.45 | 1.00 | 2.56 |
| DPR 1% sampled fine-tuned | 0.20 | 0.60 | 1.00 | 2.66 | 5.56 |
| DPR full local | 87.22 | 91.88 | 95.99 | 98.60 | 99.40 |
| DPR Colab Pro checkpoint 1 | 98.30 | 99.10 | 99.55 | 99.90 | 100.00 |
| BM25+DPR Colab RRF | 98.40 | 99.25 | 99.55 | 99.90 | 100.00 |

## Important Evaluation Note

This experiment follows the retriever training structure and Top-k retrieval evaluation style of the DPR paper. However, the original DPR paper uses human-labeled QA answer strings as gold answers. This patent dataset does not include human-labeled QA, so the gold label is defined by `patent_id`.

Because of that, this project evaluates whether the retriever can find a passage from the correct patent, not whether a generated answer exactly matches a human-written answer.

## Repository Layout

```text
.
├── DPR-main/                 # DPR codebase plus patent-specific configs/scripts
│   ├── conf/                 # Hydra configs for mBERT and patent training runs
│   ├── dpr/                  # DPR modules
│   └── scripts/              # Patent preprocessing and evaluation scripts
├── colab/                    # Colab Pro notebook and runner
├── reports/                  # Portfolio HTML reports and SVG visualizations
├── tools/                    # Report/vector-space helper scripts
└── outputs/                  # Only small summary markdown files are intended for git
```

Excluded from git:

- `patent_rawdata.csv`
- `data/`
- model checkpoints such as `dpr_biencoder.*`
- dense embeddings such as `patent_passages_*`
- full retrieval dumps such as `dpr_results.json`
- local virtual environments, logs, Colab tar bundles, and credential-like files

## Reproduction Outline

The raw patent CSV is required locally but is intentionally not committed.

```bash
cd DPR-main
python -m pip install -e .
```

Prepare patent retrieval data:

```bash
python scripts/prepare_patent_dpr.py \
  --input ../patent_rawdata.csv \
  --output-dir ../data/patent \
  --seed 12345 \
  --train-ratio 0.8 \
  --dev-ratio 0.1
```

Run DPR training with the patent config:

```bash
python train_dense_encoder.py \
  'train_datasets=[patent_train]' \
  'dev_datasets=[patent_dev]' \
  datasets.patent_train.file=../data/patent/train.json \
  datasets.patent_dev.file=../data/patent/dev.json \
  train=biencoder_patent_colab_128 \
  encoder=hf_mbert \
  do_lower_case=False \
  output_dir=../outputs/patent_dpr_colab_pro \
  hydra.run.dir=.
```

Evaluate retrieval output:

```bash
python scripts/evaluate_patent_retrieval.py \
  --results ../outputs/patent_dpr_colab_pro/dpr_results.json \
  --metadata ../data/patent/test_metadata.jsonl \
  --out-json ../outputs/patent_dpr_colab_pro/dpr_eval_summary_topk.json \
  --out-md ../outputs/patent_dpr_colab_pro/table2_dpr_colab_topk.md \
  --method "DPR Colab Pro checkpoint 1" \
  --k-values 5 10 20 50 100
```

## Colab Pro Setup

The Colab workflow is in:

- [colab/patent_dpr_colab_pro.ipynb](colab/patent_dpr_colab_pro.ipynb)
- [colab/colab_runner.py](colab/colab_runner.py)
- [colab/COLAB_STEPS.md](colab/COLAB_STEPS.md)

The target reproduction condition is:

- encoder: `bert-base-multilingual-cased`
- sequence length: `256`
- batch size: `128`
- gradient accumulation: `1`
- effective batch size: `128`
- hard negative: `1` BM25 hard negative per query
- epochs: `40`

## Method Summary

For a batch of size `B=128`, DPR encodes questions and passages independently:

```text
Q in R^(128 x d)
P in R^(256 x d)  # 128 positive passages + 128 BM25 hard negatives
S = Q P^T in R^(128 x 256)
```

Each row of `S` is optimized with cross entropy so that the positive passage receives the highest score against in-batch negatives and hard negatives.

## Caveats

- The patent queries are template-generated, not human-written natural questions.
- The corpus is much smaller than DPR's Wikipedia-scale corpus.
- Patent titles are included in many queries and passages, so BM25 is a strong baseline.
- The reported metric is patent-level retrieval success, not answer exact match.
