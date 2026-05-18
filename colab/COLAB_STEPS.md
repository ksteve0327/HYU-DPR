# Patent DPR Colab Pro Runbook

Notebook URL supplied by the user:

```text
https://colab.research.google.com/drive/1O0XOuSnVz9d5SUnn0yAt84CGph2ulIIJ?hl=ko
```

Use that notebook as the execution surface. Codex prepares the bundle locally;
Colab runs the GPU work; Codex verifies the returned artifacts locally.

The repo also contains a ready-to-run notebook:

```text
colab/patent_dpr_colab_pro.ipynb
```

Open that file in VSCode and attach a Colab GPU kernel, or upload/open it in
Colab directly. The cells below are mirrored in that notebook.

## 1. Upload the Bundle

From the local repo, create the clean bundle:

```bash
python3 tools/create_colab_bundle.py \
  --out colab/HYU-DPR_colab_input.tar.gz \
  --manifest colab/HYU-DPR_colab_input.manifest.json
```

Upload `colab/HYU-DPR_colab_input.tar.gz` to:

```text
/content/drive/MyDrive/HYU-DPR/HYU-DPR_colab_input.tar.gz
```

## 2. Colab Cell: Mount Drive and Extract

```python
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import tarfile

drive_root = Path("/content/drive/MyDrive/HYU-DPR")
bundle = drive_root / "HYU-DPR_colab_input.tar.gz"
workspace = Path("/content/HYU-DPR")
workspace.mkdir(parents=True, exist_ok=True)

with tarfile.open(bundle, "r:gz") as tar:
    tar.extractall(workspace)

print("Extracted:", workspace)
print("DPR exists:", (workspace / "DPR-main" / "train_dense_encoder.py").exists())
```

## 3. Colab Cell: Install Dependencies

```python
%cd /content/HYU-DPR/DPR-main
!python -m pip install --upgrade pip setuptools wheel
!python -m pip install "numpy<2" "transformers>=4.3" "hydra-core>=1.0.0" "omegaconf>=2.0.1" faiss-cpu jsonlines soundfile editdistance wget "spacy>=2.1.8"
!python -m pip install -e .
```

```python
import os
import subprocess
import sys

os.chdir("/content/HYU-DPR")
sys.path.insert(0, "/content/HYU-DPR/DPR-main")

import torch
import transformers
import dpr

print("cwd", os.getcwd())
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("dpr import ok")
subprocess.run(["nvidia-smi"], check=False)
subprocess.run([sys.executable, "colab/colab_runner.py", "gpu"], check=True)
```

`fp16=True` in this DPR code requires NVIDIA Apex. The default runbook uses
`fp16=False` to avoid Apex install failures. If A100/L4 `batch_size=128` fails
only because of memory, install Apex separately and rerun the same commands with
`--fp16`.

## 4. Colab Cell: Smoke Test

```python
%cd /content/HYU-DPR
!python colab/colab_runner.py gpu
!python colab/colab_runner.py smoke
```

Expected output folder:

```text
/content/drive/MyDrive/HYU-DPR/outputs/patent_dpr_colab_smoke/
```

Success means DPR imports, reads the patent JSON, runs a tiny training pass, and
writes a checkpoint plus `run_metadata.json`.

## 5. Colab Cell: Batch Feasibility Test

```python
%cd /content/HYU-DPR
!python colab/colab_runner.py profile-check --profile auto --steps 20
```

Profile selection:

```text
A100 or L4 -> actual batch_size=128, gradient_accumulation_steps=1
Other GPU -> actual batch_size=64, gradient_accumulation_steps=2
```

For T4/P100/V100 or any OOM case, rerun explicitly:

```python
!python colab/colab_runner.py profile-check --profile 64 --steps 20
!python colab/colab_runner.py profile-check --profile 32 --steps 20
```

## 6. Colab Cell: Full Training

For A100/L4, use the paper-style actual batch condition:

```python
%cd /content/HYU-DPR
!python colab/colab_runner.py train --profile 128
```

For Colab Pro auto selection with fallback:

```python
%cd /content/HYU-DPR
!python colab/colab_runner.py train --profile auto --auto-fallback
```

Required output:

```text
/content/drive/MyDrive/HYU-DPR/outputs/patent_dpr_colab_pro/dpr_biencoder.0
/content/drive/MyDrive/HYU-DPR/outputs/patent_dpr_colab_pro/train_dense_encoder.log
/content/drive/MyDrive/HYU-DPR/outputs/patent_dpr_colab_pro/run_metadata.json
```

The runner passes `val_av_rank_start_epoch=100` for the full run. This keeps the
40-epoch training on NLL validation and avoids the legacy DPR average-rank
validation path, which can hit CPU/CUDA device mismatch on recent Colab
PyTorch/Transformers versions.

## 7. Local Codex Verification After Download

Copy the Colab output folder into the local repo:

```text
/Users/dabeenkim/Documents/GitHub/HYU-DPR/outputs/patent_dpr_colab_pro/
```

Then verify locally:

```bash
python3 tools/verify_colab_artifacts.py \
  --run-dir outputs/patent_dpr_colab_pro
```

After verification, run local embedding generation, dense retrieval, and report
update using the same `data/patent/test.tsv` and `test_metadata.jsonl`.
