# Setup

Verified on Windows 11 with an NVIDIA RTX 4060 (8 GB), CUDA 12.6, Python 3.11.

---

## 1. Environment

Python **3.11 or 3.12**. Not 3.10 (the code uses the `datetime.UTC` alias added
in 3.11) and not 3.13 (several dependencies have no wheels for it yet).

```bash
conda create -n neuroscan python=3.11 -y
conda activate neuroscan
```

## 2. PyTorch first, from the CUDA index

This must come **before** `requirements.txt`. Installing PyTorch from PyPI
gives the CPU-only build, and it will not be replaced by a later install.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

For CPU-only machines:

```bash
pip install torch torchvision
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect `True` on a CUDA machine. If it prints `False` with a GPU present, the
CPU wheel was installed, `pip uninstall torch torchvision` and repeat with the
index URL.

> **Note on the framework choice.** The original design specified
> TensorFlow/Keras. This implementation uses PyTorch because TensorFlow has had
> no native-Windows GPU support since 2.11, which would have left the project's
> GPU idle. Every method, custom CNN, VGG16 and EfficientNetB0 transfer
> learning, CLAHE preprocessing, Grad-CAM, the same metrics, is unchanged; only
> the library differs.

## 3. Remaining dependencies

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

## 4. Data

```bash
python scripts/download_data.py --list
python scripts/download_data.py --dataset br35h
```

Automatic download needs Kaggle API credentials at `~/.kaggle/kaggle.json`
(Kaggle → Settings → API → Create New Token). Without them the script prints
manual download instructions and the expected directory layout.

Verify what is on disk:

```bash
python scripts/download_data.py --verify-only
```

## 5. Knowledge base index

```bash
python scripts/build_index.py --rebuild
```

Downloads the embedding model (~1.1 GB) on first run, then builds the FAISS
index. Check it:

```bash
python scripts/build_index.py --stats
python scripts/build_index.py --query "ring enhancing lesion in Nepal"
```

## 6. Local LLM (optional but recommended)

The system runs without it, the advisory falls back to showing retrieved
source text verbatim, which is honest and still useful. Generation makes it
considerably better.

Install [Ollama](https://ollama.com/download), then:

```bash
ollama pull llama3.1:8b
```

`llama3.1:8b` quantised fits comfortably in 8 GB of VRAM. On a smaller GPU or
CPU-only machine, `qwen2.5:3b` is a reasonable substitute, set it in
`configs/default.yaml` under `llm.model`, or via `NEUROSCAN_LLM_MODEL`.

Ollama serves on `http://localhost:11434` by default and must be running.

## 7. Train

Validate the data pipeline first, this catches missing data, broken class
folders and leakage problems in seconds rather than after an hour of training:

```bash
python scripts/train.py --config efficientnet_b0 --dry-run
```

Then:

```bash
python scripts/train.py --config efficientnet_b0
python scripts/train.py --compare-all          # all three architectures
```

## 8. Run

```bash
python scripts/run_app.py
```

Then open <http://127.0.0.1:5000>.

---

## Environment variables

| Variable | Purpose |
|---|---|
| `NEUROSCAN_SECRET_KEY` | Flask session key. Without it an ephemeral key is generated and sessions do not survive a restart. |
| `NEUROSCAN_LLM_PROVIDER` | `ollama`, `huggingface`, or `echo` |
| `NEUROSCAN_LLM_MODEL` | Model name, e.g. `qwen2.5:3b` |
| `NEUROSCAN_LLM_BASE_URL` | Ollama URL if not the default |
| `NEUROSCAN_DEVICE` | `auto`, `cuda`, `cpu` |
| `NEUROSCAN_WEB_PORT` | Port override |
| `NEUROSCAN_DATA_ROOT` | Data directory override |

Copy `.env.example` to `.env` if you prefer a file.

---

## Troubleshooting

**`torch.cuda.is_available()` is False with a GPU present**
The CPU wheel is installed. Uninstall and reinstall from the CUDA index (step 2).

**`No FAISS index at ...`**
Run `python scripts/build_index.py --rebuild`.

**`Index was built with embedding model X but the configuration specifies Y`**
Vectors from different embedding models are not comparable, so the mismatch is
refused rather than producing silently meaningless retrieval. Rebuild the index.

**`No trained model found`**
Train one: `python scripts/train.py --config efficientnet_b0`.

**Advisory says the language model is unavailable**
Ollama is not running or the model is not pulled. `ollama serve`, then
`ollama pull llama3.1:8b`. The system remains usable in the meantime.

**CUDA out of memory**
Lower `training.batch_size` (VGG16 is the heaviest; try 8), or set
`training.mixed_precision: true` if it was disabled.

**Nepali text is missing from the PDF**
No Devanagari font was found. Install one (Noto Sans Devanagari), or place a
`.ttf` in `src/neuroscan/resources/fonts/`.

**Windows: `DataLoader` worker errors**
Set `training.num_workers: 0`.
