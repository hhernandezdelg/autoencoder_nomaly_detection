# Assignment 2 — Autoencoder Anomaly Detection (MNIST)

This project implements several autoencoder variants in PyTorch for anomaly detection on MNIST. It covers data corruption, reconstruction, loss curves, and evaluation metrics. It is recommended, for this assignment, to run the `assignment_2.ipynb` for all the trainings, evaluations and figure generation.

---

## Requirements

- Python **3.11**
- pip (or uv)

---

## Setup (pip + venv)

### Linux / macOS
```bash
python3.11 -V
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

If `python3.11 -m venv` fails on Ubuntu/Debian:
```bash
sudo apt install -y python3.11-venv
```

### Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

---

## (Optional) Setup with uv
```bash
uv sync
source .venv/bin/activate
```

---

## Running the notebook in VS Code

1. Open this repo in VS Code.
2. Select the interpreter from this repo’s virtual environment:
   - Linux/macOS: `.venv/bin/python`
   - Windows: `.venv\Scripts\python.exe`
3. Open the notebook and select the same interpreter as the kernel.

If the kernel does not appear:
```bash
python -m pip install ipykernel
python -m ipykernel install --user --name cv-assignment-2025-2 --display-name "cv-assignment-2025-2 (py)"
```

---

## Running from CLI

```bash
python -u assignment_2/run_assignment2.py \
  --data-dir data \
  --output-dir assignment_2/output \
  --epochs 20 \
  --batch-size 64 \
  --val-frac 0.2 \
  --noise-level 0.8 \
  --noise-type gaussian \
  --corrupt-fraction 0.1 \
  --seed 42
```

---

## Outputs

All results are saved under:
```
assignment_2/output/
```

This includes:
- metrics.json
- plots (loss curves, reconstructions, confusion matrix)
- trained model checkpoints
