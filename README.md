# MVG-SDNet
A traffic flow forecasting codebase that combines **multi-view variational graph autoencoder (MVGAE)** structure representation pretraining with a **structure-aware hybrid spatiotemporal predictor (HybridSTPredictor)**.


## Method Overview

Two-stage pipeline:

1. **Structure representation pretraining (MVGAE)**  
   Builds multi-view graphs (physical / second-order / role / direction) and node static features from the road-network distance edge list, then trains a multi-view variational graph encoder unsupervised. Exports node embeddings \(Z_{\mathrm{init}}\) (\(Z^s\)), latent adjacency \(A^{\mathrm{latent}}\), and node-level structure uncertainty / confidence \(U\), \(C_v\). When GPS is unavailable, 2D coordinates for the direction view are estimated via shortest-path distances + Classical MDS.

2. **Hybrid spatiotemporal forecasting (Hybrid)**  
   Initializes learnable node embeddings with \(Z_{\mathrm{init}}\), and builds the structure prior \(\hat{A}^{\mathrm{prior}}\) from \(A^{\mathrm{phy}}\), \(A^{\mathrm{latent}}\), and \(C_v\). Multi-period historical traffic sequences (week / day / hour) are fed into HybridSTPredictor (default: dual-scale dynamic spatial branch + structure-aware temporal branch + gated fusion; fallbacks: Graph WaveNet / Temporal Transformer) for multi-step traffic prediction. Training supports a Warm → Joint staged schedule.

Main entry point: `train_hybrid.py` (for full experiments, stage-1 pretraining runs automatically when `z_init.npy` is missing).

## Experimental Environment

Recommended: Linux + NVIDIA GPU (typical setup used in development/experiments; adjust to your machine).

| Item | Suggested version / notes |
|---|---|
| OS | Linux (e.g., Ubuntu); Windows is also fine |
| Python | 3.8+ |
| PyTorch | GPU build matching your CUDA driver (e.g., CUDA 12.x → `cu124` wheels) |
| torch-geometric | ≥ 2.3 |
| Other dependencies | See `requirements.txt` (`numpy`, `scikit-learn`, `tensorboardX`, etc.) |
| GPU | A single GPU is sufficient; PEMS07 (883 nodes) is memory-heavy—recommend ≥ 24GB; watch for OOM when running multiple jobs in parallel |

```bash
# Example: PyTorch GPU build for CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Data Preparation

Place datasets under `data/`, for example:

```text
data/
├── PEMS07/   # PEMS07.npz, PEMS07.csv
└── PEMS08/   # PEMS08.npz, PEMS08.csv
```

Traffic matrices (`.npz`) must be provided by the user (key `data`, shape `(T, N, F)`); distance CSV files are usually included in the repo.  
Generate ASTGCN-style sliding-window files when needed:

```bash
python prepareData.py --config configurations/PEMS08_multi_period.conf
```


## How to Run

Run all commands from the **project root**:

```bash
cd /path/to/MVG-SDNet
```

### 1. Full experiments (recommended entry)

Configs: `configurations/*_multi_period.conf`. When `auto_pretrain=True`, missing `z_init.npy` triggers MVGAE pretraining before Hybrid training.

```bash
python train_hybrid.py --config configurations/PEMS07_multi_period.conf
python train_hybrid.py --config configurations/PEMS08_multi_period.conf
```

Stage-1 only (optional):

```bash
python train_mvgae_pretrain.py --config configurations/PEMS08_multi_period.conf
```

### 2. GPU selection and background jobs

Set the visible GPU via `[Training] ctx` in the config (e.g., `0`). Background example:

```bash
mkdir -p logs
PYTHONUNBUFFERED=1 nohup python -u train_hybrid.py \
  --config configurations/PEMS08_multi_period.conf \
  > logs/PEMS04_full.log 2>&1 &

tail -f logs/PEMS08_full.log
```
Note: `logs/*.log` only captures redirected stdout/stderr; checkpoints and metrics are still saved under `experiments/`.


## Repository Layout (brief)

```text
MVG-SDNet/
├── train_hybrid.py          # Main training entry
├── train_mvgae_pretrain.py  # MVGAE pretraining only
├── prepareData.py           # Data slicing / preparation
├── configurations/          # Experiment configs + per-folder READMEs
├── model/                   # MVGAE, Hybrid, dual-scale spatial, structure-aware temporal, GWN
├── lib/                     # Graph construction, pretrain, prior, losses, experiment I/O
├── tools/                   # Ablation, sweep, multi-seed utilities
├── data/                    # Datasets, z_init, checkpoints
├── experiments/             # Training outputs (created at runtime)
└── requirements.txt
```
