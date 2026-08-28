# xCAD

**xCAD: Temporal Knowledge Graph Representation for Tri-Element Scheduling — Diagnosis, Benchmark, and Method**

This repository provides the official implementation of the paper *xCAD: Temporal Knowledge Graph Representation for Tri-Element Scheduling*. It includes a data pipeline that reconstructs a tri-element scheduling temporal graph from the Alibaba GPU cluster trace, two new mechanisms built on the RE-GCN backbone (heterogeneous relation-basis message passing and an algorithm-compatibility head), and all diagnosis and evaluation scripts.

---

## Overview

Existing methods and benchmarks for temporal knowledge graph (TKG) extrapolation reasoning almost exclusively target general event graphs (ICEWS, GDELT). Their inductive bias — relying on the recurrence of historical patterns — has rarely been examined on graphs of other structures. xCAD reconstructs, from a real production GPU-cluster trace, the **first tri-element (Compute, Algorithm, Data) scheduling temporal graph**, and reveals that on the core relation "Algorithm → GPU suitability", general TKG methods **collapse to a frequency prior** (always predicting the most frequent GPU; per-GPU macro only 0.20). This paper proposes two targeted mechanisms that lift per-GPU macro on this relation to 0.51, approaching the random-forest upper bound of 0.59.

**Key results (r1_suits, per-GPU macro Hits@1)**

| Method | per-GPU macro |
|---|---|
| RE-GCN / CEN / RETIA | 0.20 (collapses to frequency prior) |
| Random-forest upper bound (algorithm profile only) | 0.59 |
| **xCAD (ours)** | **0.51** |

---

## Dataset: Source and Reconstruction

xCAD is reconstructed from the public dataset **Alibaba Cluster-Trace-GPU-v2020** (source: <https://github.com/alibaba/clusterdata>, path `cluster-trace-gpu-v2020`). This repository **does not directly redistribute the reconstructed graph data**; instead, it provides the complete reconstruction scripts so the graph can be reproduced from the raw trace, while respecting the original dataset's license and distribution terms.

### Reconstruction Steps

1. Download the `cluster-trace-gpu-v2020` raw trace from the Alibaba clusterdata repository and place it under `data/raw/`.
2. Run the reconstruction script to map the trace into the tri-element temporal heterogeneous graph:

   ```bash
   python tools/export_to_regcn_format.py --raw-dir data/raw --out-dir data/xcad
   ```

### Graph Composition (After Reconstruction)

- **Nodes (124,891 entities)**: Compute (GPU generation + machine), Algorithm (job-level, each with a 13-dim resource profile), Data (data group).
- **Relations (4 types, 8 with inverses)**: placement, r1_suits (Algorithm → GPU suitability, the core relation), r2_requires, r3_drives.
- **Temporal split**: train / valid / test = 585,335 / 87,553 / 90,785 edges, corresponding to 49 / 6 / 6 time snapshots.

Reconstruction follows one discipline: the existence and weight of every relation edge come from real co-occurrence / runtime statistics; no hand-crafted rules are introduced. The only author-side prior is the schema design (the mapping from raw fields to node/relation types). See paper §3 and the script comments for full details.

---

## Environment

- Python 3.11
- PyTorch 2.x + CUDA (trained on a single RTX 3090, 24 GB)
- DGL

```bash
# uv or venv recommended
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproducing Experiments

All experiments use the default history length `h=6` and report three random seeds (42 / 123 / 2024).

### Baseline (RE-GCN backbone)

```bash
python main.py -d xcad --gpu 0 \
  --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2 \
  --n-bases 100 --train-history-len 6 --dropout 0.4 \
  --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4 \
  --weight-decay 1e-4 --lr 0.001 --entity-prediction \
  --n-epochs 30 --evaluate-every 5 --seed 42
```

### Point 1: Heterogeneous Relation-Basis Message Passing

Append to the baseline command:

```bash
  --hetero --n-msg-basis 2
```

### Point 2: Algorithm-Compatibility Head

Append to the baseline command:

```bash
  --compat --compat-lambda 5.0 --compat-aux-weight 0.5
```

### Full xCAD Method (Point 1 + Point 2)

```bash
python main.py -d xcad --gpu 0 \
  --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2 \
  --n-bases 100 --train-history-len 6 --dropout 0.4 \
  --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4 \
  --weight-decay 1e-4 --lr 0.001 --entity-prediction \
  --n-epochs 30 --evaluate-every 5 --seed 42 \
  --hetero --n-msg-basis 2 --compat --compat-lambda 5.0 --compat-aux-weight 0.5
```

Test-set evaluation reports overall filter MRR / Hits@k, per-relation MRR, and per-GPU Hits@1 / macro on the core relation r1_suits.

### Diagnosis and Upper-Bound Probing

```bash
# Frequency-prior diagnosis (load a checkpoint; prints prediction entropy / confusion / per-GPU Hits@1)
python diagnose_r1.py --checkpoint <path-to-checkpoint>

# Random-forest upper-bound probe (algorithm profile only; --split-by-algo splits strictly by algorithm to predict unseen algorithms)
python diagnose_sklearn.py
python diagnose_sklearn.py --split-by-algo
```

---

## Key Hyperparameters

| Parameter | Description | Default |
|---|---|---|
| `--train-history-len` | Number of history snapshots | 6 |
| `--hetero` | Enable heterogeneous relation-basis message passing (Point 1) | off |
| `--n-msg-basis` | Number of basis matrices $B$ | 2 |
| `--compat` | Enable algorithm-compatibility head (Point 2) | off |
| `--compat-lambda` | Compatibility bias fusion weight $\lambda$ | 5.0 |
| `--compat-aux-weight` | Optional GPU auxiliary classification loss weight $\beta$ | 0.5 |
| `--seed` | Random seed | 42 |

When `--hetero` and `--compat` are both off, the model is bit-for-bit identical to vanilla RE-GCN.

---

## Citation

If this work is helpful, please cite (placeholder; will be updated after formal publication):

```bibtex
@article{xcad,
  title   = {xCAD: Temporal Knowledge Graph Representation for Tri-Element Scheduling},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

---

## Acknowledgements

The backbone and evaluation protocol are based on RE-GCN (Li et al., SIGIR 2021). The dataset is reconstructed from Alibaba Cluster-Trace-GPU-v2020.

## License

The code is released under the MIT License. The original trace data that reconstruction depends on follows Alibaba clusterdata's original license terms.
