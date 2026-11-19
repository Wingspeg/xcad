# xCAD

**xCAD: A Unified Representation Method of Heterogeneous Computing Power, Algorithm and Data Resources for Computing Power Networks**

This repository provides the official implementation of the paper *xCAD: A Unified Representation Method of Heterogeneous Computing Power, Algorithm and Data Resources for Computing Power Networks*. It includes the reconstructed tri-element scheduling temporal graph, a data pipeline that rebuilds it from the Alibaba GPU cluster trace, two new mechanisms built on the RE-GCN backbone (heterogeneous relation-basis message passing and an algorithm-compatibility head), and the diagnosis, evaluation and analysis scripts that produced every reported reading.

---

## Overview

Existing methods and benchmarks for temporal knowledge graph (TKG) extrapolation reasoning almost exclusively target general event graphs (ICEWS, GDELT). Their inductive bias — relying on the recurrence of historical patterns — has rarely been examined on graphs of other structures. xCAD reconstructs, from a real production GPU-cluster trace, the **first tri-element (Compute, Algorithm, Data) scheduling temporal graph**, and reveals that on the core relation "Algorithm → GPU suitability", general TKG methods **collapse to a frequency prior** (always predicting the most frequent GPU; per-GPU macro only 0.20). This paper proposes two targeted mechanisms that lift per-GPU macro on this relation to 0.51, approaching the 0.582 static-profile learnability reference measured on the same test edges and the same fitting objective.

**Key results (r1_suits, per-GPU macro Hits@1)**

| Method | per-GPU macro |
|---|---|
| RE-GCN / CEN / RETIA | 0.20 (collapses to frequency prior) |
| Static-profile learnability reference (algorithm profile only, random forest) | 0.582 |
| **xCAD (ours)** | **0.51** |

The reference row is a *measurement of how much of the relation is learnable from the algorithm profile alone*, not a model we claim to outperform: our arm sits below it, and the residual gap is analysed in the paper.

---

## Dataset: Source and Reconstruction

xCAD is reconstructed from the public dataset **Alibaba Cluster-Trace-GPU-v2020** (source: <https://github.com/alibaba/clusterdata>, path `cluster-trace-gpu-v2020`). The original trace follows Alibaba clusterdata's license terms and is **not** redistributed here.

What this repository does ship:

- **the reconstructed graph**, in RE-GCN format, at `xcad_code/xcad_model/data/xcad_v2` — the dataset every reported reading was produced on;
- **the pipeline intermediates** that rebuild it, `xcad_code/outputs/nodes/*.parquet` and `xcad_code/outputs/edges/*.parquet`;
- **the reconstruction scripts** under `xcad_code/src/data/`.

### Two id spaces, and why the flag is load-bearing

`xcad_code/xcad_model/data/xcad` is an earlier export in which each entity type kept its own local dictionary, so one integer could name an algorithm, a data group, a machine *and* a GPU type simultaneously. `xcad_v2` re-exports the same graph in a single global four-segment space:

```
algo [0, 102609] · data [102610, 122987] · compute [122988, 124884] · gpu [124885, 124890]
```

Both trees load and run; `main.py` detects the space and prints `[id-space] … LEGACY` or `… GLOBAL`. They are **not** interchangeable: the same compatibility-head arm gives per-GPU macro 0.4964 on `xcad` and 0.51 on `xcad_v2`, and only the latter is reported in the paper. Every command below uses `-d xcad_v2`.

### Rebuilding the graph from the shipped intermediates

```bash
cd xcad_code
python src/data/export_to_regcn_format.py \
  --out-dir xcad_model/data/xcad_v2 --expected-entities 124891
```

`--edges-dir` and `--nodes-dir` default to `outputs/edges` and `outputs/nodes`, which are shipped. Re-running this reproduces `train.txt`, `valid.txt`, `test.txt`, `entity2id.txt`, `relation2id.txt` and `stat.txt` **byte for byte** — all six checksums were compared against the committed files on 2026-11-19.

### Rebuilding from the raw trace

Place the seven CSVs (`pai_job_table`, `pai_task_table`, `pai_instance_table`, `pai_sensor_table`, `pai_group_tag_table`, `pai_machine_spec`, `pai_machine_metric`, each `.csv`) under `datasets/alibaba-cluster-trace-gpu-v2020/raw/`; that directory is the `DATA_ROOT` constant in `xcad_code/src/utils/config.py` and the file names are `TABLE_FILES` in `xcad_code/src/utils/schema.py`. Then, from `xcad_code/`:

```bash
python -m src.data.build_nodes        # -> outputs/nodes/{compute,algorithm,data}_nodes.parquet
python -m src.data.build_edges        # -> outputs/edges/{placement,r1_suits,r2_requires,r3_drives,r4_shifts}_edges.parquet
python -m src.data.temporal_split     # -> outputs/dgl/splits/*.parquet (diagnostics; the exporter does not read these)
python -m src.data.build_dgl_graph    # -> outputs/dgl/*.pt (DGL view used by the graph-side analyses)
```

These four stages take no command-line flags; their input and output paths come from `src/utils/config.py`, so the directory layout above is what you have to match. They produced the intermediates that are shipped, but this release verified the exporter round-trip rather than a from-scratch re-run of all four — if a stage fails on your machine, start from the shipped `outputs/{nodes,edges}`.

Two further scripts produce the arrays that sit next to the graph files: `src/data/export_edge_weights.py` writes `edge_weights_{train,valid,test}.npy` and takes `--data-dir` plus `--gpu-id-base`, which must match the id space of the split files it is given (`0` for `xcad`, `124885` for `xcad_v2`); `src/data/build_node_features.py` writes the per-type profile matrices. The shipped `node_features.npy` is a `[124891, 3]` block that is non-zero only on the 20,378 data-group rows; no training or evaluation path reads it.

### Graph composition (After Reconstruction)

- **Nodes (124,891 entities)**: Compute (GPU generation + machine), Algorithm (job-level, each with a 13-dim resource profile), Data (data group).
- **Relations (4 types, 8 with inverses)**: placement, r1_suits (Algorithm → GPU suitability, the core relation), r2_requires, r3_drives.
- **Temporal split**: train / valid / test = 585,335 / 87,553 / 90,785 edges, corresponding to 49 / 6 / 6 time snapshots (τ ranges [1,49] / [50,55] / [56,61]).

Reconstruction follows one discipline: the existence and weight of every relation edge come from real co-occurrence / runtime statistics; no hand-crafted rules are introduced. The only author-side prior is the schema design (the mapping from raw fields to node/relation types). See paper §3 and the script comments for full details.

---

## Environment

- Python 3.11
- PyTorch 2.x + CUDA (trained on a single RTX 3090, 24 GB)
- DGL

```bash
# uv or venv recommended; the requirements file lives one level down
python -m venv .venv && source .venv/bin/activate
pip install -r xcad_code/requirements.txt
```

---

## Reproducing Experiments

**Working directory is part of the command.** `main.py` inserts `..` into `sys.path` and resolves the dataset as `../data/<dataset>`, so the model commands must run from `xcad_code/xcad_model/src/`:

```bash
cd xcad_code/xcad_model/src
```

The published arms pass `--train-history-len 6` explicitly — the argparse default in `main.py` is 10 — and report three random seeds (42 / 123 / 2024).

The arm queues, reproducibility gates and diagnostic probes that produced the reported readings are indexed in [`xcad_code/scripts/README.md`](xcad_code/scripts/README.md); they assume the working-tree layout described there.

### Baseline (RE-GCN backbone)

```bash
python main.py -d xcad_v2 --gpu 0 \
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
python main.py -d xcad_v2 --gpu 0 \
  --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2 \
  --n-bases 100 --train-history-len 6 --dropout 0.4 \
  --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4 \
  --weight-decay 1e-4 --lr 0.001 --entity-prediction \
  --n-epochs 30 --evaluate-every 5 --seed 42 \
  --hetero --n-msg-basis 2 --compat --compat-lambda 5.0 --compat-aux-weight 0.5
```

Test-set evaluation reports overall filter MRR / Hits@k, per-relation MRR, and per-GPU Hits@1 / macro on the core relation r1_suits. On the flagship arm the three-seed means are:

| reading (r1_suits, time-aware filtered over all 124,891 entities) | three seeds |
|---|---|
| MRR | 0.9251 ± 0.0019 |
| Hits@1 | 0.8830 ± 0.0016 |
| per-GPU macro Hits@1 | 0.5074 ± 0.0009 |

Per-seed values, every ablation row and the stratum splits behind these numbers are in [`xcad_code/outputs/reports/night_2026_11_18/night_tables.md`](xcad_code/outputs/reports/night_2026_11_18/night_tables.md). Single-run noise on this dataset measures 0.0172 on whole-graph Hits@1, so differences below 0.02 are not treated as conclusions anywhere in the paper.

### External baselines (RE-GCN / CEN / RETIA)

```bash
bash xcad_code/scripts/link_xcad_v2_to_baselines.sh   # from the repository root
```

That stages the seven files the baselines read into `baselines/{CEN,RE-GCN,RETIA}/data/xcad_v2/` and checks them against the source copies. [`xcad_code/scripts/rerun_baselines_idspace.sh`](xcad_code/scripts/rerun_baselines_idspace.sh) is the driver that produced the three baseline rows on `xcad_v2` (same hyperparameters as the June runs, `-d xcad` swapped for `-d xcad_v2`); its logs are in [`xcad_code/outputs/reports/baselines_idspace/`](xcad_code/outputs/reports/baselines_idspace/). All three collapse to the frequency prior there — per-GPU macro 20.29 / 20.00 / 20.00 for RE-GCN / CEN / RETIA. Arms must be run serially: a single arm holds more than 11.6 GB.

### Diagnosis and Learnability Reference

```bash
# still in xcad_code/xcad_model/src

# Frequency-prior diagnosis (load a checkpoint; prints prediction entropy / confusion / per-GPU Hits@1).
# Checkpoints store weights and an epoch only, so the model-side flags have to match the arm that
# produced them -- shown here for the flagship. Any weight this script cannot load is named on
# [model][warn] rather than left silently at random initialisation.
python diagnose_r1.py --checkpoint <path-to-checkpoint> \
  --data xcad_v2 --hetero --n-msg-basis 2 --compat --compat-lambda 5.0

# Static-profile learnability reference (algorithm profile only; LR + RF + MLP on the same
# 13-dim profile and the same test edges). --split-by-algo holds out whole algorithms, so it
# measures generalisation to unseen algorithms rather than to unseen timestamps.
python diagnose_sklearn.py
python diagnose_sklearn.py --split-by-algo
```

Everything above reads the released `xcad_v2` id space by default: `main.py --feat-dir` and both
diagnose scripts derive their feature path from the dataset flag (`-d xcad_v2` / `--data xcad_v2` /
`--data-dir ../data/xcad_v2`), so no command needs an extra path argument. Pass the flag explicitly
to diagnose an arm trained on the legacy `data/xcad` dictionaries — the checkpoint and the graph must
come from the same numbering, and the arm's own `[id-space]` line says which one it used. The two
directories hold byte-identical feature matrices, so switching only renumbers entities.

---

## Key Hyperparameters

| Parameter | Description | argparse default | published arms |
|---|---|---|---|
| `--train-history-len` | Number of history snapshots | 10 | 6 |
| `--hetero` | Enable heterogeneous relation-basis message passing (Point 1) | off | on |
| `--n-msg-basis` | Number of basis matrices $B$ | 2 | 2 |
| `--compat` | Enable algorithm-compatibility head (Point 2) | off | on |
| `--compat-lambda` | Compatibility bias fusion weight $\lambda$ | `None` — $\lambda$ becomes a learnable scalar initialised at 0.1 | 5.0, fixed (`rrgcn.py:371-374` registers a buffer, not a parameter) |
| `--compat-aux-weight` | GPU auxiliary classification loss weight $\beta$ | 0.5 | 0.5 |
| `--seed` | Random seed | 42 | 42 / 123 / 2024 |

When `--hetero` and `--compat` are both off, the model is bit-for-bit identical to vanilla RE-GCN.

---

## Citation

If this work is helpful, please cite (placeholder; will be updated after formal publication):

```bibtex
@article{xcad,
  title   = {xCAD: A Unified Representation Method of Heterogeneous Computing Power, Algorithm and Data Resources for Computing Power Networks},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

---

## Acknowledgements

The backbone and evaluation protocol are based on RE-GCN (Li et al., SIGIR 2021). The dataset is reconstructed from Alibaba Cluster-Trace-GPU-v2020.

## License

[MIT](./LICENSE) © 2026 Wingspeg.

The original trace data that reconstruction depends on follows Alibaba clusterdata's original license terms.
