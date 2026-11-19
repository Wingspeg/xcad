#!/usr/bin/env python3
"""MIT Supercloud, corrected TRES parsing: is there a per-job GPU class label?

S9.5 recorded "tres_alloc empty on all 3,430 labelled rows".  The full slurm-log
shows tres_alloc 100% non-empty and containing tokens like `1002=1`, which per
the dataset README maps to `gpu:volta` (1001 -> gpu:tesla).  The first probe
matched on the *words* volta/tesla and therefore read 0 everywhere - wrong test.
Here we parse the TRES string into {id: count} and count properly.

Also fetch, byte-range only, the headers of the separate 2022-hpca release files
(nvidia_smi.csv / dcgm.csv / scheduler_data.csv) to see whether a real GPU *model
name* is published anywhere in this dataset.
"""
import os
import re
import subprocess

import pandas as pd

MIT = os.path.expanduser("~/xcad/datasets/mit")
TRES_NAME = {1001: "gpu:tesla", 1002: "gpu:volta"}


def tres_gpu(s):
    """Return (label, n_gpus) for a TRES string, else (None, 0)."""
    if not isinstance(s, str):
        return None, 0
    best_lbl, best_n = None, 0
    for tok in s.split(","):
        if "=" not in tok:
            continue
        k, _, v = tok.partition("=")
        try:
            k = int(k)
        except ValueError:
            continue
        if k in TRES_NAME:
            try:
                n = int(v)
            except ValueError:
                n = 0
            if n > best_n:
                best_n, best_lbl = n, TRES_NAME[k]
    return best_lbl, best_n


log = pd.read_csv(f"{MIT}/slurm-log.csv",
                  usecols=["id_job", "id_user", "nodelist", "nodes_alloc",
                           "tres_alloc", "tres_req", "partition", "constraints",
                           "time_start", "time_submit"],
                  low_memory=False)
lab = pd.read_csv(f"{MIT}/labelled_jobids.csv")

for col in ("tres_alloc", "tres_req"):
    res = log[col].map(tres_gpu)
    log["_lbl"] = [r[0] for r in res]
    log["_n"] = [r[1] for r in res]
    print(f"### full log, from `{col}`: GPU-class present {log['_lbl'].notna().sum():,} "
          f"/ {len(log):,} = {100*log['_lbl'].notna().mean():.2f}%")
    print("    label counts:", log["_lbl"].value_counts(dropna=True).to_dict())
    print("    gpu-count distribution:", log.loc[log['_lbl'].notna(), '_n']
          .value_counts().head(8).to_dict())

# how far can we get on the *unlabelled* mass: node identity per GPU job
g = log[log["_lbl"].notna()]
print(f"\n=== GPU jobs in full log: {len(g):,}; distinct nodelists "
      f"{g['nodelist'].nunique():,}; nodes_alloc mean {g['nodes_alloc'].mean():.2f}")
print("    partition of GPU jobs:", g["partition"].value_counts().head(5).to_dict())
print("    constraints of GPU jobs:", g["constraints"].value_counts().head(5).to_dict())

# --- join with the 26 named models ------------------------------------------------
al = log.drop_duplicates("id_job")
j = al.merge(lab, on="id_job", how="left")
# direct test of S9.5's claim: were the labelled rows' tres_alloc really empty?
lab_ids = set(lab["id_job"])
sub = al[al["id_job"].isin(lab_ids)]
print(f"\n=== S9.5 claim check: labelled rows in slurm-log = {len(sub):,}")
print("    raw tres_alloc top 10:",
      sub["tres_alloc"].astype(str).value_counts().head(10).to_dict())
print("    n_empty/NaN tres_alloc =", int(sub["tres_alloc"].isna().sum()),
      "| n containing '1001' or '1002' =",
      int(sub["tres_alloc"].astype(str).str.contains(r"100[12]=", regex=True).sum()))
has = j["_lbl"].notna()
print(f"\n=== labelled(model-named) jobs: {len(j):,}; with GPU class {has.sum():,} "
      f"= {100*has.mean():.2f}%")
if has.sum():
    s = j[has].copy()
    s["day"] = pd.to_numeric(s["time_start"], errors="coerce") // 86400
    print("    labels:", s["_lbl"].value_counts().to_dict())
    print(f"    models covered: {s['model'].nunique()} / {lab['model'].nunique()}")
    per_model = s.groupby("model").agg(jobs=("_lbl", "size"),
                                       days=("day", "nunique"),
                                       classes=("_lbl", "nunique"))
    print(per_model.sort_values("jobs", ascending=False).head(15).to_string())
    # the thing S9.5 died on: recurrence cohort for a held-out split
    cut = s["day"].quantile(0.8)
    hist = s[s["day"] <= cut]
    test = s[s["day"] > cut]
    seen = hist.groupby("model").agg(days=("day", "nunique"))
    recur = seen[seen["days"] >= 2].index
    print(f"\n    80/20 time split at day {cut}: test jobs={len(test):,}, "
          f"of which model seen >=2 days in history = "
          f"{test['model'].isin(recur).sum():,}")

# --- the 2022-hpca release: does it publish real model names? ---------------------
print("\n=== 2022-hpca files (first 600 bytes each, range request)")
for f in ("nvidia_smi.csv", "dcgm.csv", "scheduler_data.csv", "README.md"):
    url = f"https://mit-supercloud-dataset.s3.amazonaws.com/2022-hpca/{f}"
    p = subprocess.run(["curl", "-sS", "-L", "--max-time", "60",
                        "-r", "0-600", url], capture_output=True, text=True)
    head = (p.stdout or "").replace("\n", " | ")[:520]
    q = subprocess.run(["curl", "-sSI", "--max-time", "45", url],
                       capture_output=True, text=True)
    size = re.search(r"(?i)content-length:\s*(\d+)", q.stdout or "")
    print(f"--- {f}  bytes={size.group(1) if size else '?'}")
    print("    ", head)
print("\nDONE")
