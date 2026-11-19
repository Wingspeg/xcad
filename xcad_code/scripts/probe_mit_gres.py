#!/usr/bin/env python3
"""MIT Supercloud: did we mis-exclude it?

The 2026-11-16 screening (docs/reviews/2026-11-16_second_dataset_screening.md
S9.5) rejected MIT's GPU label on three columns: tres_alloc (empty on all 3,430
labelled rows), the README TRES table (only 1001 gpu:tesla / 1002 gpu:volta) and
partition (3,385/3,430 == 'normal').  But the slurm-log header also carries
`gres_used` and `constraints`, which were never looked at, and the bucket has a
separate 2022-hpca release with nvidia_smi.csv / dcgm.csv.

This script answers, on files already on disk:
  Q1 does gres_used / constraints carry a per-job GPU class, and how many rows?
  Q2 what is the label vocabulary, and does it survive a join to the 26 model names?
  Q3 per (model, gpu-class): how many jobs, how many distinct days -> cohort size.
  Q4 can a job be linked to node names, and are those names anonymised?
Writes nothing but stdout.
"""
import os
import sys

import pandas as pd

MIT = os.path.expanduser("~/xcad/datasets/mit")
COLS = ["id_job", "id_user", "nodes_alloc", "nodelist", "gres_used",
        "tres_alloc", "tres_req", "constraints", "partition",
        "time_submit", "time_start", "time_end", "job_type"]

log = pd.read_csv(f"{MIT}/slurm-log.csv", usecols=COLS, low_memory=False)
lab = pd.read_csv(f"{MIT}/labelled_jobids.csv")
print(f"slurm-log rows={len(log):,}  cols={list(log.columns)}")
print(f"labelled_jobids rows={len(lab):,}  distinct models={lab['model'].nunique():,}")

for c in ("gres_used", "constraints", "partition", "job_type", "tres_alloc", "tres_req"):
    ne = log[c].notna() & (log[c].astype(str).str.strip() != "")
    print(f"\n### {c}: non-empty {ne.sum():,} / {len(log):,} = {100*ne.mean():.2f}%")
    vc = log.loc[ne, c].astype(str).value_counts().head(12)
    for k, v in vc.items():
        print(f"    {v:>9,}  {k[:90]}")

# --- join to labelled model names -------------------------------------------------
j = log.merge(lab, on="id_job", how="inner")
print(f"\n=== join slurm-log x labelled_jobids: {len(j):,} rows, "
      f"{j['model'].nunique()} models, "
      f"{j['id_job'].nunique()} distinct jobs")

def cls(s):
    s = "" if pd.isna(s) else str(s).lower()
    for k in ("volta", "tesla", "ampere", "hopper", "a100", "v100", "p100", "k80"):
        if k in s:
            return k
    return None

for src in ("gres_used", "constraints", "tres_alloc", "tres_req"):
    j["_c"] = j[src].map(cls)
    ok = j["_c"].notna()
    print(f"\n--- label source `{src}`: {ok.sum():,} of {len(j):,} labelled rows "
          f"({100*ok.mean():.2f}%)")
    if ok.sum():
        print("    value counts:", j.loc[ok, "_c"].value_counts().to_dict())
        piv = (j.loc[ok].groupby(["model", "_c"]).size().rename("n")
               .reset_index())
        print(f"    (model,label) cells={len(piv):,}  models covered="
              f"{piv['model'].nunique()}")
        print(piv.sort_values("n", ascending=False).head(12).to_string(index=False))

# --- any combined source ----------------------------------------------------------
def best(r):
    for src in ("gres_used", "constraints", "tres_alloc", "tres_req"):
        c = cls(r[src])
        if c:
            return c
    return None

j["gpu_cls"] = j.apply(best, axis=1)
ok = j["gpu_cls"].notna()
print(f"\n=== any-of-4-columns GPU class present: {ok.sum():,} / {len(j):,} "
      f"= {100*ok.mean():.2f}%  over {j.loc[ok,'model'].nunique()} models")
if ok.sum():
    sub = j.loc[ok].copy()
    sub["day"] = (pd.to_numeric(sub["time_start"], errors="coerce") // 86400).astype("Int64")
    per_day = sub.groupby("day")["id_job"].size()
    print(f"    active days with >=1 labelled GPU job: {len(per_day)}  "
          f"jobs/day median={per_day.median()} max={per_day.max()}")
    # recurrence cohort: model seen on >=2 distinct days before the cut
    g = sub.dropna(subset=["day"]).groupby("model")["day"].agg(["nunique", "min", "max", "size"])
    print("    models with >=2 distinct days:", int((g["nunique"] >= 2).sum()),
          "of", len(g))
    print("    per-model day-span top 10:")
    print(g.sort_values("nunique", ascending=False).head(10).to_string())

# --- node names anonymised? -------------------------------------------------------
nn = log["nodelist"].dropna().astype(str)
print("\n=== nodelist samples:", nn.head(3).tolist())
import re
pat = nn.str.extract(r"^.\[?([a-z0-9]+)-")[0].value_counts().head(5)
print("    node-name prefixes:", pat.to_dict())
print("    distinct nodelists:", log["nodelist"].nunique())
print("\nDONE")
