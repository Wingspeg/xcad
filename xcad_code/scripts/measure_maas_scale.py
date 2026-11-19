#!/usr/bin/env python3
"""How big is the maas graph, and is it snapshot-continuous?  Pre-export survey.

Answers the four things that decide the exporter's shape:
  1. edge volume after dedup on (model_id, gpu_type, day) -- the graph's edges
  2. day coverage: which days exist, are they contiguous, edges per day
  3. head degree distribution + multi-tail-per-(head,day) conflicts
  4. column null rates for the candidate second relation (params_b/arch -> mem_alloc)
CPU only, one pass over the already-sliced shards.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

COLS = ["day", "instance_id", "app_id", "model_id", "base_id", "cluster_id", "site_id",
        "gpu_type", "workload", "model_arch", "params_b", "pd_role", "gpu_count",
        "gpu_mem_alloc"]


def main(paths):
    import pyarrow.parquet as pq
    sch = pq.ParquetFile(paths[0]).schema_arrow
    print("-- actual schema of part_0 --")
    for f in sch:
        print(f"  {f.name:16s} {f.type}")
    t = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    print(f"raw rows={len(t):,}  from {len(paths)} shards")

    print("\n-- null rates (candidate columns) --")
    for c in COLS:
        s = t[c]
        null = s.isna().mean()
        extra = ""
        if s.dtype == object:
            empty = s.astype("string").str.strip().eq("").mean()
            extra = f"  empty_str={empty:.4f}"
        print(f"  {c:14s} null={null:.4f}  distinct={s.nunique()}  "
              f"mean={s.mean() if pd.api.types.is_numeric_dtype(s) else '-'}{extra}")

    print("\n-- gpu_type distribution --")
    vc = t.gpu_type.value_counts()
    print(f"  distinct={len(vc)}  top5 share={vc.head(5).sum() / len(t):.4f}  "
          f"most frequent={vc.iloc[0] / len(t):.4f}")
    print("  " + ", ".join(f"{k}:{v / len(t):.3f}" for k, v in vc.head(8).items()))

    print("\n-- day coverage --")
    days = np.sort(t.day.unique())
    diffs = np.diff(days)
    print(f"  distinct days={len(days)}  range=[{days.min()}..{days.max()}]  "
          f"gaps>1: {(diffs > 1).sum()}  largest gap={diffs.max() if len(diffs) else '-'}")
    per = t.groupby("day").size()
    print(f"  raw rows/day  min={per.min():,} med={int(per.median()):,} max={per.max():,}")

    e = t[["model_id", "gpu_type", "day"]].drop_duplicates()
    print(f"\n-- edges (model_id, gpu_type, day) dedup --")
    print(f"  edges={len(e):,}  heads={e.model_id.nunique():,}  tails={e.gpu_type.nunique()}")
    per_e = e.groupby("day").size()
    print(f"  edges/day  min={per_e.min():,} med={int(per_e.median()):,} max={per_e.max():,}")

    hd = e.groupby("model_id").size()
    print(f"  head degree: 1={float((hd == 1).mean()):.4f}  2={float((hd == 2).mean()):.4f}  "
          f"3+={float((hd >= 3).mean()):.4f}  max={int(hd.max())}")
    hs = e.groupby("model_id").day.nunique()
    print(f"  head distinct days: 1={float((hs == 1).mean()):.4f}  "
          f">=2={float((hs >= 2).mean()):.4f}  >=3={float((hs >= 3).mean()):.4f}  max={int(hs.max())}")

    mt = e.groupby(["model_id", "day"]).gpu_type.nunique()
    print(f"  (head,day) with >1 gpu_type: {int((mt > 1).sum()):,} / {len(mt):,} "
          f"= {float((mt > 1).mean()):.4f}   max types in one day={int(mt.max())}")

    # instance-level: does one model run on several card types at the same day?
    ii = t.groupby(["model_id", "day"]).gpu_type.nunique()
    print(f"  raw-level (model,day) >1 type: {float((ii > 1).mean()):.4f}")

    # tail stability of a head over time (what entity memory can win on)
    w = e.sort_values("day").groupby("model_id")
    first = w.gpu_type.first()
    last = w.gpu_type.last()
    both = pd.concat([first, last], axis=1).dropna()
    same = float((both.iloc[:, 0] == both.iloc[:, 1]).mean())
    multi = both[both.iloc[:, 0] != both.iloc[:, 1]]
    print(f"\n-- head type drift --")
    print(f"  heads whose first-day type == last-day type: {same:.4f} "
          f"(n={len(both):,}); drifting heads={len(multi):,}")
    span = (w.day.max() - w.day.min())
    print(f"  head span in days: med={int(span.median())}  "
          f">=2 days={float((span >= 2).mean()):.4f}  >=7={float((span >= 7).mean()):.4f}")

    print("\n-- second-relation candidates --")
    if "params_b" in t and "gpu_mem_alloc" in t:
        g = t.dropna(subset=["params_b", "gpu_mem_alloc"])
        print(f"  rows with both params_b & gpu_mem_alloc: {len(g):,} ({len(g) / len(t):.4f})")
        if len(g):
            print(f"  gpu_mem_alloc distinct={g.gpu_mem_alloc.nunique()}  "
                  f"params_b distinct={g.params_b.nunique()}")
            for k in ("model_arch", "workload", "pd_role"):
                if k in t:
                    s = t[k].astype("string").str.strip()
                    print(f"  {k}: distinct={s.nunique()}  "
                          f"top={s.value_counts().head(4).to_dict()}")
    print("\n-- base_id / app_id as alternative head granularity --")
    for c in ("base_id", "app_id"):
        n = t[c].nunique()
        d = t.groupby(c).day.nunique().median()
        print(f"  {c}: distinct={n:,}  median distinct days per value={d}")


if __name__ == "__main__":
    ps = sys.argv[1:] or sorted(p for p in glob.glob(os.path.expanduser(
        "~/xcad/datasets/maas/instance_daily/part_*.parquet"))
        if "slim" not in os.path.basename(p))        # part_0_slim duplicates part_0
    main(ps)
