#!/usr/bin/env python3
"""Exercise the 'dataset-agnostic code' claim: one reader, three bundles.

Nothing here is dataset-specific except the list of dataset names, which is
discovered from the directory.  If the shared column contract (H7) holds, every
loop below runs on all three bundles with the same expressions.
"""
import os
import sys

import pandas as pd

ROOT = os.path.expanduser("~/xcad/datasets/clean")

for ds in sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))):
    b = os.path.join(ROOT, ds)
    tables = {t[:-len(".parquet")]: pd.read_parquet(os.path.join(b, t))
              for t in sorted(os.listdir(b)) if t.endswith(".parquet")}
    names = ("jobs", "requests", "placements", "placement_devices",
             "nodes", "devices", "gpu_types", "subclusters")
    missing = [t for t in names if t not in tables]
    jobs = tables["jobs"]
    gpu = jobs[jobs["is_gpu_job"]]
    line = (f"{ds:<10} tables={len(tables)} missing={missing} "
            f"jobs={len(jobs):,} gpu={len(gpu):,} "
            f"snaps={jobs['snap_submit'].nunique()} "
            f"median_run={jobs['run_s'].median()}")
    # same expression on all three: GPU jobs per snapshot, top 3
    top = (gpu.groupby("snap_submit", observed=True).size()
             .sort_values(ascending=False).head(3).to_dict())
    print(line, "gpu_per_snap_top3=", top)
    # same join across all three: job -> placement (empty tables must still join)
    pl = tables["placements"]
    need = {"job_gid", "machine_gid", "gpu_label"}
    if not need.issubset(pl.columns):
        print(f"           jobs x placements -> skipped, empty table carries "
              f"columns={list(pl.columns)}")
    else:
        if "is_gpu_placement" in pl.columns:
            pl = pl[pl["is_gpu_placement"].fillna(False).astype(bool)]
        elif "n_gpus" in pl.columns:
            pl = pl[pl["n_gpus"].fillna(0) > 0]
        pairs = pl[["job_gid", "gpu_label"]].dropna().drop_duplicates()
        final = (jobs.drop_duplicates("job_gid")
                 .set_index("job_gid")["gpu_label_final"])
        lab = pairs["job_gid"].map(final)
        uncovered = sum(l not in str(f).split("|")
                        for l, f in zip(pairs["gpu_label"], lab))
        print(f"           (job, gpu_label) placement pairs: {len(pairs):,}; "
              f"not covered by that job's gpu_label_final set: {uncovered:,}")
    print(f"           devices: {len(tables['devices']):,} rows, "
          f"gpu_types: {len(tables['gpu_types']):,} labels "
          f"published={tables['gpu_types']['model_name_published'].sum() if len(tables['gpu_types']) else '-'}")
print("OK")
sys.exit(0)
