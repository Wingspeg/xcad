#!/usr/bin/env python3
"""MIT Supercloud 2022-hpca release: does it give xCAD's task on real algorithm names?

Decisive questions, in order of cost:
  Q1 license text of the 2022-hpca release (2 KB, full read)
  Q2 scheduler_data.csv (73 MB): is gres_alloc/gres_req populated, and does the
     `gpu` partition give a card class finer than the TRES id?
  Q3 nvidia_smi.csv (44 GB): sample 10 x 6 MB ranges spread across the file and
     read off the *device inventory* -> distinct memory_free_MiB per (Node,
     gpu_index) = the capacity-tier vocabulary, plus how many nodes/cards exist.
  Q4 do nvidia_smi id_jobs join to the 3,430 model-labelled jobs?

Nothing here trains a model; it decides whether MIT is a usable second dataset.
"""
import io
import os
import re
import subprocess
import sys

import pandas as pd

BUCKET = "https://mit-supercloud-dataset.s3.amazonaws.com"
WORK = os.path.expanduser("~/xcad/datasets/mit/hpca")
os.makedirs(WORK, exist_ok=True)


def curl(args):
    p = subprocess.run(["curl", "-sS", "-L", "--max-time", "600"] + args,
                       capture_output=True, text=True)
    return (p.stdout or "").strip(), (p.stderr or "").strip()[:200]


print("=== Q1 README (full, 2,008 B)")
out, err = curl([f"{BUCKET}/2022-hpca/README.md"])
print(out)
if err:
    print("err:", err)

print("\n=== Q2 scheduler_data.csv -> download whole (73 MB)")
dst = f"{WORK}/scheduler_data.csv"
meta, err = curl(["-o", dst, "-w", "%{http_code} %{size_download}",
                  f"{BUCKET}/2022-hpca/scheduler_data.csv"])
print("fetch:", meta, err)
sd = pd.read_csv(dst, low_memory=False)
print("rows", f"{len(sd):,}", "cols", len(sd.columns))
for c in ("gres_req", "gres_alloc", "gres_used", "partition", "constraints"):
    if c not in sd.columns:
        continue
    ne = sd[c].notna() & (sd[c].astype(str).str.strip() != "")
    print(f"\n--- {c}: non-empty {ne.sum():,} / {len(sd):,} = {100*ne.mean():.2f}%")
    print("    top:", sd.loc[ne, c].astype(str).value_counts().head(8).to_dict())
print("\n--- nodelist on gpu-partition jobs:",
      sd.loc[sd["partition"] == "gpu", "nodelist"].dropna().astype(str).head(3).tolist())
print("--- distinct nodes on gpu jobs:",
      sd.loc[sd["partition"] == "gpu", "nodelist"].nunique())
sd.to_pickle(dst + ".pkl")

# ---- Q3 device inventory from a stratified byte-range sample of nvidia_smi.csv ---
print("\n=== Q3 nvidia_smi.csv stratified sample (10 x 6 MB)")
size = int(curl(["-sSI", f"{BUCKET}/2022-hpca/nvidia_smi.csv"])[0]
           .split("content-length:")[-1].split()[0])
print("total size:", f"{size:,}")
rows = []
for i in range(10):
    lo = int(size * i / 10)
    hi = lo + 6_000_000
    out, err = curl(["-r", f"{lo}-{hi}", f"{BUCKET}/2022-hpca/nvidia_smi.csv"])
    if not out:
        print(f"  chunk {i} empty ({err})")
        continue
    lines = out.split("\n")
    body = "\n".join(lines[1:] if i else lines)      # drop partial header/first line
    try:
        d = pd.read_csv(io.StringIO("Node,gpu_index,memory_free_MiB,memory_used_MiB,"
                                    "power_draw_W,temperature_gpu,id_job\n" + body,
                                    ), nrows=200_000)
    except Exception as e:
        print(f"  chunk {i} parse fail: {e}")
        continue
    rows.append(d)
    print(f"  chunk {i}: rows={len(d):,} distinct Node={d['Node'].nunique()}")
inv = pd.concat(rows, ignore_index=True)
inv = inv[inv["Node"].astype(str).str.match(r"^[a-z0-9]+-n[0-9]+$", na=False)]
print("\n--- sampled device inventory")
per_dev = (inv.groupby(["Node", "gpu_index"], as_index=False)
              .agg(mib_min=("memory_free_MiB", "min"),
                   mib_max=("memory_free_MiB", "max"),
                   pow_max=("power_draw_W", "max"),
                   n=("memory_free_MiB", "size")))
print(f"    devices sampled={len(per_dev):,}  distinct nodes={inv['Node'].nunique():,}")
print("    distinct memory_free_MiB (idle capacity tiers):",
      sorted(inv["memory_free_MiB"].dropna().round(0).unique())[:20])
tier = per_dev["mib_max"].round(0)
print("    per-device max free MiB value counts:", tier.value_counts().head(10).to_dict())

# ---- Q4 join the sensor id_jobs to the model labels ------------------------------
print("\n=== Q4 join")
lab = pd.read_csv(os.path.expanduser("~/xcad/datasets/mit/labelled_jobids.csv"))
sensor_jobs = set(inv["id_job"].dropna().astype("int64"))
print(f"    sampled sensor id_jobs={len(sensor_jobs):,}  labelled jobs={len(lab):,}  "
      f"overlap={len(sensor_jobs & set(lab['id_job'])):,}")
if "id_job" in sd.columns:
    m = sd.merge(lab, on="id_job", how="inner")
    print(f"    scheduler_data x labelled: {len(m):,} rows over "
          f"{m['model'].nunique() if len(m) else 0} models")
    if len(m):
        print("    partitions of labelled jobs:", m["partition"].value_counts().head(5).to_dict())
        print("    gres_alloc of labelled jobs:",
              m["gres_alloc"].astype(str).value_counts().head(6).to_dict())
print("\nDONE")
