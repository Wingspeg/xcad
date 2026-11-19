#!/usr/bin/env python3
"""MIT 2022-hpca nvidia_smi.csv: enumerate the per-device memory tiers.

nvidia_smi.csv is 44,162,866,622 B with columns
    Node, gpu_index, memory_free_MiB, memory_used_MiB, pcie_link_width_current,
    power_draw_W, temperature_gpu, temperature_memory, timestamp,
    utilization_gpu_pct, utilization_memory_pct, id_job
so a device's capacity is read off per row as memory_free_MiB + memory_used_MiB
(the first version of this probe used max(memory_free) and was wrong: the file is
sorted by job, so devices are almost never observed idle, and the low values that
came out were in-use readings rather than smaller cards).  We never download the
whole file: take N fixed-size byte ranges spread over it, then report
  (a) the distinct capacity values  -> how many tiers the fleet really has,
  (b) devices per node              -> is it 4x/8x,
  (c) whether sampled id_jobs hit the 3,430 model-labelled jobs.
If (a) has >=2 tiers, MIT's tail can be written as a capacity tier rather than
a bare node id, which changes S12.3's "no card-type variance" conclusion.
"""
import io
import os
import re
import subprocess

import pandas as pd

BUCKET = "https://mit-supercloud-dataset.s3.amazonaws.com/2022-hpca/nvidia_smi.csv"
HDR = ("Node,gpu_index,memory_free_MiB,memory_used_MiB,pcie_link_width_current,"
       "power_draw_W,temperature_gpu,temperature_memory,timestamp,"
       "utilization_gpu_pct,utilization_memory_pct,id_job")
N_CHUNK, CHUNK = 40, 1_500_000

p = subprocess.run(["curl", "-sS", "-I", "--max-time", "60", BUCKET],
                   capture_output=True, text=True)
m = re.search(r"(?im)^content-length:\s*(\d+)", p.stdout or "")
size = int(m.group(1)) if m else 44_162_866_622
print(f"nvidia_smi.csv size = {size:,} B  (HEAD parsed: {bool(m)})")

chunks, failed = [], 0
for i in range(N_CHUNK):
    lo = int(size * i / N_CHUNK)
    hi = lo + CHUNK - 1
    q = subprocess.run(["curl", "-sS", "-L", "--max-time", "300",
                        "-r", f"{lo}-{hi}", BUCKET], capture_output=True, text=True)
    body = q.stdout or ""
    if not body:
        failed += 1
        print(f"  chunk {i:2d} @ {lo:>15,}  EMPTY  {q.stderr[:80]}")
        continue
    lines = body.split("\n")
    if i == 0 and lines[0].startswith("Node,"):
        lines = lines[1:]
    else:
        lines = lines[1:]                    # drop the partial first line
    if lines:
        lines = lines[:-1]                  # drop the partial last line
    try:
        d = pd.read_csv(io.StringIO(HDR + "\n" + "\n".join(lines)))
    except Exception as e:
        failed += 1
        print(f"  chunk {i:2d} parse fail {e}")
        continue
    d = d[d["Node"].astype(str).str.match(r"^[a-z0-9]+-n[0-9]+$", na=False)]
    chunks.append(d)
    print(f"  chunk {i:2d} @ {lo:>15,}  rows={len(d):>8,}  nodes={d['Node'].nunique():>4}")

inv = pd.concat(chunks, ignore_index=True)
print(f"\n=== sampled rows={len(inv):,}  distinct Node={inv['Node'].nunique()}  "
      f"failed chunks={failed}")

inv["cap_MiB"] = inv["memory_free_MiB"] + inv["memory_used_MiB"]
dev = inv.groupby(["Node", "gpu_index"], as_index=False).agg(
    cap_med=("cap_MiB", "median"), cap_max=("cap_MiB", "max"),
    free_max=("memory_free_MiB", "max"),
    pow_max=("power_draw_W", "max"), temp_max=("temperature_gpu", "max"),
    links=("pcie_link_width_current", "max"), n=("cap_MiB", "size"))
print(f"devices sampled = {len(dev)}")
cap = dev["cap_med"].round(0)
print("\n--- (a) capacity tiers, per-device MEDIAN of (memory_free + memory_used)")
print(cap.value_counts().to_string())
print("    distinct capacities:", sorted(cap.unique()))
print("    [old, wrong estimator] max memory_free distinct:",
      sorted(dev['free_max'].round(0).unique()))

per_node = dev.groupby("Node").size()
print(f"\n--- (b) GPUs per node: {per_node.value_counts().to_dict()}")
print("    power_draw max per device (W):",
      dev["pow_max"].describe()[["min", "50%", "max"]].round(1).to_dict())
print("    pcie link width values:", sorted(dev["links"].dropna().unique()))

lab = pd.read_csv(os.path.expanduser("~/xcad/datasets/mit/labelled_jobids.csv"))
sj = set(pd.to_numeric(inv["id_job"], errors="coerce").dropna().astype("int64"))
print(f"\n--- (c) distinct id_jobs in sample = {len(sj):,}; overlap with the "
      f"3,430 model-labelled jobs = {len(sj & set(lab['id_job']))}")
print("\nDONE")
