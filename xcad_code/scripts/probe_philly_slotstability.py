"""Decisive check on the Philly 2x12 / 8-device contradiction.

If a machine's declared GPU count is 2 but device indices 0..7 appear, one of
two things is true:
  (a) one attempt really occupies >2 devices  -> the machine list understates
      the hardware, and n_gpus_per_node must come from the inventory;
  (b) every attempt occupies <=2 devices and only their union over time is 8
      -> gpu_index is not a stable physical slot id, so a per-device
      capability vector cannot be built from it at all.

usage: ~/xcad/xcad_code/.venv/bin/python scripts/probe_philly_slotstability.py
"""
import os
import pandas as pd

R = os.path.expanduser("~/xcad/datasets/clean")
pl = pd.read_parquet(os.path.join(R, "philly", "placements.parquet"))
pdv = pd.read_parquet(os.path.join(R, "philly", "placement_devices.parquet"))

print("placements: gpu_ids sample")
print(pl[["machine", "gpu_label", "n_gpus", "gpu_ids"]].head(8).to_string())

per = pl.assign(k=pl["gpu_ids"].astype(str)).groupby("gpu_label", observed=True)["n_gpus"]
print("\nn_gpus per placement, by tier:")
print(per.agg(["min", "max", "mean", lambda s: (s > 2).sum()]).to_string())

# per-attempt distinct device indices on 2x12 machines
k = ["job_id", "attempt", "machine"]
g = pdv.groupby(k + ["gpu_label"], observed=True)["gpu_index"].agg(["size", "nunique", "max"]).reset_index()
g["max"] = g["max"].astype(int)
print("\ndevices touched by ONE attempt, by tier:")
print(g.groupby("gpu_label", observed=True)[["size", "nunique", "max"]].agg(["min", "max", "mean"]).to_string())
t12 = g[g["gpu_label"] == "PHILLY-TIER-2x12"]
print(f"\n2x12 attempts: n={len(t12):,}  with >2 distinct indices: "
      f"{int((t12['nunique'] > 2).sum()):,} ({100*(t12['nunique'] > 2).mean():.2f}%)  "
      f"with index >=2: {int((t12['max'] >= 2).sum()):,} "
      f"({100*(t12['max'] >= 2).mean():.2f}%)")

# do the indices a machine uses stay in a fixed subset over time?
dv = pd.read_parquet(os.path.join(R, "philly", "devices.parquet"))
sub = dv[dv["gpu_label"] == "PHILLY-TIER-2x12"].groupby("machine_gid", observed=True)["gpu_index"]
sets = sub.agg(lambda s: tuple(sorted(set(s.astype(str)))))
print("\nunion-of-indices per 2x12 machine, value counts:")
print(sets.value_counts().head(6).to_string())

# cross-machine reuse: is index 2 used by many 2-GPU machines? (would show ids are per-job views)
n12 = sets[sets.map(len) == 8]
print(f"\n2x12 machines whose observed index union has size 8: {len(n12)}")
