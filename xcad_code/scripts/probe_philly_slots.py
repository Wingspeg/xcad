"""Why does a PHILLY-TIER-2x12 machine show 8 devices?

Checks, in order:
  D1  per-machine gpu_index sets, split by tier label
  D2  nodes table: n_slots_declared vs n_slots_observed, slot_conflict
  D3  does one ip carry several cluster_machine_list entries (ip reuse)?
  D4  placements.cap_conflict / label_provenance counts

usage: ~/xcad/xcad_code/.venv/bin/python scripts/probe_philly_slots.py
"""
import os
import pandas as pd

R = os.path.expanduser("~/xcad/datasets/clean")
D = os.path.expanduser("~/xcad/datasets")


def load(t):
    return pd.read_parquet(os.path.join(R, "philly", f"{t}.parquet"))


dv = load("devices")
nd = load("nodes")
pl = load("placements")

print("=" * 78)
print("D1  gpu_index sets per machine, by tier")
print("=" * 78)
g = dv.groupby(["gpu_label", "machine_gid"], observed=True)["gpu_index"]
idx = g.agg(["size", "min", "max", lambda s: sorted(set(s.astype(str)))])
idx.columns = ["n_rows", "imin", "imax", "set"]
print(idx.groupby(["gpu_label", "n_rows"], observed=True).size().to_string())
print("\nsample rows:")
print(idx.reset_index().head(8).to_string())
print("\ndistinct gpu_index spellings:", sorted(set(dv["gpu_index"].astype(str)))[:12])

print()
print("=" * 78)
print("D2  nodes: declared vs observed slots")
print("=" * 78)
print(nd.groupby("gpu_label", observed=True)[
    ["n_slots_declared", "n_slots_observed", "cap_gpu"]].agg(["min", "max", "mean"]).to_string())
print("\nslot_conflict:", nd["slot_conflict"].value_counts(dropna=False).to_dict())
print("label_provenance:", nd["label_provenance"].value_counts(dropna=False).to_dict())
dcl = dv.groupby(["gpu_label", "machine_gid"], observed=True).size().rename("obs")
j = nd.set_index("machine_gid").join(dcl)
print("\nmachines where observed devices != declared slots:",
      int((j["obs"] != j["n_slots_declared"]).sum()), "of", int(j["obs"].notna().sum()))

print()
print("=" * 78)
print("D3  raw cluster_machine_list: one ip -> several entries?")
print("=" * 78)
import glob
cand = glob.glob(os.path.join(D, "**", "*cluster_machine_list*"), recursive=True)
print("files:", cand)
for f in cand[:1]:
    raw = pd.read_csv(f, header=None)
    print(f"{f}  shape={raw.shape}")
    print(raw.head(6).to_string())
    vc = raw.groupby(0).size().value_counts().head(6)
    print("entries per ip:", vc.to_dict())

print()
print("=" * 78)
print("D4  placements capacity conflict + provenance")
print("=" * 78)
print("cap_conflict:", pl["cap_conflict"].value_counts(dropna=False).to_dict())
print("label_provenance:", pl["label_provenance"].value_counts(dropna=False).to_dict())
print("n_gpus per placement:", pl["n_gpus"].value_counts().head(10).to_dict())
