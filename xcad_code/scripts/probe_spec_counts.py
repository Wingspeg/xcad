"""Follow-up counts for probe_spec_join.py: granularity + tier-claim checks.

C1  Ali: MISC / resolvable share at three granularities (device rows, GPU
    placement rows, distinct jobs) so the report quotes one consistent number.
C2  Philly: does the machine-list tier claim (n x mem) agree with the number of
    devices actually observed on that machine?  n_gpus_per_node is only a
    shared attribute if both traces mean the same thing by it.

usage: ~/xcad/xcad_code/.venv/bin/python scripts/probe_spec_counts.py
"""
import os
import pandas as pd

R = os.path.expanduser("~/xcad/datasets/clean")
pd.set_option("display.width", 200)
OPAQUE = {"NVIDIA-LEGACY-MISC", "UNRESOLVED", "GPU", "GPU-UNKNOWN", ""}


def load(ds, t):
    p = os.path.join(R, ds, f"{t}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


def opaque(lab):
    return str(lab).upper() in OPAQUE


print("=" * 78)
print("C0  published columns per table")
print("=" * 78)
for ds in ("ali_v2020", "philly"):
    for t in ("placements", "placement_devices", "jobs"):
        df = load(ds, t)
        if df is None:
            print(f"{ds:<11} {t:<18} MISSING")
        else:
            print(f"{ds:<11} {t:<18} {len(df):>9,}  {list(df.columns)}")

print("=" * 78)
print("C1  ali_v2020 exposure, three granularities")
print("=" * 78)
dv = load("ali_v2020", "devices")
pl = load("ali_v2020", "placements")
pdv = load("ali_v2020", "placement_devices")
lab = dv.set_index("device_gid")["gpu_label"]
pdv = pdv.assign(gpu_label=pdv["device_gid"].map(lab))
pdv["opaque"] = pdv["gpu_label"].map(opaque)
gpu_pl = pdv[pdv["gpu_label"].notna()]
pid = [c for c in ("placement_gid", "request_id", "instance_gid") if c in pdv.columns]
print(f"device rows      : n={len(dv):,}  opaque {100*dv['gpu_label'].map(opaque).mean():6.2f}%")
print(f"gpu placement rows: n={len(gpu_pl):,}  opaque {100*gpu_pl['opaque'].mean():6.2f}%")
if pid:
    per_pl = gpu_pl.groupby(pid[0], observed=True)["opaque"].agg(["any", "all"])
    print(f"distinct {pid[0]:<14}: n={len(per_pl):,}  touched by opaque "
          f"{100*per_pl['any'].mean():6.2f}%   fully opaque {100*per_pl['all'].mean():6.2f}%")
job_col = "job_gid" if "job_gid" in pdv.columns else None
if job_col:
    per_job = gpu_pl.groupby(job_col, observed=True)["opaque"].agg(["any", "all"])
    print(f"distinct gpu jobs  : n={len(per_job):,}  touched by opaque "
          f"{100*per_job['any'].mean():6.2f}%   fully opaque {100*per_job['all'].mean():6.2f}%")
print("\nlabel share over gpu placement rows:")
s = gpu_pl["gpu_label"].value_counts()
print((100 * s / s.sum()).round(2).to_string())

print()
print("=" * 78)
print("C2  philly: tier claim vs observed device count per machine")
print("=" * 78)
pdv2 = load("philly", "devices")
ext = pdv2["gpu_label"].str.extract(r"TIER-(\d+)x(\d+)").astype("float")
pdv2 = pdv2.assign(claim_n=ext[0], claim_mem=ext[1])
per_m = pdv2.groupby(["gpu_label", "machine_gid"], observed=True).agg(
    observed=("device_gid", "size"), claim_n=("claim_n", "first"),
    claim_mem=("claim_mem", "first")).reset_index()
print(f"machines={len(per_m):,}")
print(per_m.groupby("gpu_label", observed=True)["observed"].describe().to_string())
print("\nobserved == claim ?")
per_m["eq"] = per_m["observed"] == per_m["claim_n"]
print(per_m.groupby("gpu_label", observed=True)["eq"].agg(["sum", "size", "mean"]).to_string())
print("\nobserved value counts by tier:")
print(per_m.groupby(["gpu_label", "observed"], observed=True).size().to_string())
