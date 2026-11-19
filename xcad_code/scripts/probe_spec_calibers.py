"""One MISC, several legitimate percentages -- settle the denominators.

Reported at every granularity the cleaning publishes, from the published tables
only, so the feasibility doc can quote one number per claim:

  placements        : one instance-machine allocation
  placement_devices : one card actually touched  (the conservative choice)
  jobs (label set)  : gpu_label_final is a SET, so a job can be both
  jobs (device rows): job_gid actually present in placement_devices

The first probe (probe_spec_mapping.py Q2) produced 50.34% / 60.83% for MISC;
those came from joining placements to device labels, which double-counts a
placement whose machine mixed labels.  This script is the corrected version.

usage: ~/xcad/xcad_code/.venv/bin/python scripts/probe_spec_calibers.py
"""
import os
import pandas as pd

R = os.path.expanduser("~/xcad/datasets/clean")
DS = "ali_v2020"
OPAQUE = {"NVIDIA-LEGACY-MISC", "UNRESOLVED", "GPU", "GPU-UNKNOWN", ""}


def load(t):
    return pd.read_parquet(os.path.join(R, DS, f"{t}.parquet"))


def isop(s):
    s = str(s).upper()
    return s in OPAQUE or "MISC" in s or s == "UNRESOLVED"


pl = load("placements")
pdv = load("placement_devices")
dv = load("devices")
jobs = load("jobs")

gp = pl[pl["is_gpu_placement"] == True]  # noqa: E712
print(f"1) placements  n={len(gp):,}")
sh = gp["gpu_label"].astype(str).value_counts()
print((100 * sh / len(gp)).round(2).to_string())
print(f"   opaque share = {100*gp['gpu_label'].astype(str).map(isop).mean():.2f}%")

key = dv.set_index("device_gid")["gpu_label"]
d = pdv.assign(gpu_label=pdv["device_gid"].map(key))
d = d[d["gpu_label"].notna()]
print(f"\n2) placement_devices (one row = one card)  n={len(d):,}")
sh = d["gpu_label"].astype(str).value_counts()
print((100 * sh / len(d)).round(2).to_string())
print(f"   opaque share = {100*d['gpu_label'].astype(str).map(isop).mean():.2f}%")

j = jobs[jobs["is_gpu_job"] == True]  # noqa: E712
j = j.drop_duplicates("job_gid")
sets = j["gpu_label_final"].astype(str)
has = sets.str.len() > 0
touch = sets.map(lambda s: any(isop(x) for x in s.split("|") if x))
only = sets.map(lambda s: all(isop(x) for x in s.split("|") if x) and len(s) > 0)
print(f"\n3) jobs, gpu_label_final is a SET  n(gpu jobs)={len(j):,} labelled={int(has.sum()):,}")
print(f"   touched by an opaque label  = {100*touch[has].mean():.2f}%")
print(f"   fully opaque (nothing named) = {100*only[has].mean():.2f}%")
print(f"   fully resolvable            = {100*((~touch)[has]).mean():.2f}%")

jj = d.groupby("job_gid", observed=True)["gpu_label"].agg(
    lambda s: any(isop(x) for x in s.astype(str)))
print(f"\n4) jobs that actually have device rows  n={len(jj):,}  "
      f"touched by opaque = {100*jj.mean():.2f}%")

print("\n5) 老探针的口径问题，记录下来：probe_spec_mapping.py Q2 的 "
      "gpu_placements_pct 与本脚本 (1) 完全一致（50.34%），不是膨胀 join；"
      "但它的 jobs_pct 把'标签集合里含 MISC 的同一个作业'同时计入 MISC 与 "
      "P100 两行，分母被放大，于是 MISC 报成 60.83%。作业粒度的正确值是上面 (3)："
      "碰到 = 68.15%，整作业无具名标签 = 60.16%，整作业全具名 = 31.85%。")
