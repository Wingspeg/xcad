"""Per-GPU memory capacity measured from the trace itself (no external lookup).

`max_gpu_wrk_mem` is reported per (worker, /dev/nvidiaN) row, but a worker may
hold several GPUs, so the raw figure is a worker total, not a device figure.
Restrict to workers that the sensor table only ever shows on ONE device: for
those rows the memory number is a usage figure on a single card, and the max
over all such rows on a device is a hard lower bound on that device's capacity
(a worker cannot allocate more memory than the card has).

The named buckets (P100 / T4 / V100 / V100M32) act as the control: if their
envelope lands just under the catalogue capacity, the estimator works, and the
envelope of the MISC bucket is then a measurement of that bucket's memory --
something no column in the trace publishes.

  python3 scripts/probe_gpu_mem_single_gpu.py
"""
import os

import pandas as pd

RAW = os.path.expanduser("~/xcad/datasets/alibaba-cluster-trace-gpu-v2020/raw")
COLS = ["job_name", "task_name", "worker_name", "inst_id", "machine", "gpu_name",
        "cpu_usage", "gpu_wrk_util", "avg_mem", "max_mem",
        "avg_gpu_wrk_mem", "max_gpu_wrk_mem", "read", "write", "read_count", "write_count"]
GE = [4, 6, 8, 10, 11, 12, 14, 16, 20, 24, 32]

sen = pd.read_csv(os.path.join(RAW, "pai_sensor_table.csv"), header=None, names=COLS,
                  usecols=["worker_name", "machine", "gpu_name", "gpu_wrk_util",
                           "max_gpu_wrk_mem"])
print(f"sensor rows={len(sen):,}  workers={sen.worker_name.nunique():,}  "
      f"devices={sen.groupby(['machine', 'gpu_name']).ngroups:,}")

# how many distinct devices does each worker appear on?
per_worker = sen.groupby("worker_name", sort=False)["gpu_name"].nunique()
sen["n_dev_of_worker"] = sen["worker_name"].map(per_worker)
single = sen[(sen.n_dev_of_worker == 1) & sen.max_gpu_wrk_mem.notna()
             & (sen.max_gpu_wrk_mem > 0)]
print(f"\nrows on workers pinned to one device: {len(single):,} "
      f"({100 * len(single) / len(sen):.1f}%)  "
      f"devices covered: {single.groupby(['machine', 'gpu_name']).ngroups:,}")

dev = (single.groupby(["machine", "gpu_name"], sort=False)
       .agg(cap_lb=("max_gpu_wrk_mem", "max"), n_workers=("worker_name", "nunique"))
       .reset_index())

spec = pd.read_csv(os.path.join(RAW, "pai_machine_spec.csv"), header=None,
                   names=["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"],
                   dtype="string")
dev["gpu_type"] = dev["machine"].map(dict(zip(spec.machine, spec.gpu_type.fillna(""))))
dev["gpu_index"] = dev["gpu_name"].str.extract(r"nvidia(\d+)", expand=False)

print("\n== lower bound on per-device memory, by the machine's declared gpu_type ==")
g = dev.groupby("gpu_type", observed=True)["cap_lb"]
rep = pd.DataFrame({"devices": g.size(), "p10": g.quantile(.10).round(2),
                    "p50": g.quantile(.50).round(2), "p90": g.quantile(.90).round(2),
                    "p99": g.quantile(.99).round(2), "max": g.max().round(2),
                    "n_workers": dev.groupby("gpu_type", observed=True)["n_workers"].sum()})
for t in GE:
    rep[f"ge_{t}G"] = g.apply(lambda s, x=t: int((s >= x).sum()))
print(rep.to_string())

print("\n== same, per MISC machine: does one machine stay inside one capacity? ==")
mis = dev[dev.gpu_type == "MISC"]
m = mis.groupby("machine")["cap_lb"].agg(["size", "min", "max", "median"]).round(2)
print(m["max"].describe().round(2).to_string())
print("\nmachines whose device envelope exceeds 12 GB (would rule out K80/K40m): "
      f"{int((m['max'] > 12).sum())} of {len(m)}")
print("machines whose device envelope exceeds 8 GB  (would rule out M60):        "
      f"{int((m['max'] > 8).sum())} of {len(m)}")
print("\nhistogram of the per-machine envelope (GB), MISC:")
print(pd.cut(m["max"], [0, 4, 6, 8, 10, 12, 16, 24, 32, 1e9]).value_counts().sort_index().to_string())

print("\n== control: named buckets vs catalogue capacity ==")
CAT = {"P100": 16, "T4": 16, "V100": 16, "V100M32": 32}
for k, v in CAT.items():
    s = dev[dev.gpu_type == k]["cap_lb"]
    print(f"  {k:<9} catalogue {v:>2} GB   measured p99={s.quantile(.99):6.2f} "
          f"max={s.max():6.2f}   devices above catalogue: {int((s > v).sum())}/{len(s)}")
