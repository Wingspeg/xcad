"""Can per-GPU memory *capacity* be measured from the Alibaba trace?

The trace publishes no GPU memory anywhere (machine_spec.cap_mem is host RAM),
so today a device's memory only exists if we look the model name up outside the
trace.  But pai_sensor_table reports, per instance and per /dev/nvidiaN,

    gpu_wrk_util      "number of GPUs used in percentage" (100 = one whole GPU)
    max_gpu_wrk_mem   "GB of GPU memory used (maximum)"

A use figure is physically bounded by the capacity of the device it was taken
on, so the *envelope* of single-GPU instances is a measurable lower bound on
that device's memory.  Two views are computed:

  A. rows with 90 <= gpu_wrk_util <= 110  -> one whole GPU: the memory number is
     already per-device, so max over instances bounds that card's capacity.
  B. rows with gpu_wrk_util >= 100        -> memory / (util/100) as a per-GPU
     estimate, a wider sample but an arithmetic one.

If view A caps just under the catalogue capacity of the named model, the
model->memory mapping is confirmed *inside* the trace.  If it also caps the
MISC bucket, the merged bucket is partly resolvable with no external table.

  python3 scripts/probe_gpu_mem_from_sensor.py            # full scan
  python3 scripts/probe_gpu_mem_from_sensor.py --rows 2e6 # smoke test
"""
import argparse
import os

import numpy as np
import pandas as pd

RAW = os.path.expanduser("~/xcad/datasets/alibaba-cluster-trace-gpu-v2020/raw")
COLS = ["job_name", "task_name", "worker_name", "inst_id", "machine", "gpu_name",
        "cpu_usage", "gpu_wrk_util", "avg_mem", "max_mem",
        "avg_gpu_wrk_mem", "max_gpu_wrk_mem", "read", "write", "read_count", "write_count"]
USECOLS = ["machine", "gpu_name", "gpu_wrk_util", "max_gpu_wrk_mem"]
GE = [4, 8, 10, 12, 16, 20, 24, 32]
AGG = dict(dev_max=("max_gpu_wrk_mem", "max"), n_rows=("max_gpu_wrk_mem", "size"))


def envelope(dev, label, tmap):
    dev = dev.copy()
    dev["gpu_type"] = dev["machine"].map(tmap).fillna("NOT_IN_SPEC")
    g = dev.groupby("gpu_type", observed=True)["dev_max"]
    rep = pd.DataFrame({
        "devices": g.size(),
        "p50": g.quantile(0.50).round(2),
        "p90": g.quantile(0.90).round(2),
        "p99": g.quantile(0.99).round(2),
        "max": g.max().round(2),
    })
    for thr in GE:
        rep[f"dev_ge_{thr}G"] = g.apply(lambda s, x=thr: int((s >= x).sum()))
    print(f"\n== view {label}: per-device envelope of GPU memory use (GB) ==")
    print(rep.to_string())
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=float, default=np.inf, help="cap scanned rows (smoke test)")
    args = ap.parse_args()

    spec = pd.read_csv(os.path.join(RAW, "pai_machine_spec.csv"), header=None,
                       names=["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"],
                       dtype="string")
    spec["gpu_type"] = spec["gpu_type"].fillna("")
    tmap = dict(zip(spec["machine"], spec["gpu_type"]))
    print(f"machine_spec rows={len(spec):,}  gpu_type="
          f"{spec.gpu_type.value_counts().to_dict()}")

    a_parts, b_parts, tot = [], [], 0
    util_hist = {}
    for ch in pd.read_csv(os.path.join(RAW, "pai_sensor_table.csv"), header=None,
                          names=COLS, usecols=USECOLS, chunksize=4_000_000):
        ch = ch[ch["max_gpu_wrk_mem"].notna() & ch["gpu_wrk_util"].notna()]
        tot += len(ch)
        u = ch["gpu_wrk_util"]
        # A: exactly one whole GPU allocated to the instance
        a = ch[(u >= 90) & (u <= 110) & (ch.max_gpu_wrk_mem > 0)]
        if len(a):
            a_parts.append(a.groupby(["machine", "gpu_name"], sort=False).agg(**AGG))
        # B: one or more whole GPUs, memory divided by the GPU count
        b = ch[(u >= 100) & (ch.max_gpu_wrk_mem > 0)].copy()
        if len(b):
            b["max_gpu_wrk_mem"] = b["max_gpu_wrk_mem"] / (b["gpu_wrk_util"] / 100.0)
            b_parts.append(b.groupby(["machine", "gpu_name"], sort=False).agg(**AGG))
        for k, v in u.round(0).value_counts().head(6).items():
            util_hist[k] = util_hist.get(k, 0) + int(v)
        print(f"  scanned {tot:,} metric rows (A={len(a_parts):,} chunks)", flush=True)
        if tot >= args.rows:
            break

    print(f"\ntop gpu_wrk_util values (100 = one whole GPU): "
          f"{dict(sorted(util_hist.items(), key=lambda x: -x[1])[:10])}")
    if a_parts:
        envelope(pd.concat(a_parts).groupby(level=[0, 1]).max().reset_index(), "A", tmap)
    if b_parts:
        envelope(pd.concat(b_parts).groupby(level=[0, 1]).max().reset_index(), "B", tmap)

    # sanity: what the same devices say when we do not filter on GPU count at all
    allp = []
    for ch in pd.read_csv(os.path.join(RAW, "pai_sensor_table.csv"), header=None,
                          names=COLS, usecols=USECOLS, chunksize=4_000_000):
        ch = ch[ch["max_gpu_wrk_mem"].notna() & (ch["max_gpu_wrk_mem"] > 0)]
        allp.append(ch.groupby(["machine", "gpu_name"], sort=False).agg(**AGG))
        tot += len(ch)
        if tot >= 2 * args.rows:
            break
    envelope(pd.concat(allp).groupby(level=[0, 1]).max().reset_index(), "C (unfiltered)", tmap)


if __name__ == "__main__":
    main()
