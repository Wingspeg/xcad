"""Second attempt at measuring per-GPU capacity from the trace, with the GPU
count taken from the *scheduler's own request* instead of from a sensor column.

pai_sensor_table.max_gpu_wrk_mem is an instance-level figure (a worker holding
8 GPUs reports the total), so dividing it by the number of GPUs the instance was
granted turns it into a per-device figure.  That count is available:

    pai_task_table.plan_gpu  = GPUs requested for the task, in percent (100=1)
    pai_task_table.inst_num  = instances launched by the task
    -> per-instance GPUs = plan_gpu / 100 / inst_num

Only rows whose per-instance count is a whole number >= 1 are used, so no
fractional-GPU sharing can blur the division.  For each device we keep the max
of the resulting per-GPU usage; a device can never hand out more memory than it
has, so that max is a lower bound on its capacity.

The named buckets are the control: P100/T4/V100 should bound at 16 GB and
V100M32 at 32 GB.  Whatever the MISC bucket bounds at is a measurement of a
quantity no column of the trace publishes.

  python3 scripts/probe_gpu_mem_by_request.py
"""
import os

import pandas as pd

RAW = os.path.expanduser("~/xcad/datasets/alibaba-cluster-trace-gpu-v2020/raw")
GE = [4, 6, 8, 10, 11, 12, 14, 16, 20, 24, 32]
CAT = {"P100": 16, "T4": 16, "V100": 16, "V100M32": 32}

task = pd.read_csv(os.path.join(RAW, "pai_task_table.csv"), header=None,
                   names=["job_name", "task_name", "inst_num", "status", "start_time",
                          "end_time", "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"],
                   usecols=["job_name", "task_name", "inst_num", "plan_gpu"])
for c in ("inst_num", "plan_gpu"):
    task[c] = pd.to_numeric(task[c], errors="coerce")
task["n_inst_gpu"] = task["plan_gpu"] / 100.0 / task["inst_num"]
ok = task[(task.n_inst_gpu >= 1) & (task.plan_gpu % 100 == 0)].drop_duplicates(
    ["job_name", "task_name"])
print(f"task rows={len(task):,}  whole-GPU-per-instance rows usable={len(ok):,} "
      f"({100 * len(ok) / len(task):.1f}%)  n_gpus distribution="
      f"{ok.n_inst_gpu.value_counts().head(8).to_dict()}")

sen = pd.read_csv(os.path.join(RAW, "pai_sensor_table.csv"), header=None,
                  names=["job_name", "task_name", "worker_name", "inst_id", "machine",
                         "gpu_name", "cpu_usage", "gpu_wrk_util", "avg_mem", "max_mem",
                         "avg_gpu_wrk_mem", "max_gpu_wrk_mem", "read", "write",
                         "read_count", "write_count"],
                  usecols=["job_name", "task_name", "machine", "gpu_name",
                           "gpu_wrk_util", "max_gpu_wrk_mem"])
j = sen.merge(ok[["job_name", "task_name", "n_inst_gpu"]], on=["job_name", "task_name"],
              how="inner")
j = j[j["max_gpu_wrk_mem"].notna() & (j["max_gpu_wrk_mem"] > 0)]
j["per_gpu"] = j["max_gpu_wrk_mem"] / j["n_inst_gpu"]
print(f"sensor rows={len(sen):,}  joined to a whole-GPU request={len(j):,} "
      f"({100 * len(j) / len(sen):.1f}%)  devices covered="
      f"{j.groupby(['machine', 'gpu_name']).ngroups:,}")

dev = (j.groupby(["machine", "gpu_name"], sort=False)
       .agg(cap_lb=("per_gpu", "max"), n_obs=("per_gpu", "size")).reset_index())

spec = pd.read_csv(os.path.join(RAW, "pai_machine_spec.csv"), header=None,
                   names=["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"],
                   dtype="string")
dev["gpu_type"] = dev["machine"].map(dict(zip(spec.machine, spec.gpu_type.fillna(""))))

print("\n== per-device lower bound on memory capacity (GB), by declared gpu_type ==")
g = dev.groupby("gpu_type", observed=True)["cap_lb"]
rep = pd.DataFrame({"devices": g.size(), "p10": g.quantile(.10).round(2),
                    "p50": g.quantile(.50).round(2), "p90": g.quantile(.90).round(2),
                    "p99": g.quantile(.99).round(2), "max": g.max().round(2)})
for t in GE:
    rep[f"ge_{t}G"] = g.apply(lambda s, x=t: int((s >= x).sum()))
print(rep.to_string())

print("\n== control: does the estimator respect the catalogue capacity? ==")
for k, v in CAT.items():
    s = dev[dev.gpu_type == k]["cap_lb"]
    print(f"  {k:<9} catalogue {v:>2} GB   p50={s.quantile(.50):6.2f} "
          f"p90={s.quantile(.90):6.2f} p99={s.quantile(.99):6.2f} max={s.max():6.2f}  "
          f"above catalogue: {int((s > v).sum())}/{len(s)}")

print("\n== MISC bucket: per-machine envelope ==")
mis = dev[dev.gpu_type == "MISC"].groupby("machine")["cap_lb"].agg(["size", "max"]).round(2)
print(mis["max"].describe().round(2).to_string())
print("\nhistogram of per-machine envelope (GB):")
print(pd.cut(mis["max"], [0, 6, 8, 10, 12, 16, 24, 32, 1e9])
      .value_counts().sort_index().to_string())
print(f"\nMISC machines with an envelope above 12 GB (rules out K40m/K80 per-GPU): "
      f"{int((mis['max'] > 12).sum())}/{len(mis)}")
print(f"MISC machines with an envelope above 8 GB  (rules out M60 per-GPU):        "
      f"{int((mis['max'] > 8).sum())}/{len(mis)}")
