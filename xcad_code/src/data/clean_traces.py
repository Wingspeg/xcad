#!/usr/bin/env python3
"""
clean_traces.py -- normalise heterogeneous GPU-cluster traces into ONE tidy
star schema, so that any later experiment (scheduling, runtime prediction,
placement, graph learning) can read all of them with dataset-agnostic code.

Datasets handled
----------------
  ali_v2020  Alibaba cluster-trace-gpu-v2020 (PAI; ML jobs; GPU sharing)
  philly     Microsoft Philly (YARN + YarnRescheduler; GPU jobs)
  eagle      NREL Eagle jobs (github NatLabRockies/eagle-jobs, OEDI 5860; 11.03M jobs)

Released table set -- every name is always written; tables a trace cannot
support are written EMPTY and the reason is recorded in manifest.absent_tables:

  jobs.parquet                one row per job, universal columns
  requests.parquet            the finest REQUEST grain published, when finer than one job
  placements.parquet          one row per (job, allocation unit, machine)
  placement_devices.parquet   one row per (job, machine, GPU device index)
  nodes.parquet               machine inventory
  devices.parquet             GPU device inventory (machine x index)
  gpu_types.parquet           canonical GPU-label dictionary + provenance
  subclusters.parquet         partition / queue dimension, when one exists
  manifest.json  profile.md   counts, per-column coverage, recurrence report,
                              id-space invariants, every drop/derive decision

Design rules
------------
R1  Ids are typed and prefixed: gid = "<dataset>#<segment>:<raw id>".  Two
    segments can never collide; the invariant is asserted and printed.
R2  Every GPU label carries `label_provenance` -- observed (device/sensor),
    joined (machine table), declared (request column), derived (count only).
    Conflicts are kept and flagged, never silently dropped.
R3  Head entities are COLUMNS, not a decision: user / account / app / job are
    all exposed and each one's recurrence statistics are reported, so the
    experiment picks the entity instead of inheriting one from the loader.
R4  Time is normalised per dataset: seconds since that trace's own t0 (`t_*_s`),
    a daily snapshot index (`snap_*`), time-of-day (`tod_*`), and UTC strings
    only where the trace publishes real wall-clock.  Alibaba v2020 ships
    de-identified dates, so its `*_utc` columns are NULL by design.
R5  Nothing is renamed into a claim.  Philly publishes no GPU model name, so its
    labels stay PHILLY-TIER-2x12GB / PHILLY-TIER-8x24GB rather than being
    guessed as K80 / DGX.

Usage
  python3 clean_traces.py --dataset ali_v2020 \
      --raw ~/xcad/datasets/alibaba-cluster-trace-gpu-v2020/raw \
      --out ~/xcad/datasets/clean/ali_v2020
  python3 clean_traces.py --dataset philly --raw ~/xcad/datasets/philly/trace-data \
      --out ~/xcad/datasets/clean/philly
  python3 clean_traces.py --dataset eagle --raw ~/xcad/datasets/eagle/raw \
      --out ~/xcad/datasets/clean/eagle
  python3 clean_traces.py --dataset mit --raw ~/xcad/datasets/mit \
      --out ~/xcad/datasets/clean/mit
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

DAY = 86400.0
NO_GPU = "CPU-ONLY"
UNRESOLVED = "UNRESOLVED"

# ------------------------------------------------------------------ vocabularies
# Only spellings the trace itself documents are canonicalised; an unknown
# spelling is kept verbatim and reported.  Merging card models quietly is the
# kind of decision that invalidates a placement claim later.
ALI_GPU_LABEL = {
    "V100M32": "NVIDIA-V100-32G",
    "V100M16": "NVIDIA-V100-16G",
    "V100": "NVIDIA-V100",        # bare "V100" in v2020: memory not stated
    "P100": "NVIDIA-P100",
    "T4": "NVIDIA-T4",
    "A10": "NVIDIA-A10",
    "G1": "ALIBABA-INTERNAL-G1",
    "G2": "ALIBABA-INTERNAL-G2",
    "G3": "ALIBABA-INTERNAL-G3",
    "MISC": "NVIDIA-LEGACY-MISC",  # README: "older generations, e.g. K40m/K80/M60"
    "CPU": NO_GPU,
    "": NO_GPU,
}
PHILLY_TIER = "PHILLY-TIER-{n}x{mem}"

# --- MIT Supercloud (datacenter-challenge/202201) ------------------------------
# The only GPU "type" MIT publishes for these jobs is the Slurm TRES *resource
# class* name, read from tres-mapping.txt (1001 -> gpu:tesla, 1002 -> gpu:volta).
# It names an architecture, not a card model, and no per-node inventory or memory
# size is published in this release -- so the label keeps the class name verbatim
# (R5) instead of being promoted to "V100-32G".
MIT_TRES = {1: "cpu", 2: "mem", 3: "energy", 4: "node", 5: "billing", 6: "fs",
            7: "vmem", 8: "pages", 1001: "gpu:tesla", 1002: "gpu:volta"}
MIT_GPU_TIER = {"gpu:tesla": "MIT-TIER-tesla", "gpu:volta": "MIT-TIER-volta"}
# Slurm dumps write a literal backslash-N for "no value" and pandas does NOT treat
# that as missing.  Without this, `constraints` looks 100% populated (1,707 of its
# 3,430 values are \N) and `gres_used` -- the column that caused the wrong
# exclusion in S9.5 -- is mis-read again.
MIT_NA = ["", "\\N", "N/A", "NULL", "None"]
MIT_NODE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")

STATE_MAP = {
    # Alibaba PAI
    "terminated": "success", "waiting": "pending", "restart": "restarted",
    # Microsoft Philly / YARN
    "pass": "success", "accept": "pending",
    # Slurm (Eagle)
    "completed": "success", "cancelled": "killed", "canceled": "killed",
    "pending": "pending", "timeout": "timeout", "node_down": "node_down",
    "preempted": "preempted", "out_of_memory": "oom", "deadline": "timeout",
    # shared
    "failed": "failed", "running": "running", "killed": "killed",
}

# Naive published clocks -> UTC.  None = the trace ships de-identified dates,
# so no real UTC exists and the *_utc columns stay NULL.
DATASET_TZ = {"ali_v2020": None, "philly": "America/New_York", "eagle": "America/Denver",
              # MIT ships epoch seconds straight out of Slurm accounting: already UTC.
              "mit": "UTC"}

HEAD_KEYS = {
    "ali_v2020": ["user_id", "account_id", "app_name", "framework", "job_id"],
    "philly": ["user_id", "account_id", "job_id"],
    "eagle": ["user_id", "account_id", "app_name", "queue_id", "job_id"],
    "mit": ["user_id", "app_name", "queue_id", "job_id"],
}

# The columns every jobs.parquet MUST carry, so one loader reads all three
# traces.  A trace that does not publish one of them gets the column filled NA
# -- a hole you can query is better than a name that silently differs per file.
CORE_JOB_COLS = [
    "dataset", "job_id", "job_gid", "task_ordinal", "n_array_tasks", "is_array_task",
    "native_id_missing", "user_id", "user_gid", "account_id", "account_gid",
    "app_name", "queue_id", "qos", "state_raw", "state_norm",
    "n_alloc_units", "n_machines", "req_cpus", "req_mem_gib",
    "n_gpus_req", "n_gpus_obs", "gpu_label_req", "gpu_label_obs", "gpu_label_final",
    "label_provenance", "is_gpu_job",
    "t_submit_s", "t_submit_utc", "snap_submit", "tod_submit",
    "t_start_s", "t_start_utc", "snap_start", "tod_start",
    "t_end_s", "t_end_utc", "snap_end", "tod_end", "wait_s", "run_s",
]


def ensure_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def resolve_fk(fact: pd.DataFrame, key_col: str, gid_col: str, dim_values,
               notes: dict, label: str) -> pd.DataFrame:
    """A foreign key points at a dimension member or at nothing -- never at a phantom.

    Alibaba's sensor table names a few thousand job ids that the released
    `pai_job_table` has no record for.  Leaving them as `ali_v2020#job:<name>`
    would silently mint 8k jobs with no user, no timestamps and no attributes --
    the kind of node that turns up later as an unexplainable hub in the graph.
    The evidence row (this worker held this card on this machine) stays, its link
    becomes NA, and the count is reported in the manifest.
    """
    if not len(fact) or key_col not in fact.columns:
        return fact
    bad = fact[key_col].notna() & ~fact[key_col].isin(dim_values)
    notes[label] = int(bad.sum())
    if int(bad.sum()):
        fact.loc[bad, gid_col] = pd.NA
    return fact


EMPTY_SCHEMA = {
    "jobs": ["dataset", "job_gid", "job_id"],
    "requests": ["dataset", "job_gid", "request_id"],
    "placements": ["dataset", "job_gid", "machine_gid"],
    "placement_devices": ["dataset", "job_gid", "machine_gid", "gpu_index"],
    "nodes": ["dataset", "machine_gid", "machine_id"],
    "devices": ["dataset", "device_gid", "machine_gid", "gpu_index"],
    "gpu_types": ["dataset", "gpu_label"],
    "subclusters": ["dataset", "kind", "subcluster_id"],
}


# ---------------------------------------------------------------------- helpers
def _missing(s: pd.Series) -> pd.Series:
    o = s.astype("object")
    return o.isna() | o.isin(["", "nan", "None", "<NA>", "NaN"])


def gid(dataset: str, segment: str, raw: pd.Series) -> pd.Series:
    """R1: typed, prefixed, collision-proof node id.

    A key the trace does not publish becomes NA, never "<dataset>#seg:nan" -- a
    fake node that every "unknown" row points at is worse than a hole, because it
    silently becomes the busiest node in the graph.
    """
    miss = _missing(raw)
    return ((dataset + "#" + segment + ":") + raw.astype(str)).where(~miss, pd.NA)


def dev_gid(dataset: str, local: pd.Series) -> pd.Series:
    """R1 for GPU devices.

    A device is identified by the slot it occupies on a machine, so the local
    part is "<machine>#dev<n>".  It still has to carry the dataset and segment
    prefix: machine ids are only unique inside one trace (Alibaba hashes them,
    Philly reuses short names like m192 across virtual clusters), so an
    unprefixed device id is exactly the defect the id-space rule exists to
    prevent.  Missing slot numbers stay NA (see union_devices).
    """
    miss = _missing(local)
    return (dataset + "#device:" + local.astype(str)).where(~miss, pd.NA)


def slot_gid(dataset: str, machine: pd.Series, gpu_index: pd.Series) -> pd.Series:
    """device_gid from the two parts, so a missing machine cannot leak "nan#dev3"."""
    ok = (~_missing(machine)) & pd.to_numeric(gpu_index, errors="coerce").notna()
    out = pd.Series(pd.NA, index=machine.index, dtype="object")
    out[ok] = dev_gid(dataset, machine[ok].astype(str) + "#dev"
                      + pd.to_numeric(gpu_index[ok]).astype("int64").astype(str))
    return out


def union_devices(declared: pd.DataFrame, observed: pd.DataFrame, nodes: pd.DataFrame,
                  ds: str, notes: dict, dec_prov: str, obs_prov: str) -> tuple:
    """Device inventory = every GPU slot the trace *evidences*, from either side.

    `declared` are the slots implied by the machine table, `observed` the slots a
    scheduler actually handed out.  Neither side is complete on its own: Philly's
    published machine_list is a snapshot that understates cards in use (238
    machines are logged with more GPU slots than the list gives them), while every
    inventory lists cards no job ever touched.  Dropping either side would break
    R2 (a conflict is kept and flagged, never deleted) and would break referential
    integrity from placement_devices -> devices.

    Machines that only exist on the observed side are added to `nodes` too, marked
    `observed_only:not_in_machine_table`, so the node dimension stays closed.
    """
    d = declared.copy()
    d["gpu_index"] = pd.to_numeric(d["gpu_index"], errors="coerce")
    d = d[d.gpu_index.notna()].assign(gpu_index=lambda x: x.gpu_index.astype("int64"))
    o = observed.copy()
    if len(o):
        o["gpu_index"] = pd.to_numeric(o["gpu_index"], errors="coerce")
        o = o[o.gpu_index.notna()]
        o = o.assign(gpu_index=lambda x: x.gpu_index.astype("int64"))
        o = (o[["machine_id", "gpu_index", "n_jobs_observed", "gpu_label"]]
             if "gpu_label" in o.columns else
             o[["machine_id", "gpu_index", "n_jobs_observed"]].assign(gpu_label=pd.NA))
    else:
        o = pd.DataFrame(columns=["machine_id", "gpu_index", "n_jobs_observed", "gpu_label"])

    d["_in_declared"] = True
    o = o.assign(_in_observed=True)
    m = d.merge(o, on=["machine_id", "gpu_index"], how="outer", suffixes=("_d", "_o"))
    in_d, in_o = m._in_declared.notna(), m._in_observed.notna()
    m["evidence"] = np.where(in_d & in_o, "declared+observed",
                             np.where(in_d, "declared_only", "observed_only"))
    m["gpu_label"] = m.gpu_label_d.fillna(m.gpu_label_o).fillna(UNRESOLVED)
    m["gpu_type_raw"] = m.gpu_type_raw.fillna("")
    m["n_jobs_observed"] = m.n_jobs_observed.fillna(0).astype("int64")
    m["label_provenance"] = np.where(
        m.evidence == "declared_only", dec_prov,
        np.where(m.evidence == "observed_only", obs_prov, dec_prov + "+" + obs_prov))
    m["dataset"] = ds
    m["machine_gid"] = gid(ds, "machine", m.machine_id)
    m["device_gid"] = slot_gid(ds, m["machine_id"], m["gpu_index"])
    notes["devices_slots_declared_only"] = int((m.evidence == "declared_only").sum())
    notes["devices_slots_observed_only"] = int((m.evidence == "observed_only").sum())
    notes["devices_machines_observed_beyond_machine_table"] = int(
        (~m.machine_id.isin(nodes.machine_id)).sum())

    # per-machine slot counts, so the capacity conflict is readable without a join.
    # the observed count is highest-slot+1, not distinct-slots: a log that names
    # gpu0 and gpu7 on a machine declared with 2 cards evidences 8 cards even
    # though only two slot numbers were seen.
    dec = d.groupby("machine_id").size()
    obs_n = (o.groupby("machine_id").gpu_index.max() + 1) if len(o) else pd.Series(dtype="int64")
    obs_d = o.groupby("machine_id").size() if len(o) else pd.Series(dtype="int64")
    nodes["n_slots_declared"] = nodes.machine_id.map(dec).fillna(0).astype("int64")
    nodes["n_slots_observed"] = nodes.machine_id.map(obs_n).fillna(0).astype("int64")
    nodes["n_slots_observed_distinct"] = nodes.machine_id.map(obs_d).fillna(0).astype("int64")
    nodes["slot_conflict"] = nodes.n_slots_observed > nodes.n_slots_declared
    notes["nodes_slot_conflict_machines"] = int(nodes.slot_conflict.sum())

    unknown = sorted(set(m.machine_id) - set(nodes.machine_id))
    if unknown:
        lab = m[m.machine_id.isin(unknown)].groupby("machine_id").gpu_label.first()
        extra = pd.DataFrame({
            "dataset": ds, "machine_id": unknown,
            "machine_gid": [f"{ds}#machine:{x}" for x in unknown],
            "cap_gpu": pd.NA, "cap_cpu": pd.NA, "cap_mem": pd.NA,
            "gpu_type_raw": "", "is_gpu_machine": True,
            "label_provenance": "observed_only:not_in_machine_table",
            "n_slots_declared": 0,
            "n_slots_observed": [int(obs_n.get(x, 0)) for x in unknown],
            "n_slots_observed_distinct": [int(obs_d.get(x, 0)) for x in unknown],
            "slot_conflict": True,
            "gpu_label": [lab.get(x, UNRESOLVED) for x in unknown]})
        extra = extra[[c for c in nodes.columns if c in extra.columns]]
        nodes = pd.concat([nodes, extra], ignore_index=True)
        notes["nodes_added_from_observations"] = len(unknown)

    devices = m[["dataset", "machine_id", "gpu_index", "gpu_label", "gpu_type_raw",
                 "machine_gid", "device_gid", "evidence", "n_jobs_observed",
                 "label_provenance"]].sort_values(
        ["machine_id", "gpu_index"]).reset_index(drop=True)
    return devices, nodes


def canon_gpu(raw, dataset: str) -> str:
    s = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw).strip()
    if s in ("nan", "None", "<NA>"):
        s = ""
    if dataset == "ali_v2020":
        return ALI_GPU_LABEL.get(s, s.upper() if s else NO_GPU)
    return s.upper() if s else NO_GPU


def pipe_distinct(s: pd.Series) -> str:
    v = sorted({str(x) for x in s.dropna().unique()
                if str(x) not in ("", "nan", "<NA>", NO_GPU)})
    return "|".join(v)


def mode_first(s: pd.Series):
    s = s.dropna()
    s = s[s.astype(str).str.len() > 0]
    if len(s) == 0:
        return pd.NA
    return s.value_counts().index[0]


def to_epoch(series, tz, name: str, notes: dict) -> pd.Series:
    """Native timestamp column -> float epoch seconds (NaN when unparseable)."""
    if isinstance(series, pd.Series) and pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("float64")
    s = pd.to_datetime(series, errors="coerce").astype("datetime64[ns]")
    n_missing = int(s.isna().sum())
    if n_missing:
        notes[f"{name}:unparsable_or_absent"] = notes.get(f"{name}:unparsable_or_absent", 0) + n_missing
    if tz:
        s = s.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    # NaT would surface as the int64 minimum, so mask it out explicitly
    ns = pd.Series(s.astype("int64"), index=s.index).where(s.notna(), np.nan)
    return ns.astype("float64") / 1e9


def add_time_cols(df: pd.DataFrame, epoch_col: str, short: str, t0: float,
                  tz, notes: dict) -> pd.DataFrame:
    """R4: seconds-since-t0, daily snapshot index, time-of-day, UTC string."""
    e = df[epoch_col]
    df[f"t_{short}_s"] = e - t0
    df[f"snap_{short}"] = np.floor(df[f"t_{short}_s"] / DAY)
    df[f"tod_{short}"] = np.mod(df[f"t_{short}_s"], DAY)
    df[f"t_{short}_utc"] = (pd.to_datetime(pd.to_numeric(e, errors="coerce").round(0),
                                           unit="s", utc=True)
                            .dt.strftime("%Y-%m-%dT%H:%M:%SZ") if tz else pd.NA)
    notes[f"time_{short}:missing"] = int(df[f"t_{short}_s"].isna().sum())
    return df


def seconds_since(df: pd.DataFrame, epoch_col: str, t0: float) -> pd.DataFrame:
    df = df.copy()
    df["t_start_s"] = df[epoch_col] - t0
    df["snap_start"] = np.floor(df["t_start_s"] / DAY)
    return df


def recurrence(df: pd.DataFrame, key: str) -> dict:
    if key not in df.columns or "snap_submit" not in df.columns:
        return {}
    sub = df[[key, "snap_submit"]].dropna()
    if len(sub) == 0:
        return {}
    per = sub.groupby(key, observed=True)["snap_submit"].nunique()
    cnt = sub.groupby(key, observed=True).size()
    return {"jobs": int(len(sub)), "entities": int(len(per)),
            "jobs_per_entity_mean": round(float(cnt.mean()), 2),
            "jobs_per_entity_p90": float(cnt.quantile(.9)),
            "entities_recurring": int((per >= 2).sum()),
            "pct_entities_recurring": round(100 * float((per >= 2).mean()), 2),
            "max_snapshots": int(per.max())}


def coverage(df: pd.DataFrame) -> dict:
    return {c: {"dtype": str(df[c].dtype),
                "non_null_pct": round(100 * float(df[c].notna().mean()), 2)}
            for c in df.columns}


class Bundle:
    """Collects the released tables of one dataset and writes them out."""

    def __init__(self, dataset: str, out: str, no_devices: bool = False):
        self.dataset, self.out, self.no_devices = dataset, out, no_devices
        self.tables, self.absent = {}, {}
        os.makedirs(out, exist_ok=True)

    def add(self, name, df):
        self.tables[name] = df

    def empty(self, name, why):
        self.tables[name] = pd.DataFrame(columns=EMPTY_SCHEMA[name])
        self.absent[name] = why

    def write(self, notes, rec, bad):
        for name in ("jobs", "requests", "placements", "placement_devices",
                     "nodes", "devices", "gpu_types", "subclusters"):
            if name not in self.tables:
                self.empty(name, "not produced by this dataset's cleaner")
        man = {"dataset": self.dataset,
               "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "id_space": "gid = '<dataset>#<segment>:<raw id>' (typed, collision-checked)",
               "timezone_assumption": DATASET_TZ[self.dataset] or
               "none -- trace publishes de-identified dates; *_utc columns are NULL",
               "deriving_and_dropping": notes,
               "tables": {}, "absent_tables": self.absent,
               "recurrence_head_candidates": rec,
               "coverage_jobs": coverage(self.tables["jobs"]),
               "invariant_violations": bad}
        for name, df in sorted(self.tables.items()):
            path = os.path.join(self.out, f"{name}.parquet")
            df.to_parquet(path, index=False)
            man["tables"][name] = {"rows": int(len(df)), "cols": len(df.columns),
                                   "size_mb": round(os.path.getsize(path) / 1e6, 1),
                                   "columns": list(df.columns)}
            print(f"  wrote {name:<20} rows={len(df):>11,}  "
                  f"{os.path.getsize(path)/1e6:9.1f} MB")
        with open(os.path.join(self.out, "manifest.json"), "w") as f:
            json.dump(man, f, indent=2, default=str)
        with open(os.path.join(self.out, "profile.md"), "w") as f:
            f.write(self.profile(notes, rec))
        return man

    def profile(self, notes, rec) -> str:
        L = [f"# Cleaned trace -- {self.dataset}", ""]
        for name, df in sorted(self.tables.items()):
            L.append(f"## {name}.parquet  ({len(df):,} rows x {len(df.columns)} cols)")
            if name in self.absent:
                L.append(f"_empty: {self.absent[name]}_")
            elif len(df) == 0:
                L.append("_empty_")
            else:
                L.append("")
                L.append("| column | dtype | non-null % |")
                L.append("|---|---|---|")
                for c in df.columns:
                    L.append(f"| `{c}` | {df[c].dtype} | {100*float(df[c].notna().mean()):.2f} |")
            L.append("")
        L.append("## head-candidate recurrence (the experiment picks the entity)")
        for k, v in rec.items():
            if v:
                L.append(f"- `{k}`: entities={v['entities']:,} jobs={v['jobs']:,} "
                         f"mean/entity={v['jobs_per_entity_mean']} "
                         f"recurring(>=2 snapshots)={v['pct_entities_recurring']}% "
                         f"max snapshots={v['max_snapshots']}")
        L.append("")
        L.append("## derive / drop decisions recorded in code")
        for k, v in notes.items():
            L.append(f"- `{k}`: {v}")
        return "\n".join(L) + "\n"


def add_task_ordinal(j: pd.DataFrame, ds: str, notes: dict) -> pd.DataFrame:
    """Make jobs.parquet row-addressable without deleting any published record.

    A repeated native job id is NOT a duplicate to be dropped.  In Eagle it is
    a Slurm ARRAY JOB: one id, many task records that share submit/start but
    have their own end_time.  Dropping them would delete real work and bias
    every duration statistic.  So the primary key of jobs.parquet is
    (job_gid, task_ordinal), and `n_array_tasks` says how big the group is.

    Rows whose native id is missing are kept, given a per-row synthetic token,
    and flagged in `native_id_missing` -- never merged into one fake entity.
    """
    j = j.reset_index(drop=True)
    ids = j["job_id"].astype("string")
    miss = ids.isna()
    notes["jobs_native_id_missing"] = int(miss.sum())
    j["native_id_missing"] = miss.to_numpy()
    if int(miss.sum()):
        tokens = pd.Series([f"__missing_{i}__" for i in range(len(j))], index=j.index)
        ids = ids.mask(miss, tokens)
    counts = ids.value_counts()
    j["task_ordinal"] = ids.groupby(ids).cumcount().to_numpy()
    j["n_array_tasks"] = ids.map(counts).astype("int64").to_numpy()
    j["is_array_task"] = j["n_array_tasks"] > 1
    notes["jobs_records_sharing_native_id"] = int((j["n_array_tasks"] > 1).sum())
    notes["jobs_native_ids_shared"] = int((counts > 1).sum())
    j["job_id"] = ids.to_numpy()
    j["job_gid"] = gid(ds, "job", j["job_id"])
    return j


def check_id_space(jobs, nodes, devices):
    segs = {"job": set(jobs.get("job_gid", pd.Series(dtype=str)).dropna()),
            "machine": set(nodes.get("machine_gid", pd.Series(dtype=str)).dropna()),
            "device": set(devices.get("device_gid", pd.Series(dtype=str)).dropna())}
    for c in ("user_gid", "account_gid"):
        if c in jobs.columns:
            segs[c[:-4]] = set(jobs[c].dropna())
    bad = []
    keys = sorted(segs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            inter = segs[a] & segs[b]
            if inter:
                bad.append(f"{a}/{b}: {len(inter)} e.g. {sorted(inter)[:2]}")
    print("  id-space: " + " ".join(f"{k}={len(v):,}" for k, v in sorted(segs.items()))
          + f" collisions={len(bad)}")
    return bad


# ==================================================================== ali v2020
ALI_COLS = {
    "job": ["job_name", "inst_id", "user", "status", "start_time", "end_time"],
    "task": ["job_name", "task_name", "inst_num", "status", "start_time", "end_time",
             "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"],
    "instance": ["job_name", "task_name", "inst_name", "worker_name", "inst_id",
                 "status", "start_time", "end_time", "machine"],
    "group_tag": ["inst_id", "user", "gpu_type_spec", "group", "workload"],
    "machine_spec": ["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"],
    "sensor": ["job_name", "task_name", "worker_name", "inst_id", "machine", "gpu_name"],
}


def read_headerless(path, cols, usecols=None):
    """The released v2020 CSVs carry no header; column names come from the README."""
    idx = usecols if usecols is not None else list(range(len(cols)))
    df = pd.read_csv(path, header=None, names=cols, usecols=idx, dtype="string")
    return df[[cols[i] for i in idx]]


def clean_ali(raw: str, bundle: Bundle, notes: dict):
    ds, tz = "ali_v2020", DATASET_TZ["ali_v2020"]
    spec = read_headerless(os.path.join(raw, "pai_machine_spec.csv"), ALI_COLS["machine_spec"])
    job = read_headerless(os.path.join(raw, "pai_job_table.csv"), ALI_COLS["job"])
    task = read_headerless(os.path.join(raw, "pai_task_table.csv"), ALI_COLS["task"])
    gt = read_headerless(os.path.join(raw, "pai_group_tag_table.csv"), ALI_COLS["group_tag"])
    ins = read_headerless(os.path.join(raw, "pai_instance_table.csv"), ALI_COLS["instance"],
                          usecols=[0, 1, 3, 4, 5, 6, 7, 8])
    notes["raw_rows"] = {"job": len(job), "task": len(task), "instance": len(ins),
                         "group_tag": len(gt), "machine_spec": len(spec)}
    print("  raw rows: " + " ".join(f"{k}={v:,}" for k, v in notes["raw_rows"].items()))

    # ---- nodes / devices -----------------------------------------------------
    for c in ("cap_cpu", "cap_mem", "cap_gpu"):
        spec[c] = pd.to_numeric(spec[c], errors="coerce")
    spec["gpu_type_raw"] = spec["gpu_type"].fillna("")
    spec["gpu_label"] = spec["gpu_type_raw"].map(lambda x: canon_gpu(x, ds))
    spec["cap_gpu"] = spec["cap_gpu"].fillna(0).astype(int)
    spec["is_gpu_machine"] = spec["cap_gpu"] > 0
    spec["machine_gid"] = gid(ds, "machine", spec["machine"])
    nodes = spec.rename(columns={"machine": "machine_id"}).assign(
        dataset=ds, label_provenance="joined:machine_spec")[
        ["dataset", "machine_id", "machine_gid", "gpu_type_raw", "gpu_label",
         "cap_cpu", "cap_mem", "cap_gpu", "is_gpu_machine", "label_provenance"]]

    gpu_m = spec[spec.is_gpu_machine]
    devices = pd.DataFrame([("ali_v2020", r.machine, i, r.gpu_label, r.gpu_type_raw)
                            for r in gpu_m.itertuples(index=False)
                            for i in range(int(r.cap_gpu))],
                           columns=["dataset", "machine_id", "gpu_index",
                                    "gpu_label", "gpu_type_raw"])
    devices["machine_gid"] = gid(ds, "machine", devices["machine_id"])
    devices["device_gid"] = slot_gid(ds, devices["machine_id"], devices["gpu_index"])
    devices["label_provenance"] = "joined:machine_spec"

    # ---- placements (instance -> machine -> gpu_type) ------------------------
    ins["worker_name"] = ins["worker_name"].fillna("")
    ins["t_start_epoch"] = pd.to_numeric(ins["start_time"], errors="coerce")
    ins["t_end_epoch"] = pd.to_numeric(ins["end_time"], errors="coerce")
    ins = ins.merge(spec[["machine", "gpu_label", "is_gpu_machine"]], on="machine", how="left")
    notes["placements_machine_unjoined"] = int(ins["gpu_label"].isna().sum())
    ins["gpu_label"] = ins["gpu_label"].fillna(UNRESOLVED)

    placements = (ins.groupby(["job_name", "task_name", "worker_name", "machine"],
                              dropna=False, observed=True)
                  .agg(inst_id=("inst_id", "first"), gpu_label=("gpu_label", "first"),
                       is_gpu_placement=("is_gpu_machine", "first"),
                       inst_status=("status", "first"),
                       t_start_epoch=("t_start_epoch", "min"),
                       t_end_epoch=("t_end_epoch", "max"),
                       n_instance_rows=("job_name", "size"))
                  .reset_index())
    placements["dataset"] = ds
    # an instance whose machine is missing from machine_spec has no evidence about
    # the card at all -- keep the row (the placement happened) and mark it
    # non-GPU/unknown rather than letting NA poison every boolean mask downstream
    placements["is_gpu_placement"] = (
        placements["is_gpu_placement"].astype("object").fillna(False).astype(bool))
    notes["placements_without_machine_in_spec"] = int(
        (placements["machine"].isna() | (placements["machine"].astype(str) == "")).sum())
    placements["job_gid"] = gid(ds, "job", placements["job_name"])
    placements["machine_gid"] = gid(ds, "machine", placements["machine"])
    placements["alloc_unit"] = "instance"
    placements["label_provenance"] = "joined:instance->machine_spec"

    # ---- device-grain allocations from the sensor table ----------------------
    placement_devices = pd.DataFrame(
        columns=["dataset", "job_name", "task_name", "worker_name", "inst_id", "machine",
                 "machine_gid", "gpu_index", "gpu_label", "device_gid", "label_provenance"])
    if not bundle.no_devices:
        # the sensor table is 1.06 GB and its distinct projection is the slow step
        # of this dataset (~15 min); cache it so a downstream fix does not pay again
        cache = os.path.join(raw, "_cache_sensor_distinct.parquet")
        if os.path.exists(cache):
            sen = pd.read_parquet(cache)
            notes["sensor_distinct_cache"] = "reused " + os.path.basename(cache)
        else:
            parts = []
            for chunk in pd.read_csv(os.path.join(raw, "pai_sensor_table.csv"), header=None,
                                     names=ALI_COLS["sensor"], usecols=[0, 1, 2, 3, 4, 5],
                                     dtype="string", chunksize=2_000_000):
                parts.append(chunk.drop_duplicates())
            sen = pd.concat(parts, ignore_index=True).drop_duplicates()
            sen.to_parquet(cache, index=False)
            notes["sensor_distinct_cache"] = "built " + os.path.basename(cache)
        notes["sensor_distinct_rows"] = len(sen)
        sen["gpu_index"] = sen["gpu_name"].str.extract(r"nvidia(\d+)", expand=False).astype("float64")
        notes["device_allocations_gpu_index_unparsed"] = int(sen["gpu_index"].isna().sum())
        sen = sen.merge(spec[["machine", "gpu_label"]], on="machine", how="left")
        placement_devices = (sen.groupby(["job_name", "task_name", "worker_name",
                                          "machine", "gpu_index", "gpu_label"],
                                         dropna=False, observed=True)
                             .agg(inst_id=("inst_id", "first"), n_events=("job_name", "size"))
                             .reset_index())
        placement_devices["dataset"] = ds
        placement_devices["job_gid"] = gid(ds, "job", placement_devices["job_name"])
        placement_devices["machine_gid"] = gid(ds, "machine", placement_devices["machine"])
        placement_devices["device_gid"] = slot_gid(
            ds, placement_devices["machine"], placement_devices["gpu_index"])
        placement_devices["label_provenance"] = "observed:sensor_gpu_name"

    # inventory = declared slots UNION slots the sensor table actually saw (R2)
    obs_dev = (placement_devices.groupby(["machine", "gpu_index"], dropna=False, observed=True)
               .agg(n_jobs_observed=("job_name", "nunique"),
                    gpu_label=("gpu_label", "first")).reset_index()
               .rename(columns={"machine": "machine_id"}))
    devices, nodes = union_devices(devices, obs_dev, nodes, ds, notes,
                                   "joined:machine_spec", "observed:sensor_gpu_name")

    # ---- requests (task grain: what the job asked for) -----------------------
    for c in ("inst_num", "start_time", "end_time", "plan_cpu", "plan_mem", "plan_gpu"):
        task[c] = pd.to_numeric(task[c], errors="coerce")
    task["gpu_type_raw"] = task["gpu_type"].fillna("")
    req = (task.groupby(["job_name", "task_name"], dropna=False, observed=True)
           .agg(inst_num=("inst_num", "sum"),
                req_cpus=("plan_cpu", lambda s: float(np.nansum(s)) / 100.0),
                req_mem_gib=("plan_mem", lambda s: float(np.nansum(s))),
                n_gpus_req=("plan_gpu", lambda s: float(np.nansum(s)) / 100.0),
                gpu_type_req_raw=("gpu_type_raw", pipe_distinct),
                req_status=("status", "first"),
                t_start_epoch=("start_time", "min"),
                t_end_epoch=("end_time", "max"))
           .reset_index())
    req["dataset"] = ds
    req["request_id"] = req["job_name"] + "/" + req["task_name"]
    req["job_gid"] = gid(ds, "job", req["job_name"])
    req["label_provenance"] = "declared:task_table.gpu_type"

    # ---- jobs ----------------------------------------------------------------
    notes["jobs_duplicate_job_name_rows"] = int(job.job_name.duplicated().sum())
    j = job.drop_duplicates("job_name").copy()
    gtd = gt.drop_duplicates("inst_id")
    j = j.merge(gtd[["inst_id", "gpu_type_spec", "group", "workload"]], on="inst_id", how="left")
    j["account_id"] = j["group"]
    j["app_name"] = j["workload"]                     # README: only ~9% carry a workload tag
    j["queue_id"] = pd.NA
    j["qos"] = pd.NA
    j["gpu_type_requested_raw"] = j["gpu_type_spec"]
    has_spec = j["gpu_type_spec"].notna() & (j["gpu_type_spec"].str.len() > 0)
    j["gpu_label_req"] = pd.NA
    j.loc[has_spec, "gpu_label_req"] = j.loc[has_spec, "gpu_type_spec"].map(
        lambda x: canon_gpu(x, ds))

    roll_task = (task.groupby("job_name", dropna=False, observed=True)
                 .agg(n_tasks=("task_name", "nunique"),
                      req_cpus=("plan_cpu", lambda s: float(np.nansum(s)) / 100.0),
                      req_mem_gib=("plan_mem", lambda s: float(np.nansum(s))),
                      n_gpus_req=("plan_gpu", lambda s: float(np.nansum(s)) / 100.0),
                      framework=("task_name", mode_first),
                      t_start_epoch=("start_time", "min"))
                 .reset_index())
    roll_inst = (placements.groupby("job_name", dropna=False, observed=True)
                 .agg(n_alloc_units=("worker_name", "nunique"),
                      n_machines=("machine", "nunique"),
                      gpu_label_obs=("gpu_label", pipe_distinct),
                      n_gpu_machines=("is_gpu_placement",
                                      lambda s: int(pd.Series(s).astype("object")
                                                    .fillna(False).astype(bool).sum())))
                 .reset_index())
    j = j.merge(roll_task, on="job_name", how="left").merge(roll_inst, on="job_name", how="left")
    if len(placement_devices):
        rd = (placement_devices.groupby("job_name", observed=True)["device_gid"]
              .nunique().rename("n_gpus_obs").reset_index())
        j = j.merge(rd, on="job_name", how="left")
    else:
        j["n_gpus_obs"] = np.nan
    j["gpu_label_final"] = j["gpu_label_obs"].where(
        j["gpu_label_obs"].notna() & (j["gpu_label_obs"] != ""), j["gpu_label_req"])
    j["label_provenance"] = np.where(
        j["gpu_label_obs"].notna() & (j["gpu_label_obs"] != ""), "joined:instance->machine_spec",
        np.where(j["gpu_label_req"].notna(), "declared:group_tag_gpu_type_spec", "unknown"))
    j["is_gpu_job"] = (j["n_gpus_req"].fillna(0) > 0) | (j["n_gpu_machines"].fillna(0) > 0)
    j["state_raw"] = j["status"]
    j["state_norm"] = j["state_raw"].str.lower().map(STATE_MAP).fillna("other")
    j["job_id"] = j["job_name"]
    j["job_gid"] = gid(ds, "job", j["job_name"])
    j["user_id"] = j["user"]
    j["user_gid"] = gid(ds, "user", j["user"])
    j["account_gid"] = gid(ds, "account", j["account_id"].fillna(j["inst_id"]))
    j["request_id"] = j["inst_id"]
    j["t_submit_epoch"] = pd.to_numeric(j["start_time"], errors="coerce")
    j["t_end_epoch"] = pd.to_numeric(j["end_time"], errors="coerce")
    j["dataset"] = ds
    j["n_gpus_obs"] = pd.to_numeric(j["n_gpus_obs"], errors="coerce")

    # native timestamps are seconds offset by an unknown constant (see README)
    t0 = float(j["t_submit_epoch"].min())
    notes["dataset_t0_native_seconds"] = t0
    notes["dataset_absolute_dates"] = ("unknown offset -- README states time-of-day and "
                                       "day-of-week are preserved, dates/years are fake")
    for col, short in (("t_submit_epoch", "submit"), ("t_start_epoch", "start"),
                       ("t_end_epoch", "end")):
        j = add_time_cols(j, col, short, t0, tz, notes)
    j["wait_s"] = j["t_start_s"] - j["t_submit_s"]
    j["run_s"] = j["t_end_s"] - j["t_start_s"]
    placements = seconds_since(placements, "t_start_epoch", t0)
    placements["t_end_s"] = placements["t_end_epoch"] - t0
    req = seconds_since(req, "t_start_epoch", t0)
    req["t_end_s"] = req["t_end_epoch"] - t0

    # ---- gpu_types -----------------------------------------------------------
    used = (placements[placements.is_gpu_placement]
            .groupby("gpu_label", observed=True)
            .agg(n_placements=("machine", "size"), n_machines_used=("machine", "nunique")))
    gt2 = (devices.groupby("gpu_label", observed=True)
           .agg(n_devices_in_inventory=("device_gid", "size"),
                n_machines_in_inventory=("machine_id", "nunique"),
                raw_spellings=("gpu_type_raw", pipe_distinct)).reset_index()
           .merge(used, on="gpu_label", how="left"))
    gt2["dataset"] = ds
    # a model name is "published" only if the spec join produced a raw spelling;
    # the UNRESOLVED bucket is the fallback, saying True there would let
    # `df[df.model_name_published]` treat "unknown" as a GPU generation
    spell = gt2["raw_spellings"].astype("string").fillna("")
    gt2["model_name_published"] = spell.str.strip().ne("")
    gt2["label_provenance"] = "trace-column:machine_spec.gpu_type"
    gt2["mem_gib_stated"] = gt2["gpu_label"].str.extract(r"(\d+)G", expand=False).astype("float64")
    gt2["note"] = gt2["gpu_label"].map(
        lambda x: "README: MISC groups several older generations (K40m/K80/M60)"
        if x == "NVIDIA-LEGACY-MISC" else
        ("fallback bucket: these machines have no gpu_type in pai_machine_spec, "
         "so no model name exists" if x == "UNRESOLVED" else ""))
    gt2 = gt2[["gpu_label", "n_devices_in_inventory", "n_machines_in_inventory",
               "raw_spellings", "n_placements", "n_machines_used", "dataset",
               "mem_gib_stated", "model_name_published", "label_provenance", "note"]]

    sub = (gt2[["gpu_label", "n_devices_in_inventory", "n_placements", "n_machines_used"]]
           .rename(columns={"gpu_label": "subcluster_id"}).assign(dataset=ds, kind="gpu_label"))

    j = add_task_ordinal(j, ds, notes)
    jobs = j[["dataset", "job_id", "job_gid", "task_ordinal", "n_array_tasks",
              "is_array_task", "native_id_missing",
              "request_id", "user_id", "user_gid",
              "account_id", "account_gid", "app_name", "queue_id", "qos", "framework",
              "state_raw", "state_norm", "n_tasks", "n_alloc_units", "n_machines",
              "req_cpus", "req_mem_gib", "n_gpus_req", "n_gpus_obs", "n_gpu_machines",
              "gpu_type_requested_raw", "gpu_label_req", "gpu_label_obs", "gpu_label_final",
              "label_provenance", "is_gpu_job",
              "t_submit_s", "t_submit_utc", "snap_submit", "tod_submit",
              "t_start_s", "t_start_utc", "snap_start", "tod_start",
              "t_end_s", "t_end_utc", "snap_end", "tod_end", "wait_s", "run_s"]].copy()

    known_jobs, known_machines = set(jobs.job_id), set(nodes.machine_id)
    for df, lbl in ((placements, "placements"), (req, "requests"),
                    (placement_devices, "placement_devices")):
        resolve_fk(df, "job_name", "job_gid", known_jobs, notes,
                   f"{lbl}_job_gid_unresolved")
    for df, lbl in ((placements, "placements"), (placement_devices, "placement_devices")):
        resolve_fk(df, "machine", "machine_gid", known_machines, notes,
                   f"{lbl}_machine_gid_unresolved")

    bundle.add("jobs", jobs)
    bundle.add("requests", req[["dataset", "job_name", "job_gid", "request_id", "task_name",
                                "inst_num", "req_cpus", "req_mem_gib", "n_gpus_req",
                                "gpu_type_req_raw", "req_status", "label_provenance",
                                "t_start_s", "t_end_s", "snap_start"]])
    bundle.add("placements", placements[["dataset", "job_name", "job_gid", "inst_id",
                                         "task_name", "worker_name", "machine", "machine_gid",
                                         "gpu_label", "is_gpu_placement", "alloc_unit",
                                         "label_provenance", "inst_status", "n_instance_rows",
                                         "t_start_s", "t_end_s", "snap_start"]])
    bundle.add("placement_devices", placement_devices)
    bundle.add("nodes", nodes)
    bundle.add("devices", devices)
    bundle.add("gpu_types", gt2)
    bundle.add("subclusters", sub)
    return jobs, nodes, devices


# ======================================================================= philly
def clean_philly(raw: str, bundle: Bundle, notes: dict):
    ds, tz = "philly", DATASET_TZ["philly"]
    lines = open(os.path.join(raw, "cluster_machine_list")).read().splitlines()
    cap, tier = {}, {}
    for line in lines[1:]:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        mid, n, mem = parts[0], parts[1], parts[2]
        cap[mid] = int(re.sub(r"[^0-9]", "", n) or 0)
        tier[mid] = PHILLY_TIER.format(n=cap[mid], mem=re.sub(r"[^0-9]", "", mem) or "NA")
    nodes = pd.DataFrame([{"dataset": ds, "machine_id": m,
                           "machine_gid": f"{ds}#machine:{m}",
                           "cap_gpu": cap[m], "cap_cpu": pd.NA, "cap_mem": pd.NA,
                           "gpu_type_raw": "", "gpu_label": tier[m],
                           "is_gpu_machine": cap[m] > 0,
                           "label_provenance": "joined:machine_list"}
                          for m in cap])
    devices = pd.DataFrame([(ds, m, i, tier[m], "")
                            for m in cap for i in range(cap[m])],
                           columns=["dataset", "machine_id", "gpu_index",
                                    "gpu_label", "gpu_type_raw"])
    devices["machine_gid"] = gid(ds, "machine", devices["machine_id"])
    devices["device_gid"] = slot_gid(ds, devices["machine_id"], devices["gpu_index"])
    devices["label_provenance"] = "joined:machine_list"

    recs = json.load(open(os.path.join(raw, "cluster_job_log")))
    notes["raw_job_records"] = len(recs)
    prows, jrows = [], []
    for r in recs:
        jid, vc, user = r.get("jobid"), r.get("vc"), r.get("user")
        n_units = g_total = 0
        machines, starts, ends = set(), [], []
        for ai, a in enumerate(r.get("attempts") or []):
            a_st, a_en = a.get("start_time"), a.get("end_time")
            for d in (a.get("detail") or []):
                m, g = d.get("ip"), d.get("gpus") or []
                prows.append({"job_id": jid, "attempt": ai, "machine": m, "n_gpus": len(g),
                              "gpu_ids": "|".join(g), "user": user, "account": vc,
                              "t_start_native": a_st, "t_end_native": a_en})
                n_units += 1
                g_total += len(g)
                if m:
                    machines.add(m)
                if a_st:
                    starts.append(a_st)
                if a_en:
                    ends.append(a_en)
        jrows.append({"job_id": jid, "user": user, "account": vc, "state_raw": r.get("status"),
                      "t_submit_native": r.get("submitted_time"),
                      "n_attempts": len(r.get("attempts") or []),
                      "n_alloc_units": n_units, "n_machines": len(machines),
                      "n_gpus_obs": g_total,
                      "gpu_label_obs": ("|".join(sorted({tier.get(m, UNRESOLVED) for m in machines}))
                                        if machines else pd.NA),
                      "t_start_native": (min(starts) if starts else pd.NA),
                      "t_end_native": (max(ends) if ends else pd.NA)})
    p = pd.DataFrame(prows)
    j = pd.DataFrame(jrows)
    notes["raw_placement_rows"] = len(p)

    p["dataset"] = ds
    p["job_gid"] = gid(ds, "job", p["job_id"])
    p["machine_gid"] = gid(ds, "machine", p["machine"])
    p["gpu_label"] = p["machine"].map(tier).fillna(UNRESOLVED)
    mc = p["machine"].map(cap)
    p["cap_conflict"] = mc.notna() & (p["n_gpus"] > mc.fillna(0))
    p["t_start_epoch"] = to_epoch(p["t_start_native"], tz, "placements_t_start", notes)
    p["t_end_epoch"] = to_epoch(p["t_end_native"], tz, "placements_t_end", notes)
    notes["placements_negative_run_s"] = int(((p["t_end_epoch"] - p["t_start_epoch"]) < 0).sum())
    notes["placements_gpu_above_machine_capacity"] = int(p["cap_conflict"].sum())

    placements = (p.groupby(["job_id", "attempt", "machine", "gpu_ids"],
                            dropna=False, observed=True)
                  .agg(n_gpus=("n_gpus", "max"),
                       t_start_epoch=("t_start_epoch", "min"),
                       t_end_epoch=("t_end_epoch", "max"),
                       cap_conflict=("cap_conflict", "max"))
                  .reset_index())
    placements["dataset"] = ds
    placements["job_gid"] = gid(ds, "job", placements["job_id"])
    placements["machine_gid"] = gid(ds, "machine", placements["machine"])
    placements["gpu_label"] = placements["machine"].map(tier).fillna(UNRESOLVED)
    placements["alloc_unit"] = "attempt-machine"
    placements["label_provenance"] = "observed:job_log_gpus_list"

    if not bundle.no_devices and len(p):
        ex = p[(p.n_gpus > 0) & p.gpu_ids.notna() & (p.gpu_ids != "")].copy()
        ex["gpu_id"] = ex["gpu_ids"].str.split("|")
        ex = ex.explode("gpu_id")
        ex["gpu_index"] = ex["gpu_id"].str.extract(r"gpu(\d+)", expand=False).astype("float64")
        notes["device_allocations_gpu_index_unparsed"] = int(ex["gpu_index"].isna().sum())
        placement_devices = (ex.groupby(["job_id", "attempt", "machine", "gpu_index"],
                                        dropna=False, observed=True)
                             .agg(n_rows=("gpu_id", "size")).reset_index())
        placement_devices["dataset"] = ds
        placement_devices["job_gid"] = gid(ds, "job", placement_devices["job_id"])
        placement_devices["machine_gid"] = gid(ds, "machine", placement_devices["machine"])
        placement_devices["gpu_label"] = placement_devices["machine"].map(tier).fillna(UNRESOLVED)
        placement_devices["device_gid"] = slot_gid(
            ds, placement_devices["machine"], placement_devices["gpu_index"])
        placement_devices["label_provenance"] = "observed:job_log_gpus_list"
    else:
        placement_devices = pd.DataFrame(columns=["dataset", "job_id", "job_gid", "machine",
                                                  "machine_gid", "gpu_index", "gpu_label",
                                                  "device_gid", "label_provenance"])

    # inventory = declared slots UNION slots the log actually handed out (R2)
    obs_dev = (placement_devices.groupby(["machine", "gpu_index"], dropna=False, observed=True)
               .agg(n_jobs_observed=("job_id", "nunique"),
                    gpu_label=("gpu_label", "first")).reset_index()
               .rename(columns={"machine": "machine_id"}))
    devices, nodes = union_devices(devices, obs_dev, nodes, ds, notes,
                                   "joined:machine_list", "observed:job_log_gpus_list")

    j["dataset"] = ds
    j["job_gid"] = gid(ds, "job", j["job_id"])
    j["user_gid"] = gid(ds, "user", j["user"])
    j["account_gid"] = gid(ds, "account", j["account"])
    j["app_name"] = pd.NA                    # Philly publishes no application/algorithm name
    j["queue_id"] = j["account"]             # vc == virtual cluster == scheduling domain
    j["qos"] = pd.NA
    j["n_gpus_req"] = pd.NA                  # no request columns are published
    j["req_cpus"] = pd.NA
    j["req_mem_gib"] = pd.NA
    j["gpu_label_final"] = j["gpu_label_obs"]
    j["label_provenance"] = np.where(j["gpu_label_obs"].notna(), "observed:job_log", "unknown")
    j["is_gpu_job"] = j["gpu_label_obs"].notna() & (j["n_gpus_obs"] > 0)
    j["state_norm"] = j["state_raw"].astype("string").str.lower().map(STATE_MAP).fillna("other")

    t0 = float(to_epoch(j["t_submit_native"], tz, "jobs_t_submit", notes).min())
    notes["dataset_t0_utc"] = dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat()
    for col, short in (("t_submit_native", "submit"), ("t_start_native", "start"),
                       ("t_end_native", "end")):
        j[f"e_{short}"] = to_epoch(j[col], tz, f"jobs_{short}", notes)
        j = add_time_cols(j, f"e_{short}", short, t0, tz, notes)
    j["wait_s"] = j["t_start_s"] - j["t_submit_s"]
    j["run_s"] = j["t_end_s"] - j["t_start_s"]
    placements = seconds_since(placements, "t_start_epoch", t0)
    placements["t_end_s"] = placements["t_end_epoch"] - t0

    gt = (devices.groupby("gpu_label", observed=True)
          .agg(n_devices_in_inventory=("device_gid", "size"),
               n_machines_in_inventory=("machine_id", "nunique"),
               raw_spellings=("gpu_type_raw", pipe_distinct)).reset_index())
    used = (placements[placements.n_gpus > 0].groupby("gpu_label", observed=True)
            .agg(n_placements=("machine", "size"), n_machines_used=("machine", "nunique")))
    gt = gt.merge(used, on="gpu_label", how="left")
    gt["dataset"] = ds
    gt["mem_gib_stated"] = gt["gpu_label"].str.extract(r"x(\d+)", expand=False).astype("float64")
    gt["model_name_published"] = False
    gt["label_provenance"] = "derived:machine_list_capacity_tier"
    gt["note"] = ("machine_list publishes (number of GPUs, single GPU mem) only; no card "
                  "model name -> the label is a capacity tier, not a type")

    sub = (j.groupby("account", observed=True)
           .agg(n_jobs=("job_id", "size"), n_gpu_jobs=("is_gpu_job", "sum"),
                median_machines=("n_machines", "median"),
                median_gpus=("n_gpus_obs", "median"),
                t_first=("t_submit_s", "min"), t_last=("t_submit_s", "max")).reset_index())
    sub["dataset"] = ds
    sub["kind"] = "virtual_cluster"
    sub = sub.rename(columns={"account": "subcluster_id"})
    sub["gpu_share"] = sub.n_gpu_jobs / sub.n_jobs

    j = add_task_ordinal(j, ds, notes)
    jobs = j[["dataset", "job_id", "job_gid", "task_ordinal", "n_array_tasks",
              "is_array_task", "native_id_missing",
              "user", "user_gid", "account", "account_gid",
              "app_name", "queue_id", "qos", "state_raw", "state_norm",
              "n_attempts", "n_alloc_units", "n_machines", "n_gpus_req", "n_gpus_obs",
              "req_cpus", "req_mem_gib", "gpu_label_obs", "gpu_label_final",
              "label_provenance", "is_gpu_job",
              "t_submit_s", "t_submit_utc", "snap_submit", "tod_submit",
              "t_start_s", "t_start_utc", "snap_start", "tod_start",
              "t_end_s", "t_end_utc", "snap_end", "tod_end", "wait_s", "run_s"]].copy()
    # the same two entities the other traces publish, under the shared column name:
    # Philly calls them `user` and `vc`, and a vc here *is* the allocation account
    # (a virtual cluster), so this is a translation, not a claim
    jobs = jobs.rename(columns={"user": "user_id", "account": "account_id"})

    known_jobs, known_machines = set(jobs.job_id), set(nodes.machine_id)
    for df, lbl in ((placements, "placements"), (placement_devices, "placement_devices")):
        resolve_fk(df, "job_id", "job_gid", known_jobs, notes, f"{lbl}_job_gid_unresolved")
        resolve_fk(df, "machine", "machine_gid", known_machines, notes,
                   f"{lbl}_machine_gid_unresolved")

    bundle.add("jobs", jobs)
    bundle.empty("requests", "Philly publishes no CPU/memory/GPU request columns; only "
                             "(machine, gpu list) allocations are logged")
    bundle.add("placements", placements[["dataset", "job_id", "job_gid", "attempt", "machine",
                                         "machine_gid", "gpu_label", "n_gpus", "gpu_ids",
                                         "cap_conflict", "alloc_unit", "label_provenance",
                                         "t_start_s", "t_end_s", "snap_start"]])
    bundle.add("placement_devices", placement_devices)
    bundle.add("nodes", nodes)
    bundle.add("devices", devices)
    bundle.add("gpu_types", gt)
    bundle.add("subclusters", sub[["dataset", "kind", "subcluster_id", "n_jobs", "n_gpu_jobs",
                                   "gpu_share", "median_machines", "median_gpus",
                                   "t_first", "t_last"]])
    return jobs, nodes, devices


# ======================================================================== eagle
EAGLE_LOW_CARD = ["user", "account", "partition", "qos", "state", "name",
                  "work_dir", "submit_line"]


def clean_eagle(raw: str, bundle: Bundle, notes: dict):
    import pyarrow as pa
    import pyarrow.parquet as pq
    ds, tz = "eagle", DATASET_TZ["eagle"]
    path = os.path.join(raw, "eagle_data.parquet")
    cols = (["job_id"] + EAGLE_LOW_CARD + ["wallclock_req", "nodes_req", "processors_req",
                                           "gpus_req", "mem_req", "submit_time", "start_time",
                                           "end_time", "run_time"])
    t = pq.read_table(path, columns=cols)
    notes["raw_job_rows"] = t.num_rows
    tgt = pa.schema([f if f.name not in EAGLE_LOW_CARD
                     else pa.field(f.name, pa.dictionary(pa.int32(), pa.string()))
                     for f in t.schema])
    df = t.cast(tgt).to_pandas()
    print(f"  rows={t.num_rows:,}")

    df["job_id"] = pd.to_numeric(df["job_id"], errors="coerce")
    notes["raw_distinct_job_ids"] = int(df.job_id.nunique())
    for c in ("wallclock_req", "nodes_req", "processors_req", "gpus_req", "mem_req", "run_time"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    j = pd.DataFrame({
        "dataset": ds,
        "job_id": df.job_id.astype("Int64").astype("string"),
        "user_id": df["user"].astype("string"),
        "account_id": df["account"].astype("string"),
        "app_name": df["name"].astype("string"),
        "queue_id": df["partition"].astype("string"),
        "qos": df["qos"].astype("string"),
        "state_raw": df["state"].astype("string"),
        "n_gpus_req": df.gpus_req.astype("float64"),
        "req_cpus": df.processors_req.astype("float64"),
        "req_mem_gib": df.mem_req.astype("float64") / 1024.0,
        "n_nodes_req": df.nodes_req.astype("float64"),
        "wallclock_limit_s": df.wallclock_req.astype("float64"),
        "run_s_native": df.run_time.astype("float64"),
        "work_dir": df["work_dir"].astype("string"),
        "submit_line": df["submit_line"].astype("string"),
    })
    j = add_task_ordinal(j, ds, notes)      # Slurm array jobs: one id, many task records
    j["user_gid"] = gid(ds, "user", j["user_id"])
    j["account_gid"] = gid(ds, "account", j["account_id"])
    j["state_norm"] = j["state_raw"].str.lower().map(STATE_MAP).fillna("other")
    t0 = float(df.submit_time.dropna().astype("datetime64[ns]").astype("int64").min() / 1e9)
    notes["dataset_t0_utc"] = dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat()
    for c, short in (("submit_time", "submit"), ("start_time", "start"), ("end_time", "end")):
        j[f"e_{short}"] = to_epoch(pd.to_datetime(df[c], errors="coerce"), tz,
                                   f"jobs_{short}", notes)
        j = add_time_cols(j, f"e_{short}", short, t0, tz, notes)
    j["run_s"] = j["run_s_native"].where(j["run_s_native"].notna(), j["t_end_s"] - j["t_start_s"])
    j["wallclock_overrun"] = j["run_s"] > j["wallclock_limit_s"]
    j["wait_s"] = j["t_start_s"] - j["t_submit_s"]
    j["n_alloc_units"] = j["n_nodes_req"]
    j["n_gpus_obs"] = pd.NA
    j["n_machines"] = pd.NA
    j["gpu_label_req"] = pd.NA
    j["gpu_label_obs"] = pd.NA
    j["gpu_label_final"] = pd.NA
    j["is_gpu_job"] = j["n_gpus_req"].fillna(0) > 0
    j["label_provenance"] = np.where(j["is_gpu_job"], "unknown:gpu_count_only", "not_a_gpu_job")

    sub = (j.groupby(["queue_id", "qos"], dropna=False, observed=True)
           .agg(n_jobs=("job_id", "size"), n_gpu_jobs=("is_gpu_job", "sum"),
                median_gpus=("n_gpus_req", "median"), median_nodes=("n_nodes_req", "median"),
                median_run_s=("run_s", "median"), median_wait_s=("wait_s", "median")).reset_index())
    sub["dataset"] = ds
    sub["kind"] = "partition+qos"
    sub["gpu_share"] = sub.n_gpu_jobs / sub.n_jobs

    jobs = j[["dataset", "job_id", "job_gid", "task_ordinal", "n_array_tasks",
              "is_array_task", "native_id_missing",
              "user_id", "user_gid", "account_id",
              "account_gid", "app_name", "queue_id", "qos", "state_raw", "state_norm",
              "n_nodes_req", "n_gpus_req", "n_gpus_obs", "req_cpus", "req_mem_gib",
              "wallclock_limit_s", "wallclock_overrun", "run_s",
              "gpu_label_req", "gpu_label_obs", "gpu_label_final", "label_provenance",
              "is_gpu_job",
              "t_submit_s", "t_submit_utc", "snap_submit", "tod_submit",
              "t_start_s", "t_start_utc", "snap_start", "tod_start",
              "t_end_s", "t_end_utc", "snap_end", "tod_end", "wait_s",
              "work_dir", "submit_line"]].copy()

    bundle.add("jobs", jobs)
    bundle.empty("requests", "Eagle's request grain == job grain; request columns live in "
                             "jobs.parquet")
    bundle.empty("placements", "OEDI 5860 publishes scheduler accounting only: no node id, "
                               "no device id")
    bundle.empty("placement_devices", "no node or device identity is published")
    bundle.empty("nodes", "no machine inventory is published for Eagle")
    bundle.empty("devices", "no machine inventory is published for Eagle")
    bundle.empty("gpu_types", "Eagle publishes gpus_req (a count) only; no card model, and "
                              "partition names are anonymised")
    # shared name: every dataset's subclusters carries `subcluster_id`
    sub = sub.assign(subcluster_id=sub.queue_id.astype(str) + "/" + sub.qos.astype(str))
    bundle.add("subclusters", sub[["dataset", "kind", "subcluster_id", "queue_id", "qos",
                                   "n_jobs", "n_gpu_jobs",
                                   "gpu_share", "median_gpus", "median_nodes",
                                   "median_run_s", "median_wait_s"]])
    return jobs, bundle.tables["nodes"], bundle.tables["devices"]


# ========================================================================= mit
# MIT Supercloud, datacenter-challenge/202201 release.  Two files hold the whole
# signal: `slurm-log.csv` (one row per job, Slurm accounting) and
# `labelled_jobids.csv` (id_job -> model name for the known DNN jobs).
#
# What this release does NOT publish, and therefore what stays empty:
#   * no machine inventory (no node-data/hostlist table in this release), so
#     `devices` cannot be built and per-node GPU *counts* are not stated;
#   * no per-device allocation (the nvidia_smi/DCGM files live in the *2022-hpca*
#     release, whose README forbids using the data "to extrapolate and draw
#     broader conclusions", and its 44 instrumented id_jobs do not intersect the
#     labelled set at all) -- so `placement_devices` is empty by policy, not by
#     oversight;
#   * no card model name: the finest GPU label is the Slurm TRES class (gpu:volta).
def mit_tres_parse(s):
    """Slurm TRES string -> (tier label or None, gpu count, {tres name: raw}).

    `tres_alloc` looks like "1=8,2=32G,4=1,1002=2": numeric resource ids whose
    meaning only tres-mapping.txt gives.  The probe that produced the wrong
    exclusion in S9.5 matched the *words* volta/tesla against these strings and
    so reported "no GPU label" on rows that were 100% labelled.
    """
    parts = {}
    if isinstance(s, str):
        for tok in s.split(","):
            if "=" not in tok:
                continue
            k, _, v = tok.partition("=")
            try:
                kid = int(k)
            except ValueError:
                continue
            parts[MIT_TRES.get(kid, "tres" + str(k))] = v.strip()
    tier, n = None, 0.0
    for name, lbl in MIT_GPU_TIER.items():
        raw = parts.get(name)
        if raw is None:
            continue
        m = re.match(r"^(\d+)", raw)
        cnt = float(m.group(1)) if m else 0.0
        if cnt > n:
            tier, n = lbl, cnt
    return tier, (n if tier is not None else 0.0), parts


def mit_node_list(s):
    """`['r4858666-n685852', 'r9-n4']` -> two node ids.

    Anonymised hostnames keep their hyphen, so the token pattern allows it and
    the list brackets/quotes act as separators.  Slurm node *ranges* would slip
    through as one token; that case is counted and reported, never expanded.
    """
    if not isinstance(s, str) or not s.strip():
        return []
    return MIT_NODE_TOKEN.findall(s)


def clean_mit(raw: str, bundle: Bundle, notes: dict):
    ds, tz = "mit", DATASET_TZ["mit"]
    cols = ["id_job", "id_user", "nodelist", "nodes_alloc", "gres_used", "constraints",
            "partition", "state", "timelimit", "time_submit", "time_eligible",
            "time_start", "time_end", "tres_alloc", "tres_req", "cpus_req", "mem_req",
            "id_array_job", "id_array_task", "exit_code", "derived_ec"]
    log = pd.read_csv(os.path.join(raw, "slurm-log.csv"), usecols=cols,
                      na_values=MIT_NA, low_memory=False)
    notes["slurm_log_rows"] = int(len(log))
    notes["slurm_log_distinct_id_job"] = int(log.id_job.nunique())
    notes["slurm_log_gres_used_non_null"] = int(log.gres_used.notna().sum())
    notes["readme_declares"] = {"jobs": 460497, "gpu_jobs_requesting": 98177}

    lab = pd.read_csv(os.path.join(raw, "labelled_jobids.csv"), na_values=MIT_NA)
    notes["labelled_rows"] = int(len(lab))
    notes["labelled_models"] = int(lab["model"].nunique())
    notes["labelled_ids_not_in_slurm_log"] = int((~lab.id_job.isin(set(log.id_job))).sum())

    for col, short in (("tres_alloc", "obs"), ("tres_req", "req")):
        parsed = log[col].map(mit_tres_parse)
        log["tier_" + short] = [p[0] for p in parsed]
        log["n_gpu_" + short] = [p[1] for p in parsed]
        log["gpu_raw_" + short] = [
            "|".join(f"{k}={v}" for k, v in sorted(p[2].items())
                     if str(k).startswith("gpu:")) or pd.NA for p in parsed]
        notes["mit_gpu_jobs_by_" + short] = int(log["tier_" + short].notna().sum())
        notes["mit_gpu_tier_counts_by_" + short] = (
            log["tier_" + short].value_counts(dropna=True).to_dict())

    log = log.merge(lab.rename(columns={"model": "app_name"}), on="id_job", how="left")
    notes["mit_named_model_rows"] = int(log.app_name.notna().sum())
    notes["mit_named_model_rows_that_requested_gpu"] = int(
        (log.app_name.notna() & (log.n_gpu_obs > 0)).sum())
    notes["mit_named_model_gpu_tier_counts"] = (
        log.loc[log.app_name.notna() & (log.n_gpu_obs > 0), "tier_obs"]
        .value_counts().to_dict())

    # ---- placements: one row per (job, node) the scheduler actually handed out
    nl = log.nodelist.map(mit_node_list)
    notes["mit_nodelist_range_like_rows"] = int(
        log.nodelist.astype(str).str.contains(r"\[\d", regex=True).sum())
    notes["mit_nodelist_empty_after_parse"] = int((nl.str.len() == 0).sum())
    notes["mit_nodelist_multi_node_rows"] = int((nl.str.len() > 1).sum())
    keep = log[nl.str.len() > 0].copy()
    keep["machine"] = nl[keep.index]
    p = keep.explode("machine")
    p = p[p.machine.notna()].copy()
    p["job_id"] = p.id_job.astype("Int64").astype("string")
    p["n_nodes_job"] = p.groupby("id_job")["machine"].transform("size")
    p["alloc_ordinal"] = p.groupby("id_job").cumcount()
    p["n_gpus_node"] = p.n_gpu_obs.where(p.n_nodes_job == 1)      # never divided up
    notes["mit_placement_rows"] = int(len(p))
    notes["mit_placement_rows_without_per_node_gpu_count"] = int(p.n_gpus_node.isna().sum())

    p["dataset"] = ds
    p["job_gid"] = gid(ds, "job", p.id_job)
    p["machine_gid"] = gid(ds, "machine", p.machine)
    p["gpu_label"] = p.tier_obs.fillna(p.tier_req)
    p["label_provenance"] = np.where(p.tier_obs.notna(), "observed:slurm_tres_alloc",
                                     np.where(p.tier_req.notna(), "requested:slurm_tres_req",
                                              "unknown:no_gpu_tres"))
    p["e_start"] = pd.to_numeric(p.time_start, errors="coerce")
    p["e_end"] = pd.to_numeric(p.time_end, errors="coerce")

    # ---- node dimension: everything the log named, nothing more
    obs = (p.groupby("machine", observed=True)
           .agg(n_jobs_on_node=("job_id", "nunique"),
                n_gpu_jobs_on_node=("tier_obs", lambda s: int(s.notna().sum())),
                gpu_label=("gpu_label", mode_first),
                gpu_labels_seen=("gpu_label", pipe_distinct),
                t_first=("e_start", "min"), t_last=("e_start", "max")).reset_index())
    notes["mit_machines_with_two_gpu_classes"] = int(
        obs.gpu_labels_seen.astype(str).str.contains(r"\|").sum())
    notes["mit_machines_without_gpu_label"] = int(obs.gpu_label.isna().sum())
    nodes = pd.DataFrame({
        "dataset": ds, "machine_id": obs.machine,
        "machine_gid": gid(ds, "machine", obs.machine),
        "cap_gpu": pd.NA,        # no machine inventory is published in this release
        "cap_cpu": pd.NA, "cap_mem": pd.NA,
        "gpu_type_raw": obs.gpu_labels_seen,
        "gpu_label": obs.gpu_label,
        "is_gpu_machine": obs.n_gpu_jobs_on_node > 0,
        "label_provenance": "joined:slurm_tres_of_jobs_on_node",
    })
    devices = pd.DataFrame(columns=["dataset", "device_gid", "machine_gid",
                                    "machine_id", "gpu_index", "gpu_label",
                                    "gpu_type_raw", "label_provenance"])

    # ---- jobs
    mem_native = pd.to_numeric(log.mem_req, errors="coerce")
    sentinel = mem_native > 8e18
    notes["mit_mem_req_int64_sentinel_rows"] = int(sentinel.sum())
    notes["mit_mem_req_native_median_excluding_sentinel"] = float(
        mem_native[~sentinel].median())
    mem_gib = (mem_native / 1024.0).where(~sentinel, pd.NA)
    j = pd.DataFrame({
        "dataset": ds,
        "job_id": log.id_job.astype("Int64").astype("string"),
        "user_id": log.id_user.astype("string"),
        "account_id": pd.NA,                    # MIT publishes no project/account field
        "app_name": log.app_name.astype("string"),
        "queue_id": log.partition.astype("string"),
        "qos": pd.NA,
        "state_raw": log.state.astype("string"),
        "n_alloc_units": pd.to_numeric(log.nodes_alloc, errors="coerce"),
        "n_machines": nl.str.len().replace(0, pd.NA),
        "n_gpus_req": pd.to_numeric(log.n_gpu_req, errors="coerce"),
        "n_gpus_obs": pd.to_numeric(log.n_gpu_obs, errors="coerce"),
        "req_cpus": pd.to_numeric(log.cpus_req, errors="coerce"),
        # Slurm writes "no memory limit" as int64 max, and MIT's number jitter keeps
        # it above 8e18: 346,766 of 395,914 rows are that sentinel, not a size.  The
        # remaining values are megabytes (median 12,288 -> 12 GiB), so GiB = /1024.
        "req_mem_gib": mem_gib,
        "gpu_label_req": log.tier_req, "gpu_label_obs": log.tier_obs,
        "gpu_label_final": log.tier_obs.fillna(log.tier_req),
        "gpu_tres_alloc_raw": log.gpu_raw_obs, "gpu_tres_req_raw": log.gpu_raw_req,
        "constraints": log.constraints.astype("string"),
        "id_array_job": log.id_array_job.astype("string"),
        "id_array_task": log.id_array_task.astype("string"),
        "exit_code": log.exit_code.astype("string"),
        "derived_ec": log.derived_ec.astype("string"),
        "timelimit_s": pd.to_numeric(log.timelimit, errors="coerce"),
    })
    notes["mit_state_distinct_codes"] = (
        log.state.astype(str).value_counts(dropna=False).head(8).to_dict())
    j["state_norm"] = pd.NA            # codes are numeric and undocumented: not mapped
    j["label_provenance"] = np.where(j.gpu_label_obs.notna(), "observed:slurm_tres_alloc",
                                     np.where(j.gpu_label_req.notna(),
                                              "requested:slurm_tres_req", "unknown"))
    j["is_gpu_job"] = (j.n_gpus_obs.fillna(0) > 0) | (j.n_gpus_req.fillna(0) > 0)

    j = add_task_ordinal(j, ds, notes)
    j["user_gid"] = gid(ds, "user", j["user_id"])
    j["account_gid"] = gid(ds, "account", j["account_id"])

    t0 = float(pd.to_numeric(log.time_submit, errors="coerce").min())
    notes["dataset_t0_utc"] = dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat()
    for col, short in (("time_submit", "submit"), ("time_start", "start"),
                       ("time_end", "end")):
        j["e_" + short] = to_epoch(pd.to_numeric(log[col], errors="coerce"),
                                   tz, "jobs_" + short, notes)
        j = add_time_cols(j, "e_" + short, short, t0, tz, notes)
    j["wait_s"] = j["t_start_s"] - j["t_submit_s"]
    j["run_s"] = j["t_end_s"] - j["t_start_s"]
    p = seconds_since(p, "e_start", t0)
    p["t_end_s"] = p["e_end"] - t0

    # Every class the scheduler actually handed out gets a row, including the six
    # tesla jobs -- deriving this table from the node *modal* label would hide them.
    used = (p.groupby("gpu_label", observed=True)
            .agg(n_placements=("machine", "size"), n_machines_used=("machine", "nunique"),
                 n_jobs_used=("job_id", "nunique"),
                 raw_spellings=("gpu_raw_obs", pipe_distinct)).reset_index())
    inv = (nodes[nodes.gpu_label.notna()].groupby("gpu_label", observed=True)
           .agg(n_machines_in_inventory=("machine_id", "nunique")))
    gt = used.merge(inv, on="gpu_label", how="left")
    gt["n_machines_in_inventory"] = gt.n_machines_in_inventory.fillna(0).astype("int64")
    gt["dataset"] = ds
    gt["n_devices_in_inventory"] = pd.NA
    gt["mem_gib_stated"] = pd.NA
    gt["mem_gib_measured"] = pd.NA
    gt["model_name_published"] = False
    gt["tres_class_published"] = True
    gt["label_provenance"] = "observed:slurm_tres_class_name"
    gt["note"] = ("label is the Slurm TRES resource class from tres-mapping.txt "
                  "(architecture family, not a card model); this release publishes no "
                  "machine inventory and no per-device allocation, so capacity is not "
                  "stated here.  A range sample of the *other* (2022-hpca) release's "
                  "nvidia_smi.csv measured 52/52 devices at 32,510 MiB free+used, i.e. "
                  "one capacity only -- that release's README restricts use to the "
                  "HPCA'22 analysis, so the number is reported, not redistributed.")

    sub = (j.groupby("queue_id", dropna=False, observed=True)
           .agg(n_jobs=("job_id", "size"), n_gpu_jobs=("is_gpu_job", "sum"),
                n_named_models=("app_name", "nunique"),
                median_gpus=("n_gpus_obs", "median"), median_nodes=("n_machines", "median"),
                median_run_s=("run_s", "median"), median_wait_s=("wait_s", "median"),
                t_first=("t_submit_s", "min"), t_last=("t_submit_s", "max")).reset_index())
    sub["dataset"] = ds
    sub["kind"] = "slurm_partition"
    sub = sub.rename(columns={"queue_id": "subcluster_id"})
    sub["gpu_share"] = sub.n_gpu_jobs / sub.n_jobs

    jobs = j[["dataset", "job_id", "job_gid", "task_ordinal", "n_array_tasks",
              "is_array_task", "native_id_missing",
              "user_id", "user_gid", "account_id", "account_gid",
              "app_name", "queue_id", "qos", "state_raw", "state_norm",
              "n_alloc_units", "n_machines", "n_gpus_req", "n_gpus_obs",
              "req_cpus", "req_mem_gib",
              "gpu_label_req", "gpu_label_obs", "gpu_label_final",
              "gpu_tres_alloc_raw", "gpu_tres_req_raw", "constraints",
              "id_array_job", "id_array_task", "exit_code", "derived_ec", "timelimit_s",
              "label_provenance", "is_gpu_job",
              "t_submit_s", "t_submit_utc", "snap_submit", "tod_submit",
              "t_start_s", "t_start_utc", "snap_start", "tod_start",
              "t_end_s", "t_end_utc", "snap_end", "tod_end", "wait_s", "run_s"]].copy()

    bundle.add("jobs", jobs)
    bundle.empty("requests", "MIT publishes the request at job grain; the request "
                             "columns (cpus_req/mem_req/tres_req) live in jobs.parquet")
    bundle.add("placements", p[["dataset", "job_id", "job_gid", "machine", "machine_gid",
                                "alloc_ordinal", "n_nodes_job", "n_gpus_node", "gpu_label",
                                "label_provenance", "t_start_s", "t_end_s", "snap_start"]])
    bundle.empty("placement_devices", "no per-device (node, gpu_index) allocation is "
                                     "published in the datacenter-challenge/202201 release")
    bundle.add("nodes", nodes)
    bundle.add("devices", devices)
    bundle.add("gpu_types", gt)
    bundle.add("subclusters", sub[["dataset", "kind", "subcluster_id", "n_jobs",
                                   "n_gpu_jobs", "gpu_share", "n_named_models",
                                   "median_gpus", "median_nodes", "median_run_s",
                                   "median_wait_s", "t_first", "t_last"]])

    known_jobs, known_machines = set(jobs.job_id.astype(str)), set(nodes.machine_id)
    resolve_fk(bundle.tables["placements"], "job_id", "job_gid", known_jobs, notes,
               "placements_job_gid_unresolved")
    resolve_fk(bundle.tables["placements"], "machine", "machine_gid", known_machines,
               notes, "placements_machine_gid_unresolved")
    return jobs, nodes, devices


CLEANERS = {"ali_v2020": clean_ali, "philly": clean_philly, "eagle": clean_eagle,
            "mit": clean_mit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(CLEANERS))
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-devices", action="store_true",
                    help="skip the device-grain table (slowest to build)")
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    bundle = Bundle(args.dataset, out, args.no_devices)
    notes: dict = {}
    t_start = time.time()
    jobs, nodes, devices = CLEANERS[args.dataset](os.path.expanduser(args.raw), bundle, notes)
    # the shared contract lives here, once, instead of in three column lists
    notes["core_columns_filled_na"] = [c for c in CORE_JOB_COLS if c not in jobs.columns]
    jobs = ensure_columns(jobs, CORE_JOB_COLS)
    bundle.add("jobs", jobs)

    bad = check_id_space(jobs, nodes, devices)
    rec = {k: recurrence(jobs, k) for k in HEAD_KEYS[args.dataset] if k in jobs.columns}
    notes["gpu_jobs"] = {"jobs": int(len(jobs)),
                         "is_gpu_job": int(jobs["is_gpu_job"].sum()),
                         "snapshots": int(jobs["snap_submit"].nunique())}
    notes["wall_seconds"] = round(time.time() - t_start, 1)
    bundle.write(notes, rec, bad)
    print(f"\n  manifest.json / profile.md -> {out}")
    for k, v in rec.items():
        if v:
            print(f"  head {k:<12} entities={v['entities']:>10,} "
                  f"recurring={v['pct_entities_recurring']:>6}% "
                  f"max_snap={v['max_snapshots']}")
    if bad:
        print("  !!! INVARIANT VIOLATIONS", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
