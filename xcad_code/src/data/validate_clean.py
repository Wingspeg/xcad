#!/usr/bin/env python3
"""
validate_clean.py -- audit the cleaned trace bundles produced by clean_traces.py.

Checks, per dataset:
  H1  jobs.parquet primary key (job_gid, task_ordinal) is unique
  H2  no synthetic "__missing" id tokens leaked into the fact tables
  H3  referential integrity: every job_gid / machine_gid / device_gid used by
      placements, placement_devices and requests exists in the dimension table
  H4  label vocabulary is closed: every gpu_label used is in gpu_types,
      CPU-ONLY or UNRESOLVED
  H5  time sanity: snapshot range, share of missing stamps, negative durations
  H6  id-space segments stay disjoint (re-checked from the written files)
  H7  jobs.parquet carries every column of the shared contract (CORE_JOB_COLS),
      so one dataset-agnostic loader can read all three traces
  H8  jobs.gpu_label_final is a pipe-separated *set* of labels: every element is
      in the H4 vocabulary, and every (job, label) GPU placement pair is covered
      by that job's set -- the composite rows are reported, not hidden

Nothing is repaired here -- a failure means the cleaner needs fixing, not the
output.  Writes validation.md next to the bundles and prints a cross-dataset
comparison so the three traces can be judged side by side.

Usage
  python3 validate_clean.py --root ~/xcad/datasets/clean \
      --datasets ali_v2020 philly eagle
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

try:
    from clean_traces import CORE_JOB_COLS
except ImportError:                      # validator used standalone, without the cleaner
    CORE_JOB_COLS = []

FACT_COLS = {"placements": ("job_gid", "machine_gid"),
             "placement_devices": ("job_gid", "machine_gid", "device_gid"),
             "requests": ("job_gid",)}
ALLOWED_LABELS = {"CPU-ONLY", "UNRESOLVED", ""}
TABLES = ("jobs", "requests", "placements", "placement_devices",
          "nodes", "devices", "gpu_types", "subclusters")


def load(out, name):
    p = os.path.join(out, f"{name}.parquet")
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p)


def validate(out: str, ds: str) -> tuple[dict, list]:
    res, fails = {"dataset": ds}, []
    tables = {n: load(out, n) for n in TABLES}
    jobs = tables["jobs"]
    if jobs is None or len(jobs) == 0:
        fails.append(f"{ds}: jobs.parquet missing or empty")
        return res, fails

    res["jobs_rows"] = int(len(jobs))
    res["jobs_cols"] = int(len(jobs.columns))

    # H1 primary key
    dup = int(jobs.duplicated(["job_gid", "task_ordinal"]).sum()) if \
        "task_ordinal" in jobs.columns else int(jobs.job_gid.duplicated().sum())
    res["pk_duplicate_rows"] = dup
    if dup:
        fails.append(f"{ds}: H1 {dup} duplicated (job_gid, task_ordinal)")

    # H2 synthetic tokens
    n_syn = int(jobs.job_id.astype(str).str.startswith("__missing").sum())
    res["native_id_missing_rows"] = n_syn
    for name, cols in FACT_COLS.items():
        df = tables.get(name)
        if df is None or len(df) == 0 or "job_gid" not in df:
            continue
        bad = int(df.job_gid.astype(str).str.contains("__missing").sum())
        if bad:
            fails.append(f"{ds}: H2 {name} carries {bad} synthetic job_gid")

    # H3 referential integrity
    dimsets = {"job_gid": set(jobs.job_gid),
               "machine_gid": set(tables["nodes"].machine_gid) if tables["nodes"] is not None else set(),
               "device_gid": set(tables["devices"].device_gid) if tables["devices"] is not None else set()}
    for name, cols in FACT_COLS.items():
        df = tables.get(name)
        if df is None or len(df) == 0:
            continue
        res[f"{name}_rows"] = int(len(df))
        for c in cols:
            if c not in df.columns:
                continue
            v = df[c]
            # a key the trace never published is a documented hole, not a broken
            # reference -- the cleaner turns "<ds>#seg:nan" pseudo-nodes into NA
            # precisely so it cannot masquerade as the busiest node in the graph
            nulls = int(v.isna().sum())
            if nulls:
                res[f"{name}:{c}_null_key_rows"] = nulls
                v = v[v.notna()]
            if c == "device_gid":
                unres = v.astype(str).str.contains(r"#dev(<NA>|nan|None)$", na=False)
                res[f"{name}:device_slot_unresolved"] = int(unres.sum())
                v = v[~unres]
            miss = int((~v.isin(dimsets[c])).sum())
            res[f"{name}:{c}_orphan_rows"] = int(miss)
            if miss:
                fails.append(f"{ds}: H3 {name}.{c} has {miss:,} rows with no match "
                             f"in the dimension table")
    if tables["devices"] is not None and len(tables["devices"]):
        orphan = int((~tables["devices"].machine_gid.isin(dimsets["machine_gid"])).sum())
        res["devices_orphan_machines"] = orphan
        if orphan:
            fails.append(f"{ds}: H3 devices references {orphan} unknown machines")

    # H4 label vocabulary
    vocab = set(tables["gpu_types"].gpu_label.astype(str)) | ALLOWED_LABELS if \
        tables["gpu_types"] is not None and len(tables["gpu_types"]) else set(ALLOWED_LABELS)
    for name in ("placements", "placement_devices", "devices"):
        df = tables.get(name)
        if df is None or len(df) == 0 or "gpu_label" not in df:
            continue
        unknown = sorted(set(df.gpu_label.dropna().astype(str)) - vocab)
        res[f"{name}_labels"] = sorted(set(df.gpu_label.dropna().astype(str)))
        if unknown:
            fails.append(f"{ds}: H4 {name} uses labels outside gpu_types: {unknown}")

    # H5 time sanity
    for c in ("t_submit_s", "t_start_s", "t_end_s"):
        if c in jobs:
            v = pd.to_numeric(jobs[c], errors="coerce")
            res[f"{c}:min"] = None if v.notna().sum() == 0 else float(v.min())
            res[f"{c}:max"] = None if v.notna().sum() == 0 else float(v.max())
            res[f"{c}:nan_pct"] = round(100 * float(v.isna().mean()), 3)
    if "snap_submit" in jobs:
        res["snapshots"] = int(jobs.snap_submit.nunique())
        res["snap_range"] = [float(jobs.snap_submit.min()), float(jobs.snap_submit.max())]
    for c in ("wait_s", "run_s"):
        if c in jobs:
            v = pd.to_numeric(jobs[c], errors="coerce")
            res[f"negative_{c}"] = int((v < 0).sum())
            res[f"median_{c}"] = None if v.notna().sum() == 0 else float(v.median())
            if res[f"negative_{c}"] > 0.005 * int(v.notna().sum()):
                fails.append(f"{ds}: H5 {res[f'negative_{c}']} rows with {c} < 0 "
                             f"(>0.5% of timed rows)")

    # extras worth reporting
    if "is_gpu_job" in jobs:
        res["gpu_job_rows"] = int(jobs.is_gpu_job.sum())
        res["gpu_job_pct"] = round(100 * float(jobs.is_gpu_job.mean()), 2)
    for c in ("user_gid", "account_gid"):
        if c in jobs:
            res[c.replace("_gid", "") + "s"] = int(jobs[c].nunique())
    if tables["nodes"] is not None:
        res["machines"] = int(len(tables["nodes"]))
    if tables["devices"] is not None:
        res["gpu_devices"] = int(len(tables["devices"]))
    if tables["gpu_types"] is not None:
        res["gpu_labels"] = int(len(tables["gpu_types"]))

    # H6 segments disjoint (re-checked from files)
    segs = {"job": dimsets["job_gid"], "machine": dimsets["machine_gid"],
            "device": dimsets["device_gid"]}
    for c in ("user_gid", "account_gid"):
        if c in jobs.columns:
            segs[c[:-4]] = set(jobs[c].dropna())
    keys = sorted(segs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            inter = segs[a] & segs[b]
            if inter:
                fails.append(f"{ds}: H6 segment collision {a}/{b}: {len(inter)}")
    # H7 shared column contract: one loader can read all three jobs tables
    miss = [c for c in CORE_JOB_COLS if c not in jobs.columns]
    res["missing_core_job_columns"] = miss
    if miss:
        fails.append(f"{ds}: H7 jobs.parquet lacks shared columns {miss}")
    res["dataset_specific_job_columns"] = [c for c in jobs.columns if c not in CORE_JOB_COLS]

    # H8 job-level labels are a pipe-set of the machine-level labels of that job.
    # A job spanning two card types keeps both ("NVIDIA-T4|NVIDIA-P100"), so the
    # column is a set-valued encoding, not a categorical: report how much of it
    # is composite and prove nothing outside the job's own placements leaked in.
    if "gpu_label_final" in jobs.columns:
        lab = jobs["gpu_label_final"].dropna().astype("string")
        comp = lab.str.contains("|", regex=False)
        res["jobs_label_distinct_values"] = int(lab.nunique())
        res["jobs_label_composite_rows"] = int(comp.sum())
        res["jobs_label_composite_pct"] = round(100 * float(comp.mean()), 2) if len(lab) else 0.0
        pl = tables["placements"]
        if pl is not None and len(pl) and {"job_gid", "gpu_label"} <= set(pl.columns):
            if "is_gpu_placement" in pl.columns:
                pl = pl[pl["is_gpu_placement"].fillna(False).astype(bool)]
            elif "n_gpus" in pl.columns:
                pl = pl[pl["n_gpus"].fillna(0) > 0]
            pairs = (pl[["job_gid", "gpu_label"]].dropna().drop_duplicates()
                     if len(pl) else pl[["job_gid", "gpu_label"]])
            final = (jobs.drop_duplicates("job_gid")
                     .set_index("job_gid")["gpu_label_final"])
            pairs = pairs.assign(_final=pairs["job_gid"].map(final))
            ok = [str(f).split("|") for f in pairs["_final"]]
            bad = int(sum(l not in s for l, s in zip(pairs["gpu_label"], ok)))
            res["placement_labels_not_in_job_label"] = bad
            if bad:
                fails.append(f"{ds}: H8 {bad} (job, gpu_label) placement pairs are not "
                             f"covered by that job's gpu_label_final set")
        # composite spellings must stay inside the closed vocabulary per element
        vocab_all = vocab | {""}
        stray = {p for s in lab.unique().dropna() for p in str(s).split("|")
                 if p not in vocab_all}
        res["jobs_label_elements_outside_vocabulary"] = sorted(stray)
        if stray:
            fails.append(f"{ds}: H8 label elements outside gpu_types: {sorted(stray)}")

    man = os.path.join(out, "manifest.json")
    if os.path.exists(man):
        with open(man) as f:
            res["manifest_absent_tables"] = sorted(json.load(f).get("absent_tables", {}))
    return res, fails


def md_table(df: pd.DataFrame) -> str:
    """Markdown without the optional `tabulate` dependency (the server venv has no network)."""
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        if isinstance(v, (float, np.floating)):
            return "--" if pd.isna(v) else f"{float(v):,.3f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |"
              for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    args = ap.parse_args()
    root = os.path.expanduser(args.root)

    rows, all_fails = [], []
    for ds in args.datasets:
        res, fails = validate(os.path.join(root, ds), ds)
        rows.append(res)
        all_fails += fails
        print(f"\n== {ds} ==")
        for k, v in res.items():
            if isinstance(v, float):
                print(f"  {k:<34} {v:,.3f}")
            else:
                print(f"  {k:<34} {v:,}" if isinstance(v, int) else f"  {k:<34} {v}")

    comp = pd.DataFrame(rows).set_index("dataset")
    keep = [c for c in ("jobs_rows", "gpu_job_rows", "gpu_job_pct", "machines",
                        "gpu_devices", "gpu_labels", "snapshots",
                        "placements_rows", "placement_devices_rows",
                        "users", "accounts", "median_wait_s", "median_run_s",
                        "jobs_label_distinct_values", "jobs_label_composite_pct",
                        "placement_labels_not_in_job_label")
            if c in comp.columns]
    comp = comp[keep]
    print("\n== cross-dataset comparison ==")
    print(comp.to_string())

    with open(os.path.join(root, "validation.md"), "w") as f:
        f.write("# Validation of the cleaned trace bundles\n\n")
        f.write("## cross-dataset comparison\n\n")
        f.write(md_table(comp.reset_index()))
        f.write("\n\n## per-dataset detail\n\n")
        for r in rows:
            f.write(f"### {r['dataset']}\n\n")
            for k, v in r.items():
                if k == "dataset":
                    continue
                f.write(f"- `{k}`: {v}\n")
            f.write("\n")
        f.write("## failures\n\n")
        if all_fails:
            for x in all_fails:
                f.write(f"- !! {x}\n")
        else:
            f.write("none -- H1..H8 passed for every dataset checked\n")
    print("\n  -> " + os.path.join(root, "validation.md"))
    if all_fails:
        print("\nFAILURES:")
        for x in all_fails:
            print("  !!", x)
        raise SystemExit(3)
    print("\nALL CHECKS PASSED (H1..H8)")


if __name__ == "__main__":
    main()
