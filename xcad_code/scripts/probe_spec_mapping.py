"""Feasibility probe for the "GPU model name -> capability spec" mapping.

Reads only the cleaned publish tables. No GPU, no model change, no writes.

Q1  inventory : which columns of which table actually describe a device?
Q2  exposure  : how much of the workload sits behind a model name we cannot
                resolve (MISC / UNRESOLVED / unpublished spellings)?
Q3  leakage   : does an unresolvable name contaminate a job's label set, or is
                every job either fully resolvable or fully opaque?
Q4  overlap   : what device attributes do the two GPU traces BOTH publish?
                (this is the attribute space a "dataset-agnostic" model could
                 actually consume -- everything else is Ali-only decoration)

usage: ~/xcad/xcad_code/.venv/bin/python scripts/probe_spec_mapping.py
"""
import os
import sys
import pandas as pd

R = os.path.expanduser("~/xcad/datasets/clean")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
DEV_COLS = ("gpu_label", "gpu_type_raw", "mem", "memory", "model",
            "arch", "cuda", "core", "bandwidth", "nvlink", "tdp")


def load(ds, name):
    p = os.path.join(R, ds, f"{name}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


def dev_columns(ds):
    """columns that plausibly carry a physical device attribute."""
    out = {}
    for t in ("gpu_types", "devices", "nodes", "placements", "placement_devices"):
        df = load(ds, t)
        if df is None:
            continue
        hit = [c for c in df.columns if any(k in c.lower() for k in DEV_COLS)]
        if hit:
            out[t] = hit
    return out


def q1():
    print("=" * 78)
    print("Q1  which published columns describe a physical device")
    print("=" * 78)
    for ds in ("ali_v2020", "philly"):
        print(f"\n-- {ds}")
        inv = dev_columns(ds)
        for t, cols in inv.items():
            df = load(ds, t)
            print(f"   {t:<18} {len(df):>9,} rows   {cols}")
        gt = load(ds, "gpu_types")
        if gt is not None:
            print(f"   gpu_types columns: {list(gt.columns)}")


def q2():
    print()
    print("=" * 78)
    print("Q2  workload exposure to names we cannot resolve")
    print("=" * 78)
    for ds in ("ali_v2020", "philly"):
        gt = load(ds, "gpu_types")
        dv = load(ds, "devices")
        pl = load(ds, "placements")
        if gt is None or dv is None:
            continue
        print(f"\n-- {ds}")
        # inventory side: how many devices carry a raw model spelling?
        raw = (dv["gpu_type_raw"].astype("string")
               if "gpu_type_raw" in dv.columns else pd.Series(pd.NA, index=dv.index))
        dv = dv.assign(_has_raw=raw.fillna("").str.strip().ne(""))
        n = dv.groupby("gpu_label", observed=True).agg(
            devices=("device_gid", "size"),
            devices_with_raw=("_has_raw", "sum"),
            machines=("machine_gid", "nunique"),
        )
        n["devices_with_raw_pct"] = (100 * n["devices_with_raw"] / n["devices"]).round(2)
        print(n.to_string())
        # demand side: placements and the jobs that own them
        if pl is not None and len(pl):
            g = pl[pl["is_gpu_placement"].fillna(False).astype(bool)] \
                if "is_gpu_placement" in pl.columns else pl
            d = (g.groupby("gpu_label", observed=True)
                 .agg(gpu_placements=("job_gid", "size"), jobs_touching=("job_gid", "nunique")))
            d["gpu_placements_pct"] = (100 * d["gpu_placements"] / d["gpu_placements"].sum()).round(2)
            d["jobs_pct"] = (100 * d["jobs_touching"] / d["jobs_touching"].sum()).round(2)
            print("\n   demand (gpu placements / distinct jobs):")
            print(d.sort_values("gpu_placements", ascending=False).to_string())


def q3():
    print()
    print("=" * 78)
    print("Q3  does an opaque name leak into otherwise-resolvable jobs?")
    print("=" * 78)
    OPAQUE = {"NVIDIA-LEGACY-MISC", "UNRESOLVED", "GPU", "GPU-UNKNOWN", ""}
    for ds in ("ali_v2020", "philly"):
        jobs = load(ds, "jobs")
        if jobs is None or "gpu_label_final" not in jobs.columns:
            continue
        lab = jobs["gpu_label_final"].astype("string").fillna("")
        gpu = jobs[jobs["is_gpu_job"].fillna(False).astype(bool)] \
            if "is_gpu_job" in jobs.columns else jobs
        sets = gpu["gpu_label_final"].astype("string").fillna("") \
            .map(lambda s: "|".join(sorted({p for p in s.split("|") if p})))
        members = sets.map(lambda s: set(s.split("|")) - {""})
        opaque = members.map(lambda s: bool(s & OPAQUE))
        named = members.map(lambda s: bool(s - OPAQUE))
        print(f"\n-- {ds}   gpu jobs {len(gpu):,}")
        print(f"   only opaque labels : {int((opaque & ~named).sum()):>9,}  "
              f"{100 * float((opaque & ~named).mean()):5.2f}%")
        print(f"   only named labels  : {int((~opaque & named).sum()):>9,}  "
              f"{100 * float((~opaque & named).mean()):5.2f}%")
        print(f"   mixed (opaque+named): {int((opaque & named).sum()):>9,}  "
              f"{100 * float((opaque & named).mean()):5.2f}%")
        print(f"   distinct label sets : {sets.nunique():,}")
        top = sets.value_counts().head(12)
        for k, v in top.items():
            print(f"      {v:>9,}  {'|'.join(sorted(k)) or '(empty)'}")


def q4():
    print()
    print("=" * 78)
    print("Q4  attribute space the two traces BOTH publish")
    print("=" * 78)
    ali = load("ali_v2020", "gpu_types")
    ph = load("philly", "gpu_types")
    print("\n-- ali gpu_types (full)")
    print(ali.to_string(index=False))
    print("\n-- philly gpu_types (full)")
    print(ph.to_string(index=False))
    for ds in ("ali_v2020", "philly"):
        nd = load(ds, "nodes")
        if nd is None:
            continue
        print(f"\n-- {ds} nodes columns: {list(nd.columns)}")
        for c in nd.columns:
            if "mem" in c.lower() or c.lower() in ("cap_gpu", "gpu_type_raw", "gpu_label"):
                vc = nd[c].value_counts(dropna=False)
                print(f"   {c:<16} nunique={nd[c].nunique(dropna=False):>5}  "
                      f"{dict(list(vc.items())[:8])}")
        dv = load(ds, "devices")
        if dv is not None:
            print(f"   devices columns: {list(dv.columns)}")
            for c in dv.columns:
                if any(k in c.lower() for k in DEV_COLS):
                    print(f"   {c:<22} nunique={dv[c].nunique(dropna=False):>5}  "
                          f"{dict(list(dv[c].value_counts(dropna=False).items())[:6])}")


def main():
    if not os.path.isdir(R):
        sys.exit(f"not on the server: {R} missing")
    q1(); q2(); q3(); q4()


if __name__ == "__main__":
    main()
