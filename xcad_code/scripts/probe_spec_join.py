"""Second half of the 型号 -> 规格 verification.

probe_spec_mapping.py answered "can we resolve the names?".  This one answers
the two questions that decide whether the mapping is worth building:

  S1  dictionary reach : after joining an external spec dictionary, what
                         fraction of devices / placements / jobs get a
                         NON-EMPTY capability vector, per attribute?
  S2  discriminative   : do the resolved labels produce pairwise distinct
                         vectors, i.e. does the vector keep the information the
                         categorical label carried (if it does not, the mapping
                         is a downgrade, not a generalisation)?
  S3  intersection     : which attributes can BOTH published GPU traces fill?
                         what is the shared subspace and do its value sets
                         actually overlap?

The dictionary lives here, not in clean_traces.py: nothing is written and no
published table is modified.  Every value carries a source id resolving to a
first-hand NVIDIA URL + retrieval date; unverifiable fields stay None so the
per-attribute coverage below is honest rather than optimistic.

usage: ~/xcad/xcad_code/.venv/bin/python scripts/probe_spec_join.py
"""
import os
import sys
import pandas as pd

R = os.path.expanduser("~/xcad/datasets/clean")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:8.3f}")

# ---------------------------------------------------------------- dictionary
# attributes = the capability vector a scheduler would consume
ATTRS = ["n_gpus_per_node", "mem_gib", "mem_bw_gbs", "fp32_tflops",
         "fp64_tflops", "fp16_tflops", "tensor_tflops", "cuda_cores",
         "compute_capability", "has_tensor_core", "has_nvlink", "tdp_w"]

SOURCES = {
    "V100-PDF": "https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet.pdf  (retr 2026-11-16)",
    "T4-PDF":   "https://web.archive.org/web/20190905153638/https://www.nvidia.com/content/dam/"
                "en-zz/Solutions/Data-Center/tesla-product-overview/nvidia-tesla-t4-product-brief.pdf (retr 2026-11-16)",
    "P100-PAGE": "https://web.archive.org/web/20190701194257/https://www.nvidia.com/en-us/about-nvidia/"
                 "kepler-architecture/  -> Tesla P100 SXM2/PCIe comparison table (retr 2026-11-16)",
    "K80-PAGE": "https://web.archive.org/web/20180817004716/https://www.nvidia.com/en-us/data-center/tesla-k80/ (retr 2026-11-16)",
    "M60-PDF":  "https://web.archive.org/web/20170329025817/https://www.nvidia.com/content/PDF/"
                "maxwell-architecture/NVIDIA-Tesla-M60-Product-Brief.pdf (retr 2026-11-16)",
    "K40-PDF":  "https://web.archive.org/web/20160304045934/http://www.nvidia.com/content/PDF/kepler/"
                "nvidia-gpu-boost-tesla-k40-06767-001-v02.pdf (retr 2026-11-16)",
    "CC-LIST":  "https://web.archive.org/web/20200102225936/https://developer.nvidia.com/cuda-gpus (retr 2026-11-16)",
    "GRID-VPAGE": "https://docs.nvidia.com/grid/latest/grid-vgpu-user-guide/index.html (retr 2026-11-16)",
}

# None  -> not published by the source we hold (kept NA on purpose)
SPEC = pd.DataFrame([
    # model        n_gpus mem  bw   fp32  fp64  fp16  tens  cores cc    tens? nvlink? tdp
    dict(model="P100",        n_gpus_per_node=None, mem_gib=16.0, mem_bw_gbs=732.0,
         fp32_tflops=9.3, fp64_tflops=4.7, fp16_tflops=18.7, tensor_tflops=None,
         cuda_cores=None, compute_capability=6.0, has_tensor_core=0, has_nvlink=1,
         tdp_w=None, src="P100-PAGE,CC-LIST"),
    dict(model="V100_16",     n_gpus_per_node=None, mem_gib=16.0, mem_bw_gbs=900.0,
         fp32_tflops=15.7, fp64_tflops=7.0, fp16_tflops=None, tensor_tflops=112.0,
         cuda_cores=5120, compute_capability=7.0, has_tensor_core=1, has_nvlink=1,
         tdp_w=250.0, src="V100-PDF,CC-LIST"),
    dict(model="V100_32",     n_gpus_per_node=None, mem_gib=32.0, mem_bw_gbs=900.0,
         fp32_tflops=15.7, fp64_tflops=7.0, fp16_tflops=None, tensor_tflops=112.0,
         cuda_cores=5120, compute_capability=7.0, has_tensor_core=1, has_nvlink=1,
         tdp_w=250.0, src="V100-PDF,CC-LIST"),
    dict(model="T4",          n_gpus_per_node=None, mem_gib=16.0, mem_bw_gbs=300.0,
         fp32_tflops=8.1, fp64_tflops=None, fp16_tflops=65.0, tensor_tflops=None,
         cuda_cores=2560, compute_capability=7.5, has_tensor_core=1, has_nvlink=0,
         tdp_w=70.0, src="T4-PDF,CC-LIST"),
    dict(model="K80",         n_gpus_per_node=2.0, mem_gib=24.0, mem_bw_gbs=480.0,
         fp32_tflops=8.73, fp64_tflops=2.91, fp16_tflops=None, tensor_tflops=None,
         cuda_cores=4992, compute_capability=3.7, has_tensor_core=0, has_nvlink=0,
         tdp_w=None, src="K80-PAGE,CC-LIST"),
    dict(model="M60",         n_gpus_per_node=2.0, mem_gib=8.0, mem_bw_gbs=None,
         fp32_tflops=None, fp64_tflops=None, fp16_tflops=None, tensor_tflops=None,
         cuda_cores=2048, compute_capability=5.2, has_tensor_core=0, has_nvlink=0,
         tdp_w=300.0, src="M60-PDF,CC-LIST"),
    dict(model="K40",         n_gpus_per_node=None, mem_gib=None, mem_bw_gbs=None,
         fp32_tflops=None, fp64_tflops=None, fp16_tflops=None, tensor_tflops=None,
         cuda_cores=None, compute_capability=3.5, has_tensor_core=0, has_nvlink=0,
         tdp_w=None, src="K40-PDF,CC-LIST"),
]).set_index("model")

# Ali gpu_label -> dictionary key.  matched on the published label, and the
# match itself is reported so a new spelling shows up as a miss, not a silent NA.
ALI_RULES = [("P100", "P100"), ("V100-32", "V100_32"), ("V100", "V100_16"),
             ("T4", "T4")]
OPAQUE_TOKENS = {"NVIDIA-LEGACY-MISC", "UNRESOLVED", "GPU", "GPU-UNKNOWN", ""}


def load(ds, name):
    p = os.path.join(R, ds, f"{name}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


def ali_model(label):
    u = str(label).upper()
    if u in OPAQUE_TOKENS:
        return None
    for pat, key in ALI_RULES:
        if pat in u:
            return key
    return None


def philly_model(label):
    return None            # model_name_published == False, by construction


def vectorize(ds):
    """device-level table + capability vector, with per-attribute provenance."""
    dv = load(ds, "devices")
    gt = load(ds, "gpu_types").set_index("gpu_label")
    dv = dv.assign(model_key=dv["gpu_label"].map(
        ali_model if ds == "ali_v2020" else philly_model))
    for a in ATTRS:
        dv[a] = dv["model_key"].map(SPEC[a]) if len(SPEC) else None
    # n_gpus_per_node + mem_gib for Philly come out of the tier label itself,
    # which is a capacity statement published by the trace, not a lookup.
    if ds == "philly":
        ext = dv["gpu_label"].str.extract(r"TIER-(\d+)x(\d+)").astype("float")
        dv["n_gpus_per_node_tier"] = ext[0]        # claimed by the machine list
        dv["mem_gib"] = ext[1]
        # model_key stays None: Philly publishes capacity, never a model, so it
        # needs no external dictionary yet still fills capability attributes.
    # device count per node is published by BOTH traces' own inventory, so it is
    # not dictionary-dependent; fill it from the observed device rows.
    cnt = dv.groupby("machine_gid", observed=True)["device_gid"].transform("size")
    dv["n_gpus_per_node_obs"] = cnt
    dv["n_gpus_per_node"] = cnt.astype(float)
    dv["attr_filled"] = dv[ATTRS].notna().sum(axis=1)
    dv["model_matched"] = dv["model_key"].notna()
    return dv, gt


def s1(ds):
    print("=" * 100)
    print(f"S1  {ds}: per-attribute coverage after the join")
    print("=" * 100)
    dv, gt = vectorize(ds)
    pl = load(ds, "placements")
    pdv = load(ds, "placement_devices")
    key = dv.set_index("device_gid")
    j = pdv.assign(model_matched=pdv["device_gid"].map(key["model_matched"]),
                   n_attr=pdv["device_gid"].map(key["attr_filled"]))
    tot = len(j)
    print(f"devices={len(dv):,}  placement_devices={tot:,}")
    print(f"label vocabulary ({dv['gpu_label'].nunique()}): "
          f"{sorted(dv['gpu_label'].astype(str).unique())}")
    m = dv.groupby("gpu_label", observed=True).agg(
        devices=("device_gid", "size"),
        matched=("model_matched", "sum"),
        attrs_filled=("attr_filled", "mean"))
    m["devices_pct"] = 100 * m["devices"] / len(dv)
    pm = j.groupby(j["device_gid"].map(key["gpu_label"]), observed=True).size()
    m["placements"] = pm
    m["placements_pct"] = 100 * m["placements"] / tot
    print(m.to_string())
    print(f"\ndictionary reach : devices {100*dv['model_matched'].mean():.2f}%   "
          f"placements {100*j['model_matched'].mean():.2f}%")
    cov = {}
    for a in ATTRS:
        cov[a] = 100 * pdv["device_gid"].map(key[a]).notna().mean()
    print("\nper-attribute coverage over GPU placements (%):")
    for a in ATTRS:
        print(f"   {a:<20} {cov[a]:7.2f}")
    print(f"\nattributes filled per device (mean of {len(ATTRS)}): "
          f"{dv['attr_filled'].mean():.2f}  "
          f"hist={dv['attr_filled'].value_counts().sort_index().to_dict()}")
    return dv, gt, j, cov


def s2(dv, gt):
    print("=" * 100)
    print("S2  discriminative power: pairwise distinctness of the vectors")
    print("=" * 100)
    lab = dv.groupby("gpu_label", observed=True)[ATTRS].first()
    nfill = dv.groupby("gpu_label", observed=True)["attr_filled"].mean()
    lab = lab.assign(**{"attrs_filled": nfill})
    print(lab.to_string())
    labs = list(lab.index)
    print("\npairs that are NOT separated by any attribute both labels actually fill:")
    bad = []
    for i in range(len(labs)):
        for k in range(i + 1, len(labs)):
            a, b = lab.loc[labs[i]], lab.loc[labs[k]]
            shared = [c for c in ATTRS if pd.notna(a[c]) and pd.notna(b[c])]
            diff = [c for c in shared if a[c] != b[c]]
            if not diff:
                bad.append((labs[i], labs[k], len(shared)))
                print(f"   {labs[i]:<22} vs {labs[k]:<22} shared={len(shared)} "
                      f"separating={0}")
    if not bad:
        print("   (none -- every resolvable pair is separated)")
    print(f"\n{len(bad)} indistinguishable pairs out of "
          f"{len(labs)*(len(labs)-1)//2}")
    return lab


def s3(cov_a, cov_p, lab_a, lab_p):
    print("=" * 100)
    print("S3  cross-trace intersection of the attribute space")
    print("=" * 100)
    print(f"{'attribute':<20} {'ali%':>8} {'philly%':>8}   ali values / philly values")
    shared = []
    for a in ATTRS:
        va = {v for v in lab_a[a].dropna().unique()}
        vp = {v for v in lab_p[a].dropna().unique()}
        if cov_a.get(a, 0) > 0 and cov_p.get(a, 0) > 0:
            shared.append(a)
        print(f"{a:<20} {cov_a.get(a,0):8.2f} {cov_p.get(a,0):8.2f}   "
              f"{sorted(va)} / {sorted(vp)}"
              + ("   <- SHARED" if a in shared else ""))
    print(f"\nshared subspace ({len(shared)}/{len(ATTRS)} attributes): {shared}")
    for a in shared:
        va = {v for v in lab_a[a].dropna().unique()}
        vp = {v for v in lab_p[a].dropna().unique()}
        print(f"   {a:<18} value overlap = {sorted(va & vp)}"
              f"   (ali {sorted(va)} vs philly {sorted(vp)})")


def main():
    missing = [d for d in ("ali_v2020", "philly") if load(d, "devices") is None]
    if missing:
        sys.exit(f"no published devices table for {missing} under {R}")
    print("dictionary attributes:", ATTRS)
    print("sources:")
    for k, v in SOURCES.items():
        print(f"   {k:<11} {v}")
    out = {}
    for ds in ("ali_v2020", "philly"):
        out[ds] = s1(ds)
    lab_a, lab_p = s2(*out["ali_v2020"][:2]), s2(*out["philly"][:2])
    s3(out["ali_v2020"][3], out["philly"][3], lab_a, lab_p)


if __name__ == "__main__":
    main()
