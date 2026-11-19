"""How much of the categorical GPU label survives as a MISSINGNESS pattern?

The proposal under test is "map Ali's model names to capability specs so that
model name stops being a dataset-specific categorical key".  This script does
the one measurement that decides it: encode each device as the binary mask of
which spec attributes are filled, then ask how well the original gpu_label can
be recovered from the mask ALONE.  If recovery is near-perfect, the mapping did
not remove the label -- it moved it from the value channel to the missingness
channel, which is a channel every dataset fills differently.

Also reported: whether the two traces share ANY missingness pattern, i.e.
whether an NA-mask-aware model would see overlapping capability classes or two
disjoint sets.

usage: ~/xcad/xcad_code/.venv/bin/python scripts/probe_spec_mask.py
"""
import os
import pandas as pd

R = os.path.expanduser("~/xcad/datasets/clean")
ATTRS = ["n_gpus_per_node", "mem_gib", "mem_bw_gbs", "fp32_tflops",
         "fp64_tflops", "fp16_tflops", "tensor_tflops", "cuda_cores",
         "compute_capability", "has_tensor_core", "has_nvlink", "tdp_w"]

# label -> the dictionary attributes that label resolves to (same dictionary as
# probe_spec_join.py; n_gpus_per_node is filled from each trace's own inventory).
ALI_SPEC = {
    "NVIDIA-P100":       dict(mem_gib=16, mem_bw_gbs=732, fp32_tflops=9.3, fp64_tflops=4.7,
                              fp16_tflops=18.7, compute_capability=6.0, has_tensor_core=0,
                              has_nvlink=1),
    "NVIDIA-V100":       dict(mem_gib=16, mem_bw_gbs=900, fp32_tflops=15.7, fp64_tflops=7.0,
                              tensor_tflops=112, cuda_cores=5120, compute_capability=7.0,
                              has_tensor_core=1, has_nvlink=1, tdp_w=250),
    "NVIDIA-V100-32G":   dict(mem_gib=32, mem_bw_gbs=900, fp32_tflops=15.7, fp64_tflops=7.0,
                              tensor_tflops=112, cuda_cores=5120, compute_capability=7.0,
                              has_tensor_core=1, has_nvlink=1, tdp_w=250),
    "NVIDIA-T4":         dict(mem_gib=16, mem_bw_gbs=300, fp32_tflops=8.1, fp16_tflops=65,
                              cuda_cores=2560, compute_capability=7.5, has_tensor_core=1,
                              has_nvlink=0, tdp_w=70),
    "NVIDIA-LEGACY-MISC": {},
    "UNRESOLVED":         {},
}
PHILLY_SPEC = {"PHILLY-TIER-2x12": dict(mem_gib=12),
               "PHILLY-TIER-8x24": dict(mem_gib=24)}


def maskstr(d):
    # n_gpus_per_node is filled from each trace's own device inventory, not from
    # the dictionary, so it is always present in both datasets.
    return "1" + "".join("1" if d.get(a) is not None else "0"
                         for a in ATTRS[1:])


def run(ds, spec):
    print("=" * 78)
    print(f"{ds}: missingness mask as a label fingerprint")
    print("=" * 78)
    dv = pd.read_parquet(os.path.join(R, ds, "devices.parquet"))
    pdv = pd.read_parquet(os.path.join(R, ds, "placement_devices.parquet"))
    lab = dv.set_index("device_gid")["gpu_label"]
    cnt = dv.groupby("machine_gid", observed=True)["device_gid"].transform("size")
    node_n = dv.assign(n=cnt).set_index("device_gid")["n"]
    pdv = pdv.assign(gpu_label=pdv["device_gid"].map(lab),
                     n_obs=pdv["device_gid"].map(node_n))
    pdv = pdv[pdv["gpu_label"].notna()].copy()
    pdv["mask"] = pdv["gpu_label"].map(lambda l: maskstr(spec.get(l, {})))
    tot = len(pdv)
    g = pdv.groupby("mask", observed=True)
    tbl = pd.DataFrame({"placements": g.size(),
                        "labels": g["gpu_label"].nunique(),
                        "top_placements": g["gpu_label"].agg(lambda s: s.value_counts().max()),
                        "labels_in_mask": g["gpu_label"].agg(
                            lambda s: "|".join(sorted(s.unique())))})
    tbl["share_pct"] = 100 * tbl["placements"] / tot
    print(tbl.to_string())
    acc = 100 * tbl["top_placements"].sum() / tot
    print(f"\ndistinct masks={pdv['mask'].nunique()}   labels={pdv['gpu_label'].nunique()}")
    print(f"label recoverable from the mask ALONE = {acc:.2f}%")
    sub = pdv.groupby("gpu_label", observed=True).agg(
        mask=("mask", "first"), n_obs_typical=("n_obs", lambda s: sorted(s.unique())),
        placements=("mask", "size"))
    sub["mem_gib"] = [spec.get(l, {}).get("mem_gib") for l in sub.index]
    sub["share_pct"] = 100 * sub["placements"] / tot
    print("\nper label:")
    print(sub.to_string())
    return set(pdv["mask"].unique()), acc


pa, aa = run("ali_v2020", ALI_SPEC)
pp, ap = run("philly", PHILLY_SPEC)

print("=" * 78)
print("cross-dataset: do the two traces share a missingness pattern?")
print("=" * 78)
print(f"ali masks     : {sorted(pa)}")
print(f"philly masks  : {sorted(pp)}")
print(f"shared masks  : {sorted(pa & pp)}")
print(f"mask-only label accuracy: ali {aa:.2f}%   philly {ap:.2f}%")
