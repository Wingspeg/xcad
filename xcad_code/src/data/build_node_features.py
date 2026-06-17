#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_node_features.py

为 xCAD 三类节点构建类型感知特征矩阵(log1p + z-score 标准化).

段与编号严格对齐 export_to_regcn_format.py:
  - Algorithm段 [0, N_algo):  job_name unique, 按 algorithm_nodes.parquet 顺序
  - Data段     [0, N_data):   group unique
  - Compute段  [0, N_compute): machine unique

输出:
  xcad_code/xcad_model/data/xcad/node_feat_algo.npy   [N_algo, 13]  float32
  xcad_code/xcad_model/data/xcad/node_feat_data.npy   [N_data, 11]  float32
  xcad_code/xcad_model/data/xcad/node_feat_compute.npy [N_compute, 10] float32

GPU type 段(6个)不在此构建,由模型单独处理 embedding.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 路径锁定
_HERE = os.path.dirname(os.path.abspath(__file__))          # xcad_code/src/data/
_XCAD_CODE = os.path.dirname(os.path.dirname(_HERE))        # xcad_code/
NODES_DIR = os.path.join(_XCAD_CODE, "outputs", "nodes")    # xcad_code/outputs/nodes
OUT_DIR = os.path.join(_XCAD_CODE, "xcad_model", "data", "xcad")

# 确保输出目录存在
os.makedirs(OUT_DIR, exist_ok=True)

# 添加路径以便导入
if _XCAD_CODE not in sys.path:
    sys.path.insert(0, _XCAD_CODE)


# ============================================================
# 辅助: log1p + z-score 标准化
# ============================================================
def log1p_zscore(raw: np.ndarray, fill_nan: float = 0.0) -> np.ndarray:
    """对原始数值做 log1p 压缩后 z-score 标准化.

    参数:
        raw:       原始特征矩阵 [N, D]
        fill_nan:  在 log1p 之前用于填充 NaN 的值(默认 0.0)
    """
    x = np.array(raw, dtype=np.float64)
    x = np.nan_to_num(x, nan=fill_nan)
    x = np.log1p(x)                       # log(1+x) 压缩量级
    scaler = StandardScaler()
    x = scaler.fit_transform(x)           # z-score: mean≈0, std≈1
    return x.astype(np.float32)


# ============================================================
# 1. Algorithm 段特征
# ============================================================
def build_algo_features(nodes_dir: str) -> np.ndarray:
    """取数值特征,NaN填0(std的单实例NaN填0合理),log1p+zscore."""
    df = pd.read_parquet(os.path.join(nodes_dir, "algorithm_nodes.parquet"))

    # 严格按 job_name unique 顺序: 遍历保持首次出现顺序
    job_names = df["job_name"].dropna().unique().tolist()
    seen = set()
    ordered_jobs = []
    for jn in job_names:
        if jn not in seen:
            seen.add(jn)
            ordered_jobs.append(jn)

    # 取每 job 首条记录作为特征代表
    first_idx = {}
    for i, row in df.iterrows():
        jn = row["job_name"]
        if pd.notna(jn) and jn not in first_idx:
            first_idx[jn] = i

    # 13个数值特征列(avg/std/min/max x cpu/mem/gpu + instance_count)
    feature_cols = [
        "avg_plan_cpu", "std_plan_cpu", "min_plan_cpu", "max_plan_cpu",
        "avg_plan_mem", "std_plan_mem", "min_plan_mem", "max_plan_mem",
        "avg_plan_gpu", "std_plan_gpu", "min_plan_gpu", "max_plan_gpu",
        "instance_count",
    ]

    raw = np.zeros((len(ordered_jobs), len(feature_cols)), dtype=np.float64)
    for i, jn in enumerate(ordered_jobs):
        idx = first_idx[jn]
        row = df.loc[idx]
        raw[i] = [row[c] for c in feature_cols]

    return log1p_zscore(raw, fill_nan=0.0)


# ============================================================
# 2. Data 段特征
# ============================================================
def build_data_features(nodes_dir: str) -> np.ndarray:
    """取读写统计,NaN填0,log1p+zscore."""
    df = pd.read_parquet(os.path.join(nodes_dir, "data_nodes.parquet"))

    # 严格按 group unique 顺序
    groups = df["group"].dropna().unique().tolist()
    seen = set()
    ordered_groups = []
    for g in groups:
        if g not in seen:
            seen.add(g)
            ordered_groups.append(g)

    # 取每 group 首条记录
    first_idx = {}
    for i, row in df.iterrows():
        g = row["group"]
        if pd.notna(g) and g not in first_idx:
            first_idx[g] = i

    feature_cols = [
        "avg_read", "sum_read", "max_read",
        "avg_write", "sum_write", "max_write",
        "avg_read_count", "sum_read_count",
        "avg_write_count", "sum_write_count",
        "instance_count",
    ]  # 11 列

    raw = np.zeros((len(ordered_groups), len(feature_cols)), dtype=np.float64)
    for i, g in enumerate(ordered_groups):
        idx = first_idx[g]
        row = df.loc[idx]
        raw[i] = [row[c] for c in feature_cols]

    return log1p_zscore(raw, fill_nan=0.0)


# ============================================================
# 3. Compute 段特征
# ============================================================
def build_compute_features(nodes_dir: str) -> np.ndarray:
    """取机器能力与利用率,NaN填0,log1p(cap列)+zscore."""
    df = pd.read_parquet(os.path.join(nodes_dir, "compute_nodes.parquet"))

    # 严格按 machine unique 顺序
    machines = df["machine"].dropna().unique().tolist()
    seen = set()
    ordered_machines = []
    for m in machines:
        if m not in seen:
            seen.add(m)
            ordered_machines.append(m)

    # 取每 machine 首条记录
    first_idx = {}
    for i, row in df.iterrows():
        m = row["machine"]
        if pd.notna(m) and m not in first_idx:
            first_idx[m] = i

    feature_cols = [
        "machine_cap_cpu", "machine_cap_mem", "machine_cap_gpu",
        "gpu_type_ordinal",
        "avg_gpu_util", "max_gpu_util",
        "avg_gpu_mem",
        "avg_host_mem",
        "avg_cpu_usage",
        "avg_machine_load",
    ]  # 10 列

    raw = np.zeros((len(ordered_machines), len(feature_cols)), dtype=np.float64)
    for i, m in enumerate(ordered_machines):
        idx = first_idx[m]
        row = df.loc[idx]
        raw[i] = [row[c] for c in feature_cols]

    # CPU机器的GPU利用率NaN→填0(gpu_cap_mem本身已经是0)
    return log1p_zscore(raw, fill_nan=0.0)


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("build_node_features.py")
    print("=" * 60)

    # --- Algorithm ---
    print("\n[1/3] Algorithm nodes...")
    algo_feat = build_algo_features(NODES_DIR)
    n_algo = algo_feat.shape[0]
    print(f"  shape   : {algo_feat.shape}")
    print(f"  NaN残留  : {np.isnan(algo_feat).sum()}")
    print(f"  col min : {np.array2string(algo_feat.min(axis=0), precision=4, separator=', ')}")
    print(f"  col max : {np.array2string(algo_feat.max(axis=0), precision=4, separator=', ')}")
    print(f"  col mean: {np.array2string(algo_feat.mean(axis=0), precision=4, separator=', ')}")
    print(f"  col std : {np.array2string(algo_feat.std(axis=0), precision=4, separator=', ')}")

    # --- Data ---
    print("\n[2/3] Data nodes...")
    data_feat = build_data_features(NODES_DIR)
    n_data = data_feat.shape[0]
    print(f"  shape   : {data_feat.shape}")
    print(f"  NaN残留  : {np.isnan(data_feat).sum()}")
    print(f"  col min : {np.array2string(data_feat.min(axis=0), precision=4, separator=', ')}")
    print(f"  col max : {np.array2string(data_feat.max(axis=0), precision=4, separator=', ')}")
    print(f"  col mean: {np.array2string(data_feat.mean(axis=0), precision=4, separator=', ')}")
    print(f"  col std : {np.array2string(data_feat.std(axis=0), precision=4, separator=', ')}")

    # --- Compute ---
    print("\n[3/3] Compute nodes...")
    compute_feat = build_compute_features(NODES_DIR)
    n_compute = compute_feat.shape[0]
    print(f"  shape   : {compute_feat.shape}")
    print(f"  NaN残留  : {np.isnan(compute_feat).sum()}")
    print(f"  col min : {np.array2string(compute_feat.min(axis=0), precision=4, separator=', ')}")
    print(f"  col max : {np.array2string(compute_feat.max(axis=0), precision=4, separator=', ')}")
    print(f"  col mean: {np.array2string(compute_feat.mean(axis=0), precision=4, separator=', ')}")
    print(f"  col std : {np.array2string(compute_feat.std(axis=0), precision=4, separator=', ')}")

    # --- 保存 ---
    algo_path    = os.path.join(OUT_DIR, "node_feat_algo.npy")
    data_path    = os.path.join(OUT_DIR, "node_feat_data.npy")
    compute_path = os.path.join(OUT_DIR, "node_feat_compute.npy")

    np.save(algo_path,    algo_feat)
    np.save(data_path,    data_feat)
    np.save(compute_path, compute_feat)

    print(f"\nSaved to {OUT_DIR}:")
    print(f"  node_feat_algo.npy     {algo_feat.shape}    {os.path.getsize(algo_path)/1024:.1f} KB")
    print(f"  node_feat_data.npy     {data_feat.shape}    {os.path.getsize(data_path)/1024:.1f} KB")
    print(f"  node_feat_compute.npy  {compute_feat.shape}    {os.path.getsize(compute_path)/1024:.1f} KB")

    # --- 校验: 全局节点数应与 export_to_regcn_format.py 的 124891 一致 ---
    total_check = n_algo + n_data + n_compute + 6
    expected = 124891
    print(f"\nGlobal entity count check:")
    print(f"  N_algo({n_algo}) + N_data({n_data}) + N_compute({n_compute}) + 6 = {total_check}")
    print(f"  Expected: {expected}  {'OK' if total_check == expected else 'MISMATCH'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
