# @Time    : 2026-06-12
# @File    : export_edge_weights.py
"""
为 xCAD 模型导出 r1_suits 边的 success_rate 权重。

【与 train.txt 的 ID 空间对齐 —— 重要】
train.txt/valid.txt/test.txt 是 export_to_regcn_format.py 用【局部字典】写出的:
  - src 走 algo_ids (job_name -> 局部 idx 0..N_algo-1)
  - r1/r2 的 dst 走 gpu_type_pool_ids (CPU/T4/MISC/P100/V100/V100M32 -> 0..5)
  - r3 的 dst 走 algo_ids
  - placement 的 dst 走 compute_ids
—— 而不是走全局 entity2id(带 4 段 offset)。

所以 r1_suits 边的 (s, o) 在 train.txt 里是【局部 idx】,不能被 entity2id 解释。
本脚本严格按 export_to_regcn_format.py 的方式重建局部字典:
  1. algo_ids: 读 algorithm_nodes.parquet 取 job_name unique() 顺序,索引 0..N_algo-1
  2. gpu_type_pool_ids: 固定顺序 ["CPU","T4","MISC","P100","V100","V100M32"] -> 0..5

输入:
  - xcad_code/xcad_model/data/xcad/{train,valid,test}.txt
  - xcad_code/outputs/nodes/algorithm_nodes.parquet
  - xcad_code/outputs/edges/r1_suits_edges.parquet

输出:
  - xcad_code/xcad_model/data/xcad/edge_weights_{train,valid,test}.npy
      shape=[N_split], dtype=float32
      weights[i] 对应 split 文件第 i 行:
        - rel_id==1 (r1_suits) 且 (s, o, tau) 在 r1_lookup 中 → success_rate
        - 其他情况 → 1.0 (中性默认, 不影响非 r1 边训练)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

# ---------- 路径 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XCAD_CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))  # .../xcad_code
DATA_DIR = os.path.join(XCAD_CODE_DIR, "xcad_model", "data", "xcad")
NODES_DIR = os.path.join(XCAD_CODE_DIR, "outputs", "nodes")
PARQUET_PATH = os.path.join(XCAD_CODE_DIR, "outputs", "edges", "r1_suits_edges.parquet")

SPLITS = ["train", "valid", "test"]
R1_REL_ID = 1
# 与 build_dgl_graph.GPU_TYPES / export_to_regcn_format.GPU_TYPES 完全一致
GPU_TYPES = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]


def build_algo_ids(nodes_dir):
    """重建 export_to_regcn_format.py 里的 algo_ids: job_name unique 顺序, 索引即局部 idx。
    严格一致(相同的 pd.read_parquet + dropna().unique().tolist())。"""
    algo_path = os.path.join(nodes_dir, "algorithm_nodes.parquet")
    algo_df = pd.read_parquet(algo_path)
    algo_unique = algo_df["job_name"].dropna().unique().tolist()
    return {pk: idx for idx, pk in enumerate(algo_unique)}, len(algo_unique)


def build_gpu_pool_ids():
    """固定 GPU 池局部 idx: CPU=0, T4=1, MISC=2, P100=3, V100=4, V100M32=5。"""
    return {gt: idx for idx, gt in enumerate(GPU_TYPES)}


def build_r1_lookup(parquet_path, algo_ids, gpu_pool_ids):
    """
    读 r1_suits parquet, 用局部字典映射:
      - src (job_name hash) -> algo_ids[hash] -> 局部 algo idx
      - dst (GPU 字符串)     -> gpu_pool_ids[str] -> 局部 GPU idx
    建 dict[(algo_local, gpu_local, tau)] = success_rate
    tau 必须参与 key (同对 (src, gpu) 在不同 tau 成功率不同, 已验证 9221 对 (src,dst) 跨多 tau)。
    """
    df = pd.read_parquet(parquet_path)
    lookup = {}
    miss = 0
    for src_str, dst_str, sr, tau in zip(
        df["src"].astype(str),
        df["dst"].astype(str),
        df["success_rate"].astype(float).values,
        df["tau"].astype(int).values,
    ):
        s = algo_ids.get(src_str)
        d = gpu_pool_ids.get(dst_str)
        if s is None or d is None:
            miss += 1
            continue
        lookup[(s, d, int(tau))] = float(sr)
    return lookup, miss


def export_split_weights(split_name, lookup):
    """
    读 split txt 逐行处理, 生成 edge_weights_{split}.npy。
    enumerate 与文件读序一致 -> weights[i] 与 split.txt 第 i 行对齐。
    train.txt 里 (s, o) 已经是局部 idx, 直接查 lookup 即可。
    """
    txt_path = os.path.join(DATA_DIR, f"{split_name}.txt")
    npy_path = os.path.join(DATA_DIR, f"edge_weights_{split_name}.npy")

    with open(txt_path, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)

    weights = np.ones(n_lines, dtype=np.float32)

    total = 0
    r1_total = 0
    r1_hit = 0
    r1_miss = 0  # r1 但 (s, o, tau) 不在 lookup 中 -> 保持 1.0

    # r1 命中权重的统计 (用于 r1 边 < 0.8 占比等)
    r1_lt08 = 0
    r1_w_min, r1_w_max, r1_w_sum = None, None, 0.0

    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            try:
                s, r, o, t = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                continue
            if r != R1_REL_ID:
                # 非 r1_suits: 保持 1.0
                continue
            r1_total += 1
            w = lookup.get((s, o, t))
            if w is not None:
                weights[i] = np.float32(w)
                r1_hit += 1
                r1_w_min = w if r1_w_min is None else min(r1_w_min, w)
                r1_w_max = w if r1_w_max is None else max(r1_w_max, w)
                r1_w_sum += w
                if w < 0.8:
                    r1_lt08 += 1
            else:
                r1_miss += 1

    np.save(npy_path, weights)
    return {
        "npy_path": npy_path,
        "weights": weights,
        "total": total,
        "r1_total": r1_total,
        "r1_hit": r1_hit,
        "r1_miss": r1_miss,
        "r1_lt08": r1_lt08,
        "r1_w_min": r1_w_min,
        "r1_w_max": r1_w_max,
        "r1_w_mean": (r1_w_sum / r1_hit) if r1_hit > 0 else None,
    }


def print_stats(split_name, stat, parquet_miss):
    w = stat["weights"]
    n = w.size
    n_lt08_all = int((w < 0.8).sum())
    pct_lt08_all = n_lt08_all / n * 100.0

    r1_total = stat["r1_total"]
    r1_hit = stat["r1_hit"]
    r1_miss = stat["r1_miss"]
    hit_rate = (r1_hit / r1_total * 100.0) if r1_total > 0 else 0.0

    print("=" * 78)
    print(f"[{split_name}] file={stat['npy_path']}")
    print(f"  total edges:            {stat['total']}")
    print(f"  r1_suits edges:         {r1_total}  (hit={r1_hit}, miss_dict={r1_miss})")
    print(f"  r1 hit rate:            {hit_rate:.4f}%  ({r1_hit}/{r1_total})")
    print(f"  weight array shape:     {w.shape}, dtype={w.dtype}")
    print(f"  weight min/max/mean:    {w.min():.6f} / {w.max():.6f} / {w.mean():.6f}")
    print(f"  rate<0.8 (全 split):    {n_lt08_all}/{n}  = {pct_lt08_all:.4f}%")
    if stat["r1_w_mean"] is not None:
        print(f"  r1 命中 weight min/max/mean: "
              f"{stat['r1_w_min']:.6f} / {stat['r1_w_max']:.6f} / {stat['r1_w_mean']:.6f}")
        pct_lt08_r1 = stat["r1_lt08"] / r1_hit * 100.0
        print(f"  r1 命中 < 0.8 占比:    {stat['r1_lt08']}/{r1_hit}  = {pct_lt08_r1:.4f}%")
    else:
        print(f"  r1 命中 0 条 (< 0.8 占比不适用)")
    print(f"  parquet 映射失败行数(全 split 累计): {parquet_miss}")
    print("=" * 78)


def main():
    if not os.path.isdir(NODES_DIR):
        sys.exit(f"nodes dir not found: {NODES_DIR}")
    if not os.path.exists(PARQUET_PATH):
        sys.exit(f"parquet not found: {PARQUET_PATH}")

    print(f"[build] algo_ids from {NODES_DIR}/algorithm_nodes.parquet")
    algo_ids, n_algo = build_algo_ids(NODES_DIR)
    print(f"  algo_ids size: {n_algo}")

    gpu_pool_ids = build_gpu_pool_ids()
    print(f"[build] gpu_type_pool_ids = {gpu_pool_ids}  (顺序固定)")

    print(f"[load] r1_suits parquet from {PARQUET_PATH}")
    lookup, miss = build_r1_lookup(PARQUET_PATH, algo_ids, gpu_pool_ids)
    print(f"  lookup size: {len(lookup)}  (parquet rows whose src/dst not in dict: {miss})")

    if miss > 0:
        warnings.warn(
            f"r1_suits parquet 有 {miss} 行 src/dst 映射不到局部字典, 已被 skip。"
            f"请检查 algorithm_nodes.parquet 与 GPU_TYPES 是否与 export 脚本一致。"
        )

    for split in SPLITS:
        stat = export_split_weights(split, lookup)
        print_stats(split, stat, miss)

    print("\n[done] all 3 npy written under", DATA_DIR)


if __name__ == "__main__":
    main()
