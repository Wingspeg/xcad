#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_to_regcn_format.py

把 xCAD 三要素时序图导出为 RE-GCN / CEN / RETIA 通用的磁盘格式:
    train.txt / valid.txt / test.txt   (每行: s_id r_id o_id tau, 整数, 制表符分隔)
    stat.txt                           (一行: num_entities num_relations)
    entity2id.txt / relation2id.txt    (名字 -> id 映射, 制表符分隔)

切分与关系选择严格对齐 temporal_split.py(保证与已得 RE-GCN baseline 0.0879 可比):
    train tau[1,49] / val[50,55] / test[56,61]
    关系: placement, r1_suits, r2_requires, r3_drives   (排除 r4_shifts)
    退化设定: 丢弃 success_rate 连续边权, 边存在即为一条三元组 (二元化)

节点统一编号(方式2, 与 build_dgl_graph 完全对齐):
    四段全局 id 空间, 顺序固定:
        [0 .. N_algo)             algorithm_nodes.parquet  -> job_name
        [N_algo .. N_algo+N_data) data_nodes.parquet      -> group
        [... .. +N_machine)       compute_nodes.parquet   -> machine
        [... .. +6)               6 个 gpu_type 枚举值
    任何一条边, 只要 src/dst 在其对应字典里查不到(无标签 job、被
    MIN_GROUP_SIZE 过滤掉的 group 等), 直接 drop, 与 build_dgl_graph
    的 dropna 行为完全一致. 这是 0.0879 那次 baseline 的真实图空间.

用法:
    python export_to_regcn_format.py \
        --edges-dir xcad_code/outputs/edges \
        --nodes-dir xcad_code/outputs/nodes \
        --out-dir   baselines/data/xcad
"""

import os
import sys
import argparse
import logging
from typing import Dict, Tuple
import pandas as pd

# ---- 让脚本能从任意 cwd 直接跑 ----
# 路径锁定: 与 build_dgl_graph.py 写入的实际目录一致
# __file__ = xcad_code/src/data/export_to_regcn_format.py
_HERE = os.path.dirname(os.path.abspath(__file__))          # xcad_code/src/data/
_XCAD_CODE = os.path.dirname(os.path.dirname(_HERE))        # xcad_code/
OUTPUT_ROOT = os.path.join(_XCAD_CODE, "outputs")           # xcad_code/outputs

# 让其他模块(如 config)能以 src.xxx 形式被导入
if _XCAD_CODE not in sys.path:
    sys.path.insert(0, _XCAD_CODE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- 与 temporal_split.py / build_dgl_graph.py 完全一致的配置 ----
EDGE_TYPES = ["placement", "r1_suits", "r2_requires", "r3_drives"]  # r4_shifts 排除
UNIFIED_SPLIT = {
    "train": (1, 49),
    "val":   (50, 55),
    "test":  (56, 61),
}
# 关系 id 固定顺序(决定 relation2id 编号; 与 EDGE_TYPES 顺序一致)
RELATION_ORDER = EDGE_TYPES

# 与 build_dgl_graph.GPU_TYPES 完全一致
GPU_TYPES = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]


# ============================================================
# 1. 四段全局 entity 编号空间(从节点表建立, 与 build_dgl_graph 对齐)
# ============================================================
def build_entity_space(nodes_dir: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], int]:
    """加载三个节点表 + GPU 类型枚举, 拼成统一的 entity2id 全集.

    返回:
        entity2id:        全局主字典 {原始字符串 -> 全局 id}, 长度 = num_entities
        algo_ids:         算法段局部字典 {job_name -> 段内 id 0..N_algo-1}
        data_ids:         数据段局部字典 {group -> 段内 id 0..N_data-1}
        compute_ids:      机器段局部字典 {machine -> 段内 id 0..N_machine-1}
        gpu_type_pool_ids:GPU 池段局部字典 {gpu_type_name -> 段内 id 0..5}
    """
    logger.info("Building 4-segment entity space from node tables...")

    # --- 段 1: algorithm_nodes.parquet -> job_name ---
    algo_path = os.path.join(nodes_dir, "algorithm_nodes.parquet")
    algo_df = pd.read_parquet(algo_path)
    # 与 build_dgl_graph.load_algorithm_and_data_nodes 完全一致: 用 unique() 而非整行
    algo_unique = algo_df["job_name"].dropna().unique().tolist()
    algo_ids = {pk: idx for idx, pk in enumerate(algo_unique)}
    n_algo = len(algo_ids)
    logger.info(f"  segment 1 [algorithm]: {n_algo:,} nodes (job_name unique)")

    # --- 段 2: data_nodes.parquet -> group ---
    data_path = os.path.join(nodes_dir, "data_nodes.parquet")
    data_df = pd.read_parquet(data_path)
    data_unique = data_df["group"].dropna().unique().tolist()
    data_ids = {pk: idx for idx, pk in enumerate(data_unique)}
    n_data = len(data_ids)
    logger.info(f"  segment 2 [data]:      {n_data:,} nodes (group unique)")

    # --- 段 3: compute_nodes.parquet -> machine ---
    compute_path = os.path.join(nodes_dir, "compute_nodes.parquet")
    compute_df = pd.read_parquet(compute_path)
    compute_unique = compute_df["machine"].dropna().unique().tolist()
    compute_ids = {pk: idx for idx, pk in enumerate(compute_unique)}
    n_compute = len(compute_ids)
    logger.info(f"  segment 3 [compute]:   {n_compute:,} nodes (machine unique)")

    # --- 段 4: GPU 类型枚举 ---
    gpu_type_pool_ids = {gt: idx for idx, gt in enumerate(GPU_TYPES)}
    n_gpu = len(gpu_type_pool_ids)
    logger.info(f"  segment 4 [gpu_type]:  {n_gpu} nodes (enum {GPU_TYPES})")

    # --- 拼全局 entity2id, 用累计 offset ---
    offset_algo = 0
    offset_data = offset_algo + n_algo
    offset_compute = offset_data + n_data
    offset_gpu = offset_compute + n_compute

    entity2id: Dict[str, int] = {}
    for pk, idx in algo_ids.items():
        entity2id[pk] = offset_algo + idx
    for pk, idx in data_ids.items():
        entity2id[pk] = offset_data + idx
    for pk, idx in compute_ids.items():
        entity2id[pk] = offset_compute + idx
    for pk, idx in gpu_type_pool_ids.items():
        entity2id[pk] = offset_gpu + idx

    num_entities = len(entity2id)
    logger.info(f"  >>> num_entities = {num_entities:,} "
                f"(algo {n_algo} + data {n_data} + compute {n_compute} + gpu {n_gpu})")
    if num_entities != 124891:
        logger.warning(f"  WARNING: num_entities={num_entities} != expected 124891. "
                       f"Check node tables / GPU_TYPES list.")

    return entity2id, algo_ids, data_ids, compute_ids, gpu_type_pool_ids


# ============================================================
# 2. 边映射: 按 edge_type 选字典, dropna 对齐 build_dgl_graph
# ============================================================
def map_edge_columns(
    edges_dir: str,
    algo_ids: Dict[str, int],
    data_ids: Dict[str, int],
    compute_ids: Dict[str, int],
    gpu_type_pool_ids: Dict[str, int],
) -> Dict[str, pd.DataFrame]:
    """读取 4 类边 parquet, 按边类型分别选字典做 src/dst 映射.

    placement:   src=algo_ids,        dst=compute_ids        (machine 段)
    r1_suits:    src=algo_ids,        dst=gpu_type_pool_ids  (GPU 池段)
    r2_requires: src=algo_ids,        dst=gpu_type_pool_ids
    r3_drives:   src=data_ids,        dst=algo_ids

    映射后 src_id 或 dst_id 为 NaN 的行直接 drop(对齐 build_dgl_graph).
    """
    logger.info("Mapping edge columns with per-type dictionaries...")

    EDGE_DICT_PLAN = {
        "placement":   ("algo_ids",            "compute_ids"),
        "r1_suits":    ("algo_ids",            "gpu_type_pool_ids"),
        "r2_requires": ("algo_ids",            "gpu_type_pool_ids"),
        "r3_drives":   ("data_ids",            "algo_ids"),
    }
    DICTS = {
        "algo_ids": algo_ids,
        "data_ids": data_ids,
        "compute_ids": compute_ids,
        "gpu_type_pool_ids": gpu_type_pool_ids,
    }

    out: Dict[str, pd.DataFrame] = {}
    drop_stats = {}

    for et in EDGE_TYPES:
        path = os.path.join(edges_dir, f"{et}_edges.parquet")
        df = pd.read_parquet(path)
        df = df[["src", "dst", "tau"]].copy()
        df["edge_type"] = et
        df = df[df["tau"] >= 1]
        n_in = len(df)

        src_dict_name, dst_dict_name = EDGE_DICT_PLAN[et]
        df["s"] = df["src"].map(DICTS[src_dict_name])
        df["o"] = df["dst"].map(DICTS[dst_dict_name])

        n_drop = int(df[["s", "o"]].isna().any(axis=1).sum())
        df = df.dropna(subset=["s", "o"])
        df["s"] = df["s"].astype(int)
        df["o"] = df["o"].astype(int)
        n_keep = len(df)

        drop_stats[et] = (n_in, n_drop, n_keep)
        out[et] = df
        logger.info(f"  {et:12s}: in={n_in:>9,}  drop={n_drop:>9,}  keep={n_keep:>9,}  "
                    f"(src={src_dict_name}, dst={dst_dict_name})")

    logger.info("Edge drop/keep summary (对齐 build_dgl_graph 的 dropna 行为):")
    for et, (n_in, n_drop, n_keep) in drop_stats.items():
        logger.info(f"  {et:12s}: in={n_in:>9,}  drop={n_drop:>9,}  keep={n_keep:>9,}")

    return out


# ============================================================
# 3. 关系编号 + 切分 + 去重
# ============================================================
def build_relation2id():
    relation2id = {r: i for i, r in enumerate(RELATION_ORDER)}
    logger.info(f"relation2id = {relation2id}")
    return relation2id


def split_and_encode(edges_by_type, relation2id):
    """按统一 tau 边界切分, 编码成整数四元组 (s r o t), 按 (s,r,o,tau) 去重."""
    out = {}
    for split, (lo, hi) in UNIFIED_SPLIT.items():
        frames = []
        for et, df in edges_by_type.items():
            sub = df[(df["tau"] >= lo) & (df["tau"] <= hi)].copy()
            sub["r"] = relation2id[et]
            frames.append(sub[["s", "r", "o", "tau"]])
        quad = pd.concat(frames, ignore_index=True).astype(int)
        before = len(quad)
        quad = quad.drop_duplicates()
        out[split] = quad
        logger.info(f"  {split} tau[{lo},{hi}]: {len(quad):,} quads (去重前 {before:,})")
    return out


# ============================================================
# 4. 写出 RE-GCN 格式
# ============================================================
def write_outputs(out_dir, splits, entity2id, relation2id, num_entities_expected):
    os.makedirs(out_dir, exist_ok=True)

    name_map = {"train": "train.txt", "val": "valid.txt", "test": "test.txt"}
    for split, fname in name_map.items():
        quad = splits[split]
        # RE-GCN 的 split_by_time 要求按时间顺序排列: 主键 tau 升序, 次级键 s 升序
        quad = quad.sort_values(by=["tau", "s"], kind="mergesort").reset_index(drop=True)
        quad.to_csv(os.path.join(out_dir, fname), sep="\t", header=False, index=False)
        logger.info(f"wrote {fname}: {len(quad):,} rows")

    # stat.txt : num_entities num_relations
    # 与节点表锁死的 124,891 实体数 + 4 类边, 严格对齐 build_dgl_graph
    with open(os.path.join(out_dir, "stat.txt"), "w") as f:
        f.write(f"{num_entities_expected}\t{len(relation2id)}")
    logger.info(f"wrote stat.txt: {num_entities_expected}\t{len(relation2id)}")

    # entity2id.txt / relation2id.txt
    with open(os.path.join(out_dir, "entity2id.txt"), "w") as f:
        for e, i in entity2id.items():
            f.write(f"{e}\t{i}\n")
    with open(os.path.join(out_dir, "relation2id.txt"), "w") as f:
        for r, i in relation2id.items():
            f.write(f"{r}\t{i}\n")
    logger.info("wrote entity2id.txt / relation2id.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges-dir", default=os.path.join(OUTPUT_ROOT, "edges"))
    ap.add_argument("--nodes-dir", default=os.path.join(OUTPUT_ROOT, "nodes"))
    ap.add_argument("--out-dir", default="baselines/data/xcad")
    ap.add_argument("--expected-entities", type=int, default=124891,
                    help="期望的实体总数(与节点表锁死的 124,891 对齐; 不一致会告警但仍写出)")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("Export xCAD graph -> RE-GCN/CEN/RETIA format")
    logger.info(f"edges-dir: {args.edges_dir}")
    logger.info(f"nodes-dir: {args.nodes_dir}")
    logger.info(f"out-dir:   {args.out_dir}")
    logger.info("=" * 60)

    # 1. 节点表 -> 四段全局 entity2id
    entity2id, algo_ids, data_ids, compute_ids, gpu_type_pool_ids = build_entity_space(args.nodes_dir)

    # 2. 边按 edge_type 选字典映射 + dropna
    edges_by_type = map_edge_columns(
        args.edges_dir, algo_ids, data_ids, compute_ids, gpu_type_pool_ids
    )

    # 3. 关系字典
    relation2id = build_relation2id()

    # 4. 切分 + 去重
    splits = split_and_encode(edges_by_type, relation2id)

    # 5. 写出
    write_outputs(args.out_dir, splits, entity2id, relation2id, args.expected_entities)

    logger.info("=" * 60)
    logger.info("DONE. 验证:")
    logger.info(f"  num_entities={len(entity2id):,}  num_relations={len(relation2id)}")
    logger.info(f"  train={len(splits['train']):,} val={len(splits['val']):,} test={len(splits['test']):,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
