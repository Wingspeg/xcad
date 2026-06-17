"""
export_node_features.py
为 xCAD RE-GCN 模型导出节点 IO 特征。

- 读取 data_nodes.parquet，取 group + avg_read + avg_write + instance_count
- 读取 baselines/data/xcad/entity2id.txt，建立 group hash → 全局 entity id 映射
  Data 段位于 [102610, 122987]
- 对 Data 节点原始值做 log1p，再在 Data 行上单独计算 z-score 标准化
  (非 Data 行不参与统计，以免 0 拉偏均值/方差)
- 输出 baselines/data/xcad/node_features.npy，形状 [124891, 3]，dtype float32
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path("/Volumes/code/xcad")
ENTITY2ID_PATH = ROOT / "baselines/data/xcad/entity2id.txt"
DATA_NODES_PATH = ROOT / "xcad_code/outputs/nodes/data_nodes.parquet"
OUTPUT_PATH = ROOT / "baselines/data/xcad/node_features.npy"

# ---------- 全局常量 ----------
GLOBAL_NUM_ENTITIES = 124_891   # entity2id.txt 总行数
DATA_START = 102_610             # Data 节点全局 id 下界 (inclusive)
DATA_END   = 122_987             # Data 节点全局 id 上界 (inclusive)
FEAT_DIM   = 3                   # [avg_read, avg_write, instance_count]
DTYPE      = np.float32

# ---------- Step 1: 读 entity2id，建立 hash → global_id ----------
entity2id = {}
with open(ENTITY2ID_PATH, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        entity_hash, idx = parts[0], int(parts[1])
        entity2id[entity_hash] = idx

# ---------- Step 2: 读 data_nodes.parquet ----------
df = pd.read_parquet(DATA_NODES_PATH)
df = df[["group", "avg_read", "avg_write", "instance_count"]].copy()

# ---------- Step 3: 过滤只在 entity2id 中出现且落在 Data 段的行 ----------
df["global_id"] = df["group"].map(entity2id)
df = df.dropna(subset=["global_id"])
df["global_id"] = df["global_id"].astype(int)
df_in_data_range = df[
    (df["global_id"] >= DATA_START) & (df["global_id"] <= DATA_END)
].copy()

data_node_count = len(df_in_data_range)
print(f"[INFO] Data 节点在 entity2id 范围内且 id in [{DATA_START}, {DATA_END}]: {data_node_count}")

# ---------- Step 4: log1p 压缩 ----------
raw_feats = df_in_data_range[["avg_read", "avg_write", "instance_count"]].values.astype(np.float64)
log_feats = np.log1p(raw_feats)          # [N_data, 3]

# ---------- Step 5: 在 Data 行上单独计算 z-score ----------
mu  = log_feats.mean(axis=0)              # [3,]
std = log_feats.std(axis=0)               # [3,]
std[std == 0] = 1.0                       # 避免除零

z_feats = (log_feats - mu) / std          # [N_data, 3]

print(f"[INFO] z-score 参数 (mean/std):")
for i, col in enumerate(["avg_read", "avg_write", "instance_count"]):
    print(f"       {col}: mean={mu[i]:.6f}, std={std[i]:.6f}")

# ---------- Step 6: 构建完整特征矩阵 [124891, 3] ----------
full_feats = np.zeros((GLOBAL_NUM_ENTITIES, FEAT_DIM), dtype=DTYPE)

# 把 z-score 后的 Data 行写回对应全局 id
global_ids = df_in_data_range["global_id"].values.astype(int)
for row_idx, gid in enumerate(global_ids):
    full_feats[gid] = z_feats[row_idx]

# ---------- Step 7: 完整性校验 ----------
assert full_feats.shape == (GLOBAL_NUM_ENTITIES, FEAT_DIM), \
    f"形状错误: {full_feats.shape}"
assert full_feats.dtype == DTYPE, f"dtype 错误: {full_feats.dtype}"
assert not np.any(np.isnan(full_feats)), "存在 NaN!"
assert not np.any(np.isinf(full_feats)), "存在 Inf!"

# 各列 min/max（全局，含 0 行）
for i, col in enumerate(["avg_read", "avg_write", "instance_count"]):
    col_min = float(full_feats[:, i].min())
    col_max = float(full_feats[:, i].max())
    print(f"[INFO] 全局列 {col} min={col_min:.6f}, max={col_max:.6f}")

print(f"[INFO] 特征矩阵形状: {full_feats.shape}")
print(f"[INFO] dtype: {full_feats.dtype}")

# ---------- Step 8: 保存 ----------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT_PATH, full_feats)
print(f"[DONE] 已保存 -> {OUTPUT_PATH}")
