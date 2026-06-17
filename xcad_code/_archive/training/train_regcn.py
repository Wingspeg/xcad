"""
xCAD RE-GCN Full Training (Stage 3.1)

充分压榨 RTX 3090 + i7-13700K + 125 GB RAM:
  - 强制 CUDA(无 GPU 直接报错)
  - AMP 混合精度 (FP16) + GradScaler
  - Cosine LR + Warmup
  - Checkpoint:每 5 epoch 保存,只留最近 3 个 + best val MRR
  - Early stopping (patience=7 on val MRR)
  - GPU 显存 / 系统 RAM 峰值监控
  - 报告输出: ${OUTPUT_ROOT}/reports/stage3.1_full.md

差异化时间窗切分(与阶段二 SPLIT_CONFIG 对齐):
  - r1/r2/r3: train [1, 50] / val [51, 55] / test [56, 62]
  - placement: train [1, 60] / val [61, 65] / test [66, 69]
  - r4_shifts:不参与本次训练(留待消融)

运行:
    cd /home/leosue/xcad/xcad_code
    source .venv/bin/activate
    python -m src.training.train_regcn_full
"""

import os
import sys
import time
import glob
import random
import logging
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 抗显存碎片:让 PyTorch 的 caching allocator 用 expandable segments,
# 避免 22GB 占用但只腾不出 1GB 的情况。必须在 import torch 之前设。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import dgl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "..", "xcad_code"))

from src.utils.config import OUTPUT_ROOT, RANDOM_SEED
from src.utils.data_loader import load_xcad_dataset
from src.models.regcn import REGCNModel


# =============================================================================
# Logging
# =============================================================================
LOG_DIR = Path(OUTPUT_ROOT) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "train_regcn_full.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Output paths
# =============================================================================
REPORT_DIR = Path(OUTPUT_ROOT) / "reports"
CHECKPOINT_DIR = Path(OUTPUT_ROOT) / "checkpoints"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Configuration(全量训练,跟阶段二 SPLIT_CONFIG 对齐)
# =============================================================================
EDGE_TYPES = ["r1_suits", "r2_requires", "r3_drives", "placement"]  # r4_shifts 不进
RELATION_TO_ID = {rel: i for i, rel in enumerate(EDGE_TYPES)}
RELATION_TO_ID.update({f"{rel}_inv": i + len(EDGE_TYPES) for i, rel in enumerate(EDGE_TYPES)})
NUM_RELATIONS = len(RELATION_TO_ID)  # 8 (4 正向 + 4 反向)

# 与 build_dgl_graph.py 一致(只影响 gpu_type_pool 的子集,不在 EDGE_TYPES 内用)
GPU_TYPES = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]

# 异构图扁平化时,边端点类型映射(与 build_dgl_graph.py RELATION_SCHEMA 一致)
EDGE_TYPE_TO_SRC_DST = {
    "r1_suits":     ("algorithm", "compute"),  # dst 实际是 gpu_type_pool,在 compute 空间内
    "r2_requires":  ("algorithm", "compute"),
    "r3_drives":    ("data",      "algorithm"),
    "placement":    ("algorithm", "compute"),
}

# 时间窗上界(只筛边,不动 split parquet 内容;以时间窗 mask 在数据加载时过滤)
MAX_TIME_WINDOWS_R123 = 62
MAX_TIME_WINDOWS_PLACEMENT = 69

# 训练超参(对标 RE-GCN 论文标配 + 3090 显存 24GB)
# HIDDEN_DIM 改为 100 而非论文标配的 200:RGCN 的 msg_func 会按边数 N 物化
# [N, h_dim, h_dim] 的 per-edge 权重张量。在 no_grad + detach 架构下,RGCN
# 60 张图的计算图不再驻留显存,峰值由"单张图"决定:τ=1 的 placement 图 ~125K 条
# 边。RGCN 带梯度前向(每 100 batch refresh)后,per-edge w 的显存开销跟 h_dim²
# 成正比,h_dim=100 时单图 ≈ 2.5 GB,加上 60 张图的计算图驻留会 OOM。降到 64
# (100² → 64² ≈ 41%)后单图 per-edge w ≈ 1 GB,腾出空间给 60 张图计算图。
# h_dim=200 会让单图峰值 ~16GB 超限。
HIDDEN_DIM = 64
NUM_LAYERS = 2
BATCH_SIZE = 256
NEG_SAMPLE_RATIO = 5
# 原版 RE-GCN 训练循环:每个时间步做一次完整 forward + backward,RGCN 和 decoder
# 同步更新。每个时间步用前 HISTORY_LEN 张图作为 history(原版 src/main.py 的
# train_sample_num 上界)。如果当前时间步的 history 图不足 HISTORY_LEN 张,
# 自动从 τ=1 开始取(代码里 hist_start = max(0, t - HISTORY_LEN))。
HISTORY_LEN = 10
MAX_TRIPLES_PER_STEP = 2048  # 每时间步最多采样多少条正例三元组,避免 OOM
LEARNING_RATE = 3e-4
FINAL_LR = 1e-5
# warmup 第 0 个 epoch 的起始 lr(不再是 LEARNING_RATE/warmup_epochs,改成显式 5e-5)
WARMUP_START_LR = 5e-5
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
GRAD_CLIP = 1.0
DROPOUT = 0.2
WARMUP_EPOCHS = 5
LR_SCHEDULER = "cosine"  # cosine annealing,from LR to FINAL_LR
# 硬件利用
DATALOADER_NUM_WORKERS = 8
DATALOADER_PIN_MEMORY = True
AMP_ENABLED = True

# 日志 / Checkpoint
LOG_EVERY_N_BATCHES = 50
SAVE_CHECKPOINT_EVERY_N_EPOCHS = 5
KEEP_LAST_N_CHECKPOINTS = 3

# Chunked cross-entropy:避免 [B, num_nodes=124,891] 的 log_softmax 中间值 OOM
# 单次 cross_entropy 默认会物化整张 log_softmax(F=11,264 行 × 124,891 列 × 4B FP32
# ≈ 5.6 GB),跟 RGCN 计算图加在一起直接打爆 24GB 显存。chunk_size=1024 时每块
# 峰值 ≈ 500MB(1024 × 124,891 × 4B FP32),留更多余量给 RGCN 计算图。
CE_CHUNK_SIZE = 1024

# Chunked evaluate:val/test 一次性物化 [2*max_eval_samples=10000, 124891] 的
# scores,峰值约 5 GB;raw + filt 两次打分再翻倍到 10 GB。分块后每块 [chunk, 124891],
# chunk=2500 时 ~2.5 GB(filtered clone 后 ~5 GB),跟 RGCN 一起可控。
EVAL_CHUNK_SIZE = 2500

# 设备(GPU 强制)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    raise RuntimeError(
        "Full training requires CUDA. Use train_regcn_smoke.py for CPU testing."
    )


# =============================================================================
# 数据加载与处理
# =============================================================================
def build_node_offsets(dataset) -> Tuple[Dict[str, int], int]:
    """根据 meta['node_counts'] 计算异构 → 同构的全局节点偏移。

    Returns:
        offsets: {node_type: global_offset}
        num_total: 全局节点总数
    """
    counts = dataset.meta["node_counts"]
    offsets: Dict[str, int] = {}
    running = 0
    for node_type in ["algorithm", "compute", "data"]:
        offsets[node_type] = running
        running += int(counts.get(node_type, 0))
    return offsets, running


def build_node_id_mappings() -> Dict[str, Dict[str, int]]:
    """从原始 nodes/*.parquet 复现 build_dgl_graph.py 里的字符串→整数 ID 映射。

    返回 dict 含 4 个子 dict:
        algo_ids          {job_name: idx}        0..N_algo-1
        compute_ids       {machine:  idx}        0..N_machines-1
        gpu_type_pool_ids {gpu_type: idx}        N_machines..N_machines+5
        data_ids          {group:    idx}        0..N_data-1

    调用方需保证 train_regcn_full.py 与 build_dgl_graph.py 读到的 nodes parquet
    行序一致(unique() 的顺序由首次出现位置决定),否则 ID 会错位。
    """
    node_dir = Path(OUTPUT_ROOT) / "nodes"
    algo_df = pd.read_parquet(node_dir / "algorithm_nodes.parquet")
    algo_ids = {pk: idx for idx, pk in enumerate(algo_df["job_name"].unique())}

    data_df = pd.read_parquet(node_dir / "data_nodes.parquet")
    data_ids = {pk: idx for idx, pk in enumerate(data_df["group"].unique())}

    compute_df = pd.read_parquet(node_dir / "compute_nodes.parquet")
    num_machines = len(compute_df)
    compute_ids = {pk: idx for idx, pk in enumerate(compute_df["machine"].unique())}
    gpu_type_pool_ids = {gt: num_machines + i for i, gt in enumerate(GPU_TYPES)}

    logger.info(
        f"  ID mappings: algo={len(algo_ids):,}, compute={len(compute_ids):,}, "
        f"gpu_type_pool={len(gpu_type_pool_ids)}, data={len(data_ids):,}"
    )
    return {
        "algo_ids": algo_ids,
        "compute_ids": compute_ids,
        "gpu_type_pool_ids": gpu_type_pool_ids,
        "data_ids": data_ids,
    }


def map_edges_to_global(
    df: pd.DataFrame,
    mappings: Dict[str, Dict[str, int]],
    global_offsets: Dict[str, int],
) -> pd.DataFrame:
    """把 split DataFrame 的 src/dst 字符串列映射到全局同构 ID。

    输入 df 来自 train_edges.parquet 等,列至少含:
        src, dst, tau, edge_type
    输出新增列:src_global, dst_global, rel_id

    注:
    - parquet 内可能含 r4_shifts(temporal_split 按 80/10/10 随机切了),但本次训练
      只用 EDGE_TYPES 内的 4 类,这里直接在入口过滤掉 r4,避免下游 KeyError。
    - r1/r2 的 dst 在 build_dgl_graph.py 里走 gpu_type_pool_ids(不是 compute_ids),
      需要按 edge_type 区分。
    """
    df = df[df["edge_type"].isin(EDGE_TYPES)].copy()
    if len(df) == 0:
        return df

    algo_ids = mappings["algo_ids"]
    compute_ids = mappings["compute_ids"]
    gpu_type_pool_ids = mappings["gpu_type_pool_ids"]
    data_ids = mappings["data_ids"]

    n = len(df)
    src_int = np.full(n, -1, dtype=np.int64)
    dst_int = np.full(n, -1, dtype=np.int64)

    for et in df["edge_type"].unique():
        if et not in EDGE_TYPE_TO_SRC_DST:
            continue
        src_t, dst_t = EDGE_TYPE_TO_SRC_DST[et]
        mask = (df["edge_type"] == et).values

        # src id 字典:placement/r1/r2 都是 algorithm;r3 是 data
        if src_t == "algorithm":
            src_dict = algo_ids
        elif src_t == "data":
            src_dict = data_ids
        else:  # compute(本次 EDGE_TYPES 中只有 placement 走 compute src,但保留)
            src_dict = compute_ids

        # dst id 字典:关键差异点 —— r1/r2 的 dst 是 gpu_type_pool,placement 的 dst 是 machine
        if dst_t == "algorithm":
            dst_dict = algo_ids
        elif dst_t == "data":
            dst_dict = data_ids
        else:  # compute
            dst_dict = compute_ids if et == "placement" else gpu_type_pool_ids

        src_int[mask] = df.loc[mask, "src"].map(src_dict).fillna(-1).astype(np.int64).values
        dst_int[mask] = df.loc[mask, "dst"].map(dst_dict).fillna(-1).astype(np.int64).values

    out = df.copy()
    out["src_int"] = src_int
    out["dst_int"] = dst_int

    # 丢掉映射失败的行(原节点被 MIN_GROUP_SIZE 等过滤掉时会出现)
    bad = (out["src_int"] < 0) | (out["dst_int"] < 0)
    if bad.any():
        n_drop = int(bad.sum())
        logger.warning(f"  map_edges_to_global: dropping {n_drop:,} rows with unmapped src/dst")
        out = out[~bad].copy()

    # 全局偏移
    src_t_col = out["edge_type"].map(lambda et: EDGE_TYPE_TO_SRC_DST[et][0])
    dst_t_col = out["edge_type"].map(lambda et: EDGE_TYPE_TO_SRC_DST[et][1])
    out["src_global"] = out["src_int"].values + src_t_col.map(global_offsets).values
    out["dst_global"] = out["dst_int"].values + dst_t_col.map(global_offsets).values
    out["rel_id"] = out["edge_type"].map(RELATION_TO_ID).astype(np.int64).values
    return out


def build_glist_from_df(
    df: pd.DataFrame,
    num_total_nodes: int,
    tau_min: int,
    tau_max: int,
) -> List[Tuple[int, dgl.DGLGraph]]:
    """按 τ 构造 (tau, dgl_graph) 列表。

    - 只保留 tau ∈ [tau_min, tau_max] 内的边
    - 只保留 EDGE_TYPES 中的边(r4_shifts 等会被过滤)
    - 每个 τ 一个同构 DGL 图,含 ndata['id']=arange(N) 与 edata['type']
    - 没有任何边的 τ 跳过(空图对 DGL RGCN update_all 是 no-op,但避免麻烦)
    """
    if len(df) == 0:
        return []
    df = df[df["edge_type"].isin(EDGE_TYPES)].copy()
    df = df[(df["tau"] >= tau_min) & (df["tau"] <= tau_max)].copy()
    if len(df) == 0:
        return []

    glist: List[Tuple[int, dgl.DGLGraph]] = []
    for tau in sorted(df["tau"].unique()):
        sub = df[df["tau"] == tau]
        src = torch.from_numpy(sub["src_global"].to_numpy())
        dst = torch.from_numpy(sub["dst_global"].to_numpy())
        etype = torch.from_numpy(sub["rel_id"].to_numpy())
        g = dgl.graph((src, dst), num_nodes=num_total_nodes)
        g.ndata["id"] = torch.arange(num_total_nodes, dtype=torch.long)
        g.edata["type"] = etype
        glist.append((int(tau), g))
    return glist


def build_node_feature_tensor(
    node_feats_by_type: Dict[str, torch.Tensor],
    node_counts: Dict[str, int],
) -> torch.Tensor:
    """把 per-type 节点特征拼成一个 [N_total, feat_dim] 的同构张量。

    按 algorithm → compute → data 顺序拼接,每个类型都填到 node_counts[t] 行:
      - feat_dim 维度用 0 在右侧 padding 到 max_dim
      - 行数用 0 行 padding 到 node_counts[t](例如 compute 在异构图里 1903 节点
        = 1897 machine + 6 gpu_type_pool,但 node_features.pt 只有 1897 行,
        需要在末尾补 6 个零行才能跟 model.dynamic_emb 的 124,891 行对得上)

    末尾补零的节点通常是 gpu_type_pool 节点,本来就没有显式特征,零向量是合理
    的占位(模型会通过 dynamic_emb 的随机初始化和后续训练学到合适的表示)。
    """
    types_in_order = ["algorithm", "compute", "data"]
    max_dim = max((t.shape[1] for t in node_feats_by_type.values()), default=0)
    rows: List[torch.Tensor] = []
    for t in types_in_order:
        expected_n = int(node_counts.get(t, 0))
        if t in node_feats_by_type:
            feat = node_feats_by_type[t]
            pad = max_dim - feat.shape[1]
            if pad > 0:
                feat = F.pad(feat, (0, pad))
            # 行数补到 expected_n
            if feat.shape[0] < expected_n:
                feat = F.pad(feat, (0, 0, 0, expected_n - feat.shape[0]))
            elif feat.shape[0] > expected_n:
                # 防御:特征比预期多,截断
                feat = feat[:expected_n]
        else:
            feat = torch.zeros((expected_n, max_dim), dtype=torch.float32)
        rows.append(feat)
    full = torch.cat(rows, dim=0)
    return full  # [N_total, max_dim]


# =============================================================================
# 训练/评估辅助
# =============================================================================
def build_dataloaders(
    train_triples: torch.Tensor,
    val_triples: torch.Tensor,
    test_triples: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """把所有 split 三元组搬 device;训练时 GPU 端做负采样,不需要 DataLoader。

    (DataLoader + num_workers/pin_memory 的设置保留在 config 里以备数据集大到
    必须流式读取时启用;目前 125 GB RAM 一次性装下全量边,直接 GPU tensor 最快。)
    """
    return {
        "train": train_triples.to(DEVICE),
        "val":   val_triples.to(DEVICE),
        "test":  test_triples.to(DEVICE),
    }


def build_filter_dict(
    train_t: torch.Tensor,
    val_t: torch.Tensor,
    test_t: torch.Tensor,
    num_rels: int,
) -> Dict[Tuple[int, int], set]:
    """构造 filtered-MRR 用的 filter 集合。

    对每个 (h, r),collect 所有在 train+val+test 里出现过的合法 t(并包含反向
    (t, r+num_rels) → {h} 条目,这样评估反向三元组时也能命中 filter)。

    Returns: dict[(h, r)] -> set of valid t
    """
    filter_dict: Dict[Tuple[int, int], set] = {}
    for triples in (train_t, val_t, test_t):
        if triples.numel() == 0:
            continue
        arr = triples.detach().cpu().numpy()
        for h, r, t in arr:
            h, r, t = int(h), int(r), int(t)
            filter_dict.setdefault((h, r), set()).add(t)
            # 反向三元组 (t, r+num_rels) 也有合法头 h
            filter_dict.setdefault((t, r + num_rels), set()).add(h)
    return filter_dict


def _apply_filter_mask(
    scores: torch.Tensor,
    h_batch: torch.Tensor,
    r_batch: torch.Tensor,
    t_batch: torch.Tensor,
    filter_dict: Dict[Tuple[int, int], set],
) -> torch.Tensor:
    """把 filter_dict 中"非自身"的合法候选位置设为 -inf,返回新 tensor。

    对每个 (h_i, r_i, t_i) 三元组:
      valid = filter_dict.get((h_i, r_i))   # 所有合法 t'
      for j in valid:
          if j != t_i:                       # 关键:绝不 mask 真正的 t
              masked[i, j] = -inf
    """

    masked = scores.clone()
    h_arr = h_batch.detach().cpu().numpy()
    r_arr = r_batch.detach().cpu().numpy()
    t_arr = t_batch.detach().cpu().numpy()
    mask_i: List[int] = []
    mask_j: List[int] = []
    for i in range(scores.shape[0]):
        valid = filter_dict.get((int(h_arr[i]), int(r_arr[i])))
        if not valid:
            continue
        t_i = int(t_arr[i])
        for j in valid:
            if j != t_i:
                mask_i.append(i)
                mask_j.append(j)
    if mask_i:
        mh = torch.tensor(mask_i, device=scores.device, dtype=torch.long)
        mw = torch.tensor(mask_j, device=scores.device, dtype=torch.long)
        masked[mh, mw] = float("-inf")
    return masked


def evaluate(
    model: REGCNModel,
    history_glist: List[Tuple[int, dgl.DGLGraph]],
    eval_glist:    List[Tuple[int, dgl.DGLGraph]],
    eval_df:       pd.DataFrame,
    filter_dict:   Optional[Dict[Tuple[int, int], set]] = None,
    eval_chunk_size: int = EVAL_CHUNK_SIZE,
) -> Dict[str, float]:
    """滑动窗口评估,对齐原版 RE-GCN test 逻辑。
    对每个 eval 时间步 t:
      1. 用当前 history window 做 forward,得到 h_final
      2. 取当前步 tau 的三元组打分,计算 MRR/Hits
      3. 把当前步图加入 history window,pop 最旧的(滑动窗口)
    最终 MRR = 所有三元组的均值。
    """
    zeros = {k: 0.0 for k in (
        "raw_mrr", "raw_h1", "raw_h3", "raw_h10",
        "filt_mrr", "filt_h1", "filt_h3", "filt_h10",
    )}
    if len(eval_glist) == 0:
        return zeros

    model.eval()
    window = list(history_glist[-HISTORY_LEN:])

    sum_raw_mrr = sum_raw_h1 = sum_raw_h3 = sum_raw_h10 = 0.0
    sum_filt_mrr = sum_filt_h1 = sum_filt_h3 = sum_filt_h10 = 0.0
    counted = 0

    with torch.no_grad():
        for tau, g in eval_glist:
            cur_df = eval_df[
                (eval_df["tau"] == tau) &
                (eval_df["edge_type"].isin(EDGE_TYPES))
            ]
            if len(cur_df) == 0:
                window.append((tau, g))
                if len(window) > HISTORY_LEN:
                    window.pop(0)
                continue

            triples = torch.from_numpy(
                cur_df[["src_global", "rel_id", "dst_global"]].to_numpy()
            ).long().to(DEVICE)

            g_list = [g_ for _, g_ in window]
            with autocast(enabled=AMP_ENABLED):
                history_embs, r_emb = model.forward(g_list)
                h_final = history_embs[-1]
                if model.layer_norm:
                    h_final = F.normalize(h_final, p=2, dim=1)
                del history_embs

            total = triples.shape[0]
            n_chunks = (total + eval_chunk_size - 1) // eval_chunk_size
            for ci in range(n_chunks):
                s = ci * eval_chunk_size
                e = min(s + eval_chunk_size, total)
                chunk = triples[s:e]
                targets = chunk[:, 2]

                scores = model.predict(h_final, r_emb, chunk)
                true_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
                larger = (scores > true_scores.unsqueeze(1)).sum(dim=1).float()
                rank = larger + 1

                sum_raw_mrr += (1.0 / rank).sum().item()
                sum_raw_h1  += (rank <= 1).float().sum().item()
                sum_raw_h3  += (rank <= 3).float().sum().item()
                sum_raw_h10 += (rank <= 10).float().sum().item()

                if filter_dict is not None:
                    filt = _apply_filter_mask(
                        scores, chunk[:, 0], chunk[:, 1], chunk[:, 2], filter_dict
                    )
                    filt_true = filt.gather(1, targets.unsqueeze(1)).squeeze(1)
                    filt_larger = (filt > filt_true.unsqueeze(1)).sum(dim=1).float()
                    filt_rank = filt_larger + 1
                    sum_filt_mrr += (1.0 / filt_rank).sum().item()
                    sum_filt_h1  += (filt_rank <= 1).float().sum().item()
                    sum_filt_h3  += (filt_rank <= 3).float().sum().item()
                    sum_filt_h10 += (filt_rank <= 10).float().sum().item()

            counted += total
            window.append((tau, g))
            if len(window) > HISTORY_LEN:
                window.pop(0)

    model.train()
    N = max(1, counted)
    has_filter = filter_dict is not None
    return {
        "raw_mrr":  sum_raw_mrr  / N,
        "raw_h1":   sum_raw_h1   / N,
        "raw_h3":   sum_raw_h3   / N,
        "raw_h10":  sum_raw_h10  / N,
        "filt_mrr": sum_filt_mrr / N if has_filter else sum_raw_mrr  / N,
        "filt_h1":  sum_filt_h1  / N if has_filter else sum_raw_h1   / N,
        "filt_h3":  sum_filt_h3  / N if has_filter else sum_raw_h3   / N,
        "filt_h10": sum_filt_h10 / N if has_filter else sum_raw_h10  / N,
    }


def chunked_cross_entropy(
    scores: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """分块 cross_entropy,避免 [B, num_classes=124,891] 的 log_softmax OOM。

    PyTorch 的 F.cross_entropy 会物化整张 [B, num_classes] 的 log_softmax
    中间张量(FP32 for stability)。B=11,264 × 124,891 × 4B ≈ 5.6 GB,
    跟 RGCN 计算图加在一起直接打爆 24GB 显存。改成按行分块后,每块峰值 ≈
    chunk_size × num_classes × 4B(2 GB for chunk_size=4096,1 GB for 2048)。

    数值上跟 F.cross_entropy(reduction="mean") 完全一致(都对行 sum 后除 N),
    所以梯度流也不变。
    """
    total = scores.new_zeros(())
    n = scores.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        total = total + F.cross_entropy(
            scores[start:end], targets[start:end], reduction="sum"
        )
    return total / n


def cosine_warmup_lr(epoch: int, base_lr: float, final_lr: float,
                     total_epochs: int, warmup_epochs: int) -> float:
    """计算指定 epoch 的学习率(warmup 线性上升 + cosine 衰减到 final_lr)。

    warmup 起点显式锚定 WARMUP_START_LR(默认 5e-5),不再随 base_lr/warmup_epochs
    自动推出。线性插值到 base_lr,之后 cosine 衰减到 final_lr。
    """
    if epoch < warmup_epochs:
        # 线性:epoch 0 → WARMUP_START_LR;epoch warmup_epochs-1 → base_lr
        span = base_lr - WARMUP_START_LR
        return WARMUP_START_LR + span * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, progress))
    return final_lr + 0.5 * (base_lr - final_lr) * (1.0 + np.cos(np.pi * progress))


def save_checkpoint(
    model: REGCNModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_mrr: float,
    is_best: bool,
    saved_paths: List[Path],
) -> None:
    """保存 checkpoint;维持 '最近 N 个 + 1 个 best' 的策略。"""
    fname = CHECKPOINT_DIR / f"regcn_full_epoch_{epoch}.pt"
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_mrr": val_mrr,
    }
    torch.save(state, fname)
    saved_paths.append(fname)
    logger.info(f"  ✓ Checkpoint saved: {fname.name} (val_mrr={val_mrr:.4f})")

    # best 单独命名
    if is_best:
        best_path = CHECKPOINT_DIR / "regcn_full_best.pt"
        torch.save(state, best_path)
        logger.info(f"  ✓ New best checkpoint: {best_path.name}")

    # 清理多余的 last-N(只动 step-based 的 epoch 文件,不动 best)
    if len(saved_paths) > KEEP_LAST_N_CHECKPOINTS:
        for old in saved_paths[:-KEEP_LAST_N_CHECKPOINTS]:
            if old.exists() and "best" not in old.name:
                old.unlink()
                logger.info(f"  ✗ Removed old checkpoint: {old.name}")
        saved_paths[:] = saved_paths[-KEEP_LAST_N_CHECKPOINTS:]


def cleanup_checkpoints() -> None:
    """脚本开始时清掉残留的旧 epoch/best checkpoint(避免磁盘越积越多)。"""
    for p in glob.glob(str(CHECKPOINT_DIR / "regcn_full_epoch_*.pt")):
        Path(p).unlink()
    best = CHECKPOINT_DIR / "regcn_full_best.pt"
    if best.exists():
        best.unlink()


# =============================================================================
# 主训练流程
# =============================================================================
def main() -> Dict:
    tracemalloc.start()
    train_start = time.time()

    # 设备信息
    logger.info("=" * 70)
    logger.info("RE-GCN Stage 3.1 — FULL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info(f"AMP: {AMP_ENABLED}  |  Workers: {DATALOADER_NUM_WORKERS}  |  Pin mem: {DATALOADER_PIN_MEMORY}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    cleanup_checkpoints()

    # -------------------- 数据加载 --------------------
    logger.info("\n[1/5] Loading dataset...")
    dataset = load_xcad_dataset()
    global_offsets, num_total_nodes = build_node_offsets(dataset)
    logger.info(f"  Node offsets: {global_offsets}")
    logger.info(f"  Total nodes: {num_total_nodes:,}")

    # 边数据:从 parquet 加载(已经是 train/val/test 切好的)
    splits_dir = Path(OUTPUT_ROOT) / "dgl" / "splits"
    train_df = pd.read_parquet(splits_dir / "train_edges.parquet")
    val_df = pd.read_parquet(splits_dir / "val_edges.parquet")
    test_df = pd.read_parquet(splits_dir / "test_edges.parquet")
    logger.info(f"  Loaded splits: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    # 节点 ID 映射(从 nodes/*.parquet 复现 build_dgl_graph.py 的字符串→整数映射)
    logger.info("  Building node ID mappings from nodes/*.parquet...")
    mappings = build_node_id_mappings()

    # 边 → 全局 ID(按 edge_type 选 src/dst 字典;r1/r2 dst 走 gpu_type_pool)
    train_df = map_edges_to_global(train_df, mappings, global_offsets)
    val_df = map_edges_to_global(val_df, mappings, global_offsets)
    test_df = map_edges_to_global(test_df, mappings, global_offsets)

    # 按 edge_type 统计
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = df["edge_type"].value_counts().to_dict()
        logger.info(f"  {name} per-type: {counts}")

    # glist:按 τ 分组,每个 τ 一个 DGL 同构图
    # 训练:[1, 50] (r123) + [1, 60] (placement) → 实际用 [1, 60] 整体,map_edges_to_global 时已按各自范围
    # 切分 parquet 时已经按 SPLIT_CONFIG 切过,所以 train_df 内 τ 自动对齐
    train_glist = build_glist_from_df(
        train_df, num_total_nodes,
        tau_min=1, tau_max=MAX_TIME_WINDOWS_PLACEMENT,
    )
    val_glist = build_glist_from_df(
        val_df, num_total_nodes,
        tau_min=1, tau_max=MAX_TIME_WINDOWS_PLACEMENT,
    )
    test_glist = build_glist_from_df(
        test_df, num_total_nodes,
        tau_min=1, tau_max=MAX_TIME_WINDOWS_PLACEMENT,
    )
    logger.info(f"  glist sizes: train={len(train_glist)}, val={len(val_glist)}, test={len(test_glist)}")
    if train_glist:
        logger.info(f"    train τ range: [{train_glist[0][0]}, {train_glist[-1][0]}]")

    # 三元组(只保留 EDGE_TYPES 内的,r4 被过滤)
    def df_to_triples(df: pd.DataFrame) -> torch.Tensor:
        df = df[df["edge_type"].isin(EDGE_TYPES)]
        if len(df) == 0:
            return torch.zeros((0, 3), dtype=torch.long)
        return torch.from_numpy(
            df[["src_global", "rel_id", "dst_global"]].to_numpy()
        ).long()

    train_triples = df_to_triples(train_df)
    val_triples = df_to_triples(val_df)
    test_triples = df_to_triples(test_df)
    logger.info(f"  triples (filtered): train={train_triples.shape[0]:,}, val={val_triples.shape[0]:,}, test={test_triples.shape[0]:,}")

    data = build_dataloaders(train_triples, val_triples, test_triples)

    # -------------------- 节点特征 → 初始化 dynamic_emb --------------------
    logger.info("\n[2/5] Initializing node embeddings from node_features.pt...")
    node_feats_by_type = dataset.node_feats_by_type
    # node_counts 决定每个类型该有多少行(compute 1903 = 1897 machine + 6 gpu_type_pool)
    feat_tensor = build_node_feature_tensor(
        node_feats_by_type, dataset.meta["node_counts"]
    ).to(DEVICE)
    feat_dim = feat_tensor.shape[1]
    logger.info(
        f"  Concat feat shape: {tuple(feat_tensor.shape)} "
        f"(expected N_total={num_total_nodes:,}, types: {list(node_feats_by_type.keys())})"
    )
    assert feat_tensor.shape[0] == num_total_nodes, (
        f"feat rows ({feat_tensor.shape[0]}) != N_total ({num_total_nodes}); "
        f"check build_node_feature_tensor padding"
    )

    # -------------------- 模型 --------------------
    logger.info("\n[3/5] Building model...")
    model = REGCNModel(
        num_nodes=num_total_nodes,
        num_rels=NUM_RELATIONS,
        h_dim=HIDDEN_DIM,
        n_layers=NUM_LAYERS,
        num_bases=-1,
        dropout=DROPOUT,
    ).to(DEVICE)

    # 用节点特征(经单层线性投影)初始化 dynamic_emb
    if feat_dim != HIDDEN_DIM:
        feat_proj = nn.Linear(feat_dim, HIDDEN_DIM).to(DEVICE)
        nn.init.xavier_uniform_(feat_proj.weight)
        nn.init.zeros_(feat_proj.bias)
        with torch.no_grad():
            init_h = feat_proj(feat_tensor)  # [N_total, HIDDEN_DIM]
        logger.info(f"  Projected features: {feat_dim} → {HIDDEN_DIM}")
    else:
        init_h = feat_tensor
    with torch.no_grad():
        model.dynamic_emb.data.copy_(init_h)
    logger.info(f"  dynamic_emb initialized from features: {tuple(model.dynamic_emb.shape)} on {model.dynamic_emb.device}")

    # glist 搬 device(评估/前向时直接用,不在每 epoch 重复 to)
    train_glist = [(tau, g.to(DEVICE)) for tau, g in train_glist]
    val_glist   = [(tau, g.to(DEVICE)) for tau, g in val_glist]
    test_glist  = [(tau, g.to(DEVICE)) for tau, g in test_glist]

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = GradScaler()

    # -------------------- 训练循环 --------------------
    logger.info("\n[4/5] Training...")
    logger.info(f"  Optimizer: Adam (wd={WEIGHT_DECAY})  |  Scheduler: {LR_SCHEDULER} (warmup={WARMUP_EPOCHS})")
    logger.info(f"  Epochs: {MAX_EPOCHS}  |  Patience: {EARLY_STOP_PATIENCE}  |  Batch: {BATCH_SIZE}  |  Neg: {NEG_SAMPLE_RATIO}")

    best_val_mrr = -1.0
    best_epoch = -1
    patience_counter = 0
    saved_paths: List[Path] = []

    history = {
        "epoch_loss": [],
        "epoch_time": [],
        "epoch_lr": [],
        "epoch_gpu_peak_mb": [],
        "val_raw_mrr": [], "val_raw_h1": [], "val_raw_h3": [], "val_raw_h10": [],
        "val_filt_mrr": [], "val_filt_h1": [], "val_filt_h3": [], "val_filt_h10": [],
        "test_raw_mrr_at_best": None, "test_raw_h1_at_best": None,
        "test_raw_h3_at_best": None, "test_raw_h10_at_best": None,
        "test_filt_mrr_at_best": None, "test_filt_h1_at_best": None,
        "test_filt_h3_at_best": None, "test_filt_h10_at_best": None,
    }

    # -------------------- 构造 filter dict(全量边)--------------------
    logger.info("\n[Filter] Building filter dict for filtered MRR...")
    filter_dict = build_filter_dict(
        data["train"].cpu(), data["val"].cpu(), data["test"].cpu(), NUM_RELATIONS
    )
    logger.info(f"  filter_dict: {len(filter_dict):,} (h, r) keys")

    train_t = data["train"]
    num_train = train_t.shape[0]
    num_nodes = num_total_nodes

    for epoch in range(MAX_EPOCHS):
        # ---- LR schedule(每个 epoch 前更新 lr)----
        cur_lr = cosine_warmup_lr(epoch, LEARNING_RATE, FINAL_LR, MAX_EPOCHS, WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
        history["epoch_lr"].append(cur_lr)

        # ---- Epoch 训练(原版 RE-GCN 风格:每个时间步一次完整 forward + backward)----
        epoch_start = time.time()
        model.train()
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)

        # 随机打乱时间步顺序(原版 src/main.py: random.shuffle(idx))
        # 从 HISTORY_LEN 开始:每步都保证有完整 HISTORY_LEN 张图作为 history,
        # 避免早期时间步 history 不足导致 loss 爆炸(之前 tau=2 loss=796 那种情况)
        idx = list(range(HISTORY_LEN, len(train_glist)))
        random.shuffle(idx)

        epoch_losses: List[float] = []
        n_steps = len(idx)
        for step, t in enumerate(idx, start=1):
            # history:取前 HISTORY_LEN 步的图(原版 train_list[max(0, t-history_len):t])
            hist_start = max(0, t - HISTORY_LEN)
            history_graphs = [g for _, g in train_glist[hist_start:t]]
            if len(history_graphs) == 0:
                continue

            # 当前时间步的三元组(只取正向 EDGE_TYPES,不含逆向)
            tau_t = train_glist[t][0]
            cur_df = train_df[train_df["tau"] == tau_t]
            cur_df = cur_df[cur_df["edge_type"].isin(EDGE_TYPES)]
            if len(cur_df) == 0:
                continue
            cur_triples = torch.from_numpy(
                cur_df[["src_global", "rel_id", "dst_global"]].to_numpy()
            ).long().to(DEVICE)

            # tail-only 负采样
            B = cur_triples.shape[0]
            neg_tail = cur_triples.clone().repeat(NEG_SAMPLE_RATIO, 1)
            rand_t_idx = torch.randint(0, num_nodes, (B * NEG_SAMPLE_RATIO,), device=DEVICE)
            neg_tail[:, 2] = rand_t_idx
            all_samples = torch.cat([cur_triples, neg_tail], dim=0)

            # 完整 forward + loss(不 detach,RGCN 和 decoder 同步更新)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=AMP_ENABLED):
                history_embs, r_emb = model.forward(history_graphs)
                h_final = history_embs[-1]
                if model.layer_norm:
                    h_final = F.normalize(h_final, p=2, dim=1)
                del history_embs  # 立即释放中间层,只保留 h_final 和 r_emb

                # 限制每步正例数量,避免 all_samples 过大 OOM
                if B > MAX_TRIPLES_PER_STEP:
                    perm = torch.randperm(B, device=DEVICE)[:MAX_TRIPLES_PER_STEP]
                    cur_triples = cur_triples[perm]
                    B = MAX_TRIPLES_PER_STEP
                    neg_tail = cur_triples.clone().repeat(NEG_SAMPLE_RATIO, 1)
                    rand_t_idx = torch.randint(0, num_nodes, (B * NEG_SAMPLE_RATIO,), device=DEVICE)
                    neg_tail[:, 2] = rand_t_idx
                    all_samples = torch.cat([cur_triples, neg_tail], dim=0)

                scores = model.decoder(h_final, r_emb, all_samples)
                loss = chunked_cross_entropy(scores, all_samples[:, 2], CE_CHUNK_SIZE)

            # NaN/Inf 保护:跳过这个时间步,不污染参数
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(
                    f"[Epoch {epoch+1}] step {step}/{n_steps} (tau={tau_t}): "
                    f"NaN/Inf loss, skip."
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

            # 每步都打 loss 日志(总共 59 步/epoch,不会刷屏)
            logger.info(
                f"  [Epoch {epoch+1}] tau={tau_t} step={step}/{n_steps}  "
                f"loss={loss.item():.4f}  B={B}  lr={cur_lr:.2e}"
            )

        # === Epoch 末尾:评估、checkpoint、early stopping(原版 RE-GCN 风格)===
        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        epoch_time = time.time() - epoch_start
        gpu_peak_mb = (torch.cuda.max_memory_allocated(DEVICE) / 1024**2) if DEVICE.type == "cuda" else 0.0
        history["epoch_loss"].append(avg_loss)
        history["epoch_time"].append(epoch_time)
        history["epoch_gpu_peak_mb"].append(gpu_peak_mb)

        # ---- 验证(raw + filtered)----
        val_metrics = evaluate(
            model,
            history_glist=train_glist,
            eval_glist=val_glist,
            eval_df=val_df,
            filter_dict=filter_dict,
        )
        # === evaluate 内部已 del 了 history_embs/r_emb;此处再 empty_cache 保险一次,
        #     避免 val 评估占用的临时张量在下一 epoch 起始 forward 时还占着显存 ===
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)
        v_raw  = val_metrics["raw_mrr"]
        v_filt = val_metrics["filt_mrr"]

        logger.info(
            f"Epoch {epoch+1}/{MAX_EPOCHS}  loss={avg_loss:.4f}  "
            f"time={epoch_time:.2f}s  lr={cur_lr:.2e}  "
            f"GPU_peak={gpu_peak_mb:.0f}MB  "
            f"val_raw_MRR={v_raw:.4f}  val_filt_MRR={v_filt:.4f}  "
            f"raw_H@1/3/10={val_metrics['raw_h1']:.3f}/{val_metrics['raw_h3']:.3f}/{val_metrics['raw_h10']:.3f}  "
            f"filt_H@1/3/10={val_metrics['filt_h1']:.3f}/{val_metrics['filt_h3']:.3f}/{val_metrics['filt_h10']:.3f}"
        )
        # === 用户 spec 格式的 DONE 行:强调 GPU/RAM 峰值,便于一眼看显存走势 ===
        logger.info(
            f"[Epoch {epoch+1}] DONE  "
            f"avg_loss={avg_loss:.4f}  time={epoch_time:.1f}s  "
            f"val_raw_MRR={v_raw:.4f}  val_filt_MRR={v_filt:.4f}  "
            f"GPU_peak={gpu_peak_mb / 1024:.2f}GB"
        )

        # ---- Checkpoint & Early Stopping(以 raw val MRR 为指标,跟原脚本一致)----
        is_best = v_raw > best_val_mrr
        if is_best:
            best_val_mrr = v_raw
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % SAVE_CHECKPOINT_EVERY_N_EPOCHS == 0 or is_best:
            save_checkpoint(model, optimizer, epoch + 1, v_raw, is_best, saved_paths)

        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.info(
                f"  ⏹ Early stopping at epoch {epoch+1} "
                f"(no val raw-MRR improvement for {EARLY_STOP_PATIENCE} epochs)"
            )
            break

    # -------------------- 加载 best checkpoint 跑 test --------------------
    logger.info("\n[5/5] Loading best checkpoint for test evaluation...")
    best_ckpt = CHECKPOINT_DIR / "regcn_full_best.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(state["model_state"])
        logger.info(f"  Loaded: {best_ckpt.name}  (epoch={state['epoch']}, val_mrr={state['val_mrr']:.4f})")
    else:
        logger.warning("  No best checkpoint found, using current model state")

    test_metrics = evaluate(
        model,
        history_glist=train_glist + val_glist,
        eval_glist=test_glist,
        eval_df=test_df,
        filter_dict=filter_dict,
    )
    # === test evaluate 完显式 empty_cache,清掉临时张量 ===
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    for k, v in test_metrics.items():
        history[f"test_{k}_at_best"] = v

    total_time = time.time() - train_start
    cur_ram, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_ram / 1024**2
    peak_gpu_mb = (torch.cuda.max_memory_allocated(DEVICE) / 1024**2) if DEVICE.type == "cuda" else 0.0

    # -------------------- 报告 --------------------
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE — Summary")
    logger.info("=" * 70)
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best val raw MRR: {best_val_mrr:.4f}")
    logger.info(f"Test raw   MRR: {test_metrics['raw_mrr']:.4f}, "
                f"Hits@1/3/10: {test_metrics['raw_h1']:.4f}/{test_metrics['raw_h3']:.4f}/{test_metrics['raw_h10']:.4f}")
    logger.info(f"Test filt   MRR: {test_metrics['filt_mrr']:.4f}, "
                f"Hits@1/3/10: {test_metrics['filt_h1']:.4f}/{test_metrics['filt_h3']:.4f}/{test_metrics['filt_h10']:.4f}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Peak GPU memory: {peak_gpu_mb:.0f} MB")
    logger.info(f"Peak system RAM (tracemalloc): {peak_ram_mb:.0f} MB")
    logger.info("=" * 70)

    _write_report(
        history=history,
        best_epoch=best_epoch,
        best_val_mrr=best_val_mrr,
        test_metrics=test_metrics,
        total_time=total_time,
        peak_gpu_mb=peak_gpu_mb,
        peak_ram_mb=peak_ram_mb,
        num_total_nodes=num_total_nodes,
        num_train=train_t.shape[0],
        num_val=data["val"].shape[0],
        num_test=data["test"].shape[0],
    )

    return {
        "best_epoch": best_epoch,
        "best_val_mrr": best_val_mrr,
        "test_raw_mrr": test_metrics["raw_mrr"],
        "test_filt_mrr": test_metrics["filt_mrr"],
        "test_raw_hits10": test_metrics["raw_h10"],
        "test_filt_hits10": test_metrics["filt_h10"],
        "total_time": total_time,
        "peak_gpu_mb": peak_gpu_mb,
        "peak_ram_mb": peak_ram_mb,
    }


def _write_report(**kw) -> None:
    """把训练结果写到 ${OUTPUT_ROOT}/reports/stage3.1_full.md。"""
    h = kw["history"]
    tm = kw["test_metrics"]
    report = []
    P = report.append
    P("# RE-GCN Stage 3.1 — Full Training Report\n")
    P("> Generated by `src.training.train_regcn_full`\n")
    P("## Configuration\n")
    P(f"- Edge types: `{EDGE_TYPES}`")
    P(f"- r1/r2/r3 max τ: {MAX_TIME_WINDOWS_R123}")
    P(f"- placement max τ: {MAX_TIME_WINDOWS_PLACEMENT}")
    P(f"- Hidden dim: {HIDDEN_DIM}  |  Layers: {NUM_LAYERS}  |  Dropout: {DROPOUT}")
    P(f"- Optimizer: Adam (wd={WEIGHT_DECAY}, lr={LEARNING_RATE} → {FINAL_LR}, {LR_SCHEDULER} + {WARMUP_EPOCHS}-ep warmup)")
    P(f"- Batch size: {BATCH_SIZE}  |  Neg ratio: {NEG_SAMPLE_RATIO}")
    P(f"- AMP: {AMP_ENABLED}  |  Grad clip: {GRAD_CLIP}  |  Patience: {EARLY_STOP_PATIENCE}")
    P(f"- Max epochs: {MAX_EPOCHS}  |  Random seed: {RANDOM_SEED}")
    P(f"- Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")
    P(f"- RGCN history: epoch-level shared (RE-GCN 原版做法)")
    P(f"- Filtered MRR: enabled (filter dict over train+val+test)")
    P(f"- Total nodes: {kw['num_total_nodes']:,}")
    P(f"- Train triples: {kw['num_train']:,}  |  Val: {kw['num_val']:,}  |  Test: {kw['num_test']:,}")
    P("")
    P("## Final Results\n")
    P(f"- **Best epoch**: {kw['best_epoch']}")
    P(f"- **Best val raw MRR** (early-stop 监控): {kw['best_val_mrr']:.4f}")
    P("")
    P("### Test Set (loaded from best checkpoint)\n")
    P("| Metric | Raw | Filtered |")
    P("|---|---|---|")
    P(f"| MRR        | {tm['raw_mrr']:.4f} | {tm['filt_mrr']:.4f} |")
    P(f"| Hits@1     | {tm['raw_h1']:.4f} | {tm['filt_h1']:.4f} |")
    P(f"| Hits@3     | {tm['raw_h3']:.4f} | {tm['filt_h3']:.4f} |")
    P(f"| Hits@10    | {tm['raw_h10']:.4f} | {tm['filt_h10']:.4f} |")
    P("")
    P(f"- Total time: {kw['total_time']:.2f}s")
    P(f"- Peak GPU memory: {kw['peak_gpu_mb']:.0f} MB")
    P(f"- Peak system RAM (tracemalloc): {kw['peak_ram_mb']:.0f} MB")
    P("")
    P("## Training Loss & Validation Curve\n")
    P("| Epoch | Loss | Time (s) | LR | GPU (MB) | Val raw MRR | Val filt MRR | raw H@1 | raw H@3 | raw H@10 | filt H@1 | filt H@3 | filt H@10 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    n = len(h["epoch_loss"])
    for i in range(n):
        P(
            f"| {i+1} | {h['epoch_loss'][i]:.4f} | {h['epoch_time'][i]:.2f} | "
            f"{h['epoch_lr'][i]:.2e} | {h['epoch_gpu_peak_mb'][i]:.0f} | "
            f"{h['val_raw_mrr'][i]:.4f} | {h['val_filt_mrr'][i]:.4f} | "
            f"{h['val_raw_h1'][i]:.4f} | {h['val_raw_h3'][i]:.4f} | {h['val_raw_h10'][i]:.4f} | "
            f"{h['val_filt_h1'][i]:.4f} | {h['val_filt_h3'][i]:.4f} | {h['val_filt_h10'][i]:.4f} |"
        )
    P("")
    P("## Status\n")
    P(f"- Training completed: {'YES' if kw['best_epoch'] > 0 else 'NO'}")
    P(f"- Output non-NaN metrics: {'YES' if not np.isnan(tm['raw_mrr']) and tm['raw_mrr'] > 0 else 'NO'}")
    P("")

    report_path = REPORT_DIR / "stage3.1_full.md"
    report_path.write_text("\n".join(report))
    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
    sys.exit(0)
