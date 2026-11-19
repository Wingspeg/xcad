#!/usr/bin/env python3
# @File    : probe_r21_r22_feasibility.py
"""
0-GPU 预探针：在把任何臂排进夜间队列之前，先量两件事的"可测性"。

R2-1（r1_suits 是否只是历史调度器偏置）
    需要知道 cooccur_count / success_rate 这两个支持度字段在【测试边】上的分布：
    若支持度高的那一段样本量足够（每档 ≥ 数百条），"按观测充分性分层重报"就能做，
    而且只需要一次 test-only + --dump-preds（约 30 秒 GPU），不用训练。

R2-2（画像是否含未来信息）
    画像是对一个 job 实体的所有实例做聚合（build_nodes.py:139 groupby job_name）。
    若绝大多数 job 的全部实例本来就落在【同一个快照】里，那"聚合到了未来"这件事在
    结构上就几乎不存在，逐快照重算画像的臂不会改变任何一行数字 —— 这条必须在跑臂
    之前先量，否则会浪费 45 分钟 GPU 去做一个恒等变换。

只读，不改任何文件。输出到 stdout，供 docs/reviews/ 记录。
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/xcad/xcad_code")
EDGES = os.path.join(ROOT, "outputs", "edges")
CLEAN = os.path.expanduser("~/xcad/datasets/clean/ali_v2020")
DATA = os.path.join(ROOT, "xcad_model", "data", "xcad_v2")
GPU_BASE = 124885


def sec(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


# ────────────────────────────── R2-1: 支持度分布 ──────────────────────────────
def r21():
    sec("R2-1  r1_suits 支持度（cooccur_count / success_rate）分布")
    r1 = pd.read_parquet(os.path.join(EDGES, "r1_suits_edges.parquet"))
    print(f"r1_suits 边总数 = {len(r1)}  列 = {list(r1.columns)}")
    print("\nsuccess_rate 分布（注意 0.0 档：这些边进了图但权重为 0）:")
    print(r1.success_rate.round(3).value_counts().head(8).to_string())
    print("\ncooccur_count 分位: " +
          "  ".join(f"{q}={r1.cooccur_count.quantile(q):.0f}" for q in (.5, .75, .9, .99)) +
          f"  max={r1.cooccur_count.max()}")
    for lo, hi in [(1, 1), (2, 4), (5, 9), (10, 49), (50, 10 ** 9)]:
        m = (r1.cooccur_count >= lo) & (r1.cooccur_count <= hi)
        print(f"  支持度 [{lo},{hi}] 边数 = {int(m.sum()):7d} ({m.mean():.4f})"
              f"   其中 success_rate==1.0 占 {r1.loc[m, 'success_rate'].eq(1.0).mean():.4f}")

    # 测试边侧：把 test.txt 里的 r1 行（rel_id==1）映回 (job_name, gpu_type)
    split = os.path.join(DATA, "test.txt")
    if not os.path.exists(split):
        print(f"\n[skip] 找不到 {split}，无法把支持度分层落到测试边上")
        return
    heads, gpus = [], []
    with open(split) as f:
        for line in f:
            p = line.split()
            if len(p) < 3 or p[1] != "1":
                continue
            heads.append(int(p[0]))
            gpus.append(int(p[2]) - GPU_BASE)
    heads = np.asarray(heads); gpus = np.asarray(gpus)
    print(f"\ntest.txt 里 rel_id==1 的边数 = {len(heads)}（应为 11,836）")
    print(f"tail-GPU 类分布 = {np.bincount(gpus, minlength=6).tolist()}（应为 5032/5246/552/811/195 顺序）")
    print("[下一步] 需要 algo_ids 字典（job_name -> id，首次出现顺序）才能把 head id 换成 job_name；"
          "本次探针先只报全图侧的支持度分布。")


# ────────────────────────────── R2-2: 跨快照聚合的量级 ──────────────────────────────
def r22():
    sec("R2-2  画像聚合是否真的跨越时间（一个 job 的实例分布在几个快照）")
    path = os.path.join(CLEAN, "placements.parquet")
    if not os.path.exists(path):
        print(f"[skip] 找不到 {path}")
        return
    pl = pd.read_parquet(path, columns=["job_name", "snap_start"])
    n0 = len(pl)
    pl = pl.dropna(subset=["snap_start"]).copy()
    pl["snap"] = pl.snap_start.astype("int64")
    g = pl.groupby("job_name")["snap"]
    nun = g.nunique()
    span = g.agg(lambda s: int(s.max() - s.min()))
    first = pl.groupby("job_name")["snap"].transform("min")
    later = (pl["snap"] > first).to_numpy()
    print(f"placements 行数 = {n0}（snap_start 缺失 {n0 - len(pl)}）  job 数 = {len(nun)}")
    print(f"每个 job 的【不同快照数】: 均值 {nun.mean():.3f} | >1 的 job 占 {float((nun > 1).mean()):.4f} | 最大 {int(nun.max())}")
    print(f"首末快照跨度: 均值 {span.mean():.3f} | p90 {span.quantile(.9):.0f} | p99 {span.quantile(.99):.0f} | max {int(span.max())}")
    print(f"落在该 job 首快照【之后】的 placement 行占比 = {later.mean():.4f}")
    print(f"有'之后行'的 job 占比 = {float(pd.Series(later).groupby(pl.job_name.to_numpy()).max().mean()):.4f}")
    print("\n判读：若上面两个比例都很小，逐快照重算画像 ≈ 恒等变换，不该排 GPU 臂；"
          "若之后行占比可观，才值得做 causal 画像臂。")


def main():
    print(f"python={sys.version.split()[0]}  pandas={pd.__version__}")
    r21()
    r22()


if __name__ == "__main__":
    main()
