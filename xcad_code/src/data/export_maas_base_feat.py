#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_maas_base_feat.py -- base 的历史卡型直方图 -> 可直接灌进模型的 31 维特征矩阵。

为什么是"注入"而不是"学习"（§21.3 的 Q1）：
  base 共享 embedding 那一臂里，918 行是可学习空白表，训练完拿它预测"该 base 历史最常上
  哪种卡"只有 5 折 acc 0.1224±0.0241，与随机同形状矩阵 0.1152±0.0184 在噪声内；而同一批
  边上【直接查历史直方图】的闭式通道 H@1 = 0.2980。差别不在信息，在于信息的来源：闭式那条
  就是统计表本身，空白表没有任何力量要求它长成统计表。本脚本把统计表当成输入特征，模型只需
  要学一个投影。

列定义（全部只用 train+valid 的边，与 §20 闭式通道同一张表，不碰 test、不碰 gpu 之外的答案列）：
  [0..29] 该 base 历史上每种卡型的【占比】p_k = count_k / total  —— 保证不同规模的 base 可比
  [30]    log1p(total) 支持量 —— 让模型能区分"看过 5 次的父"和"看过 5 万次的父"
然后对 31 列各自 z-score（均值/标准差只从【有历史边的 base】算），pad 行（第 918 行）取全 0，
即"落在均值上的无信息父"；pad 行是否被使用由模型侧的 base_valid mask 决定，不会被灌进 head。

Usage (server):
    ~/xcad/xcad_code/.venv/bin/python xcad_code/src/data/export_maas_base_feat.py
"""
import json
import os

import numpy as np
import pandas as pd

D = os.path.expanduser("~/xcad/xcad_code/xcad_model/data/maas")
OUT_NPY = os.path.join(D, "base_feat.npy")
OUT_JSON = os.path.join(D, "base_feat_report.json")

n_ents = int(open(os.path.join(D, "stat.txt")).read().split()[0])
base_of = np.load(os.path.join(D, "base_of.npy"))
assert base_of.shape == (n_ents,)
pad = int(base_of.max())
n_bases = pad

# tail 段 = 末尾连续段（与 export_maas_base_map.py 同法，独立再判一次）
tails = set()
for fn in ("train.txt", "valid.txt", "test.txt"):
    for ln in open(os.path.join(D, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            tails.add(int(p[2]))
head_hi = min(tails)
assert max(tails) == n_ents - 1 and len(tails) == n_ents - head_hi
n_types = len(tails)
print(f"[id-space] n_ents={n_ents} head_hi={head_hi} n_types={n_types} n_bases={n_bases}")

# ── 只用 train+valid 构表
rows = []
for fn in ("train.txt", "valid.txt"):
    for ln in open(os.path.join(D, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            rows.append((int(p[0]), int(p[2]) - head_hi))
H = pd.DataFrame(rows, columns=["hd", "tc"])
assert H.tc.between(0, n_types - 1).all()
H["b"] = base_of[H.hd.values]
cnt = H.groupby(["b", "tc"]).size().unstack(fill_value=0).reindex(
    range(n_bases), fill_value=0).to_numpy(dtype=np.float64)
total = cnt.sum(axis=1)
print(f"[hist] 有历史边的 base={int((total > 0).sum())}/{n_bases}  总计数={int(total.sum())}"
      f"  单 base 支持量 中位={np.median(total[total > 0]):.1f} 最大={total.max():.0f}")
assert cnt.shape == (n_bases, n_types)

p = np.divide(cnt, total[:, None], out=np.zeros_like(cnt), where=total[:, None] > 0)
sup = np.log1p(total)
raw = np.concatenate([p, sup[:, None]], axis=1)                  # [n_bases, 31]

# z-score 统计量只在有边的 base 上算
used = total > 0
mu = raw[used].mean(axis=0)
sd = raw[used].std(axis=0)
sd[sd <= 0] = 1.0
feat = np.zeros((n_bases + 1, raw.shape[1]), dtype=np.float32)
feat[:n_bases] = (raw - mu) / sd
assert not np.isfinite(feat[pad]).any() or np.allclose(feat[pad], 0.0)

np.save(OUT_NPY, feat)
rep = {"n_bases": n_bases, "pad_index": pad, "n_types": n_types, "feat_dim": raw.shape[1],
       "columns": "p_k = count_k/total for k in 0..n_types-1, then log1p(total)",
       "normalisation": "per-column z-score, stats over bases with >=1 history edge",
       "mu": [round(float(x), 6) for x in mu], "sd": [round(float(x), 6) for x in sd],
       "n_bases_with_history": int(used.sum()),
       "history_only": "train.txt + valid.txt rel0 edges; test never touched",
       "feat_std_after_norm": round(float(feat[:n_bases][used].std()), 4)}
with open(OUT_JSON, "w") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
print(f"[write] {OUT_NPY} shape={feat.shape}  z-score 后 std={rep['feat_std_after_norm']}")
print(json.dumps({k: v for k, v in rep.items() if k not in ("mu", "sd")},
                 ensure_ascii=False, indent=2))
