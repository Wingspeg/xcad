#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_maas_base_emb.py -- base 共享表征这一臂的机制检查（0 GPU，1 分钟级）。

臂的读数只能回答"过没过线"。这一脚本回答"为什么"，三个问题各对应一个下一步选项：

Q1 base_emb 里到底学没学到"父节点 -> 卡型"？
   对 918 个 base 的 embedding 做 5 折逻辑回归预测它的历史众数卡型，
   同时跑两个对照：【随机 normal_ 同形状矩阵】（无信号应有的水平）和【众数类基线】。
   => 学到了但总分没上去 = 通道通的、被别的项拖住；没学到 = 通道根本没接上。
Q2 base_emb 真的被训练了吗？
   pad 行（索引 918，mask 恒 0，只吃 weight decay）当"零梯度参照"，
   比较它和 918 个真实行的标准差/范数变化。
Q3 涨的那层和跌的那层是不是同一批边？
   用 base.pt 与 baseemb.pt 的 filt_rank 逐边配对（同一批 121,371 行、同一把尺子），
   按 cohort 算 H@1 差值，再算"只有 baseemb 答对 / 只有 base 答对"的边数。
   => 若全新 head 层涨、复发层跌且两边互补量大，说明缺的是"见过/没见过"的门，不是容量。
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

D = os.path.expanduser("~/xcad/xcad_code/xcad_model")
ARMS = os.path.join(D, "src/maas_arms")
DATA = os.path.join(D, "data/maas")
sys.path.insert(0, DATA)

n_ents = int(open(os.path.join(DATA, "stat.txt")).read().split()[0])
base_of = np.load(os.path.join(DATA, "base_of.npy"))
pad = int(base_of.max())
n_bases = pad

# ── 历史（train+valid）边：head -> tail 局部编号（减去 gpu_base）
names = {}
with open(os.path.join(DATA, "entity2id.txt")) as f:
    for ln in f:
        p = ln.rstrip("\n").split("\t")
        if len(p) == 2 and p[0] not in ("head", "relation"):
            names[int(p[1])] = p[0]
gpu_base = int(min(t for fn in ("train.txt", "valid.txt", "test.txt")
                   for t in [int(ln.split()[2]) for ln in open(os.path.join(DATA, fn))
                             if len(ln.split()) >= 4 and int(ln.split()[1]) == 0]))
print(f"[id-space] n_ents={n_ents} gpu_base={gpu_base} n_bases={n_bases} pad={pad}")

hist_edges, test_rows = [], []
for fn in ("train.txt", "valid.txt"):
    for ln in open(os.path.join(DATA, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            hist_edges.append((int(p[0]), int(p[2]) - gpu_base))
for ln in open(os.path.join(DATA, "test.txt")):
    p = ln.split()
    if len(p) >= 4 and int(p[1]) == 0:
        test_rows.append((int(p[0]), int(p[2]) - gpu_base, int(p[3])))
T = pd.DataFrame(test_rows, columns=["hd", "tail", "tau"])
hist = pd.DataFrame(hist_edges, columns=["hd", "tail"])
hist_pairs = set(map(tuple, hist.values))
hist_heads = set(hist.hd)

# base -> 历史众数卡型（只用 train+valid，不碰 test）
h = hist.assign(b=base_of[hist.hd.values])
cnt = h.groupby(["b", "tail"]).size().reset_index(name="n")
modal = cnt.sort_values(["b", "n", "tail"], ascending=[True, False, True]).drop_duplicates("b")
lab = dict(zip(modal["b"], modal["tail"]))
used_all = sorted(set(int(x) for x in base_of[:gpu_base]))
print(f"[labels] 有历史边的 base={len(lab)}/{n_bases}  众数卡型类分布 top5="
      f"{pd.Series(list(lab.values())).value_counts().head(5).to_dict()}")

# ── 载入 checkpoint
ck = glob.glob(os.path.join(D, "models/*baseemb-seed42"))
assert len(ck) == 1, ck
sd = torch.load(ck[0], map_location="cpu")["state_dict"]
base_emb = sd["base_emb"].numpy().astype(np.float64)
dyn = sd["dynamic_emb"].numpy().astype(np.float64)
print(f"[ckpt] {os.path.basename(ck[0])}  epoch={torch.load(ck[0], map_location='cpu')['epoch']}")

# Q2 零梯度参照
print(f"[Q2] base_emb 真实行 std={base_emb[:n_bases].std():.4f} 范数中位={np.linalg.norm(base_emb[:n_bases], axis=1).mean():.4f}"
      f"  |  pad 行 std={base_emb[pad].std():.4f} 范数={np.linalg.norm(base_emb[pad]):.4f}"
      f"  |  dynamic_emb std={dyn.std():.4f}")

# Q1 探针
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 只有【历史里真有边】的 base 才有标签（93/918 个 base 只带全新 head，无从监督）
used = [b for b in used_all if b in lab]
Xb = base_emb[used]
yb = np.array([lab[b] for b in used])
best = max(set(yb.tolist()), key=list(yb).count)
rng = np.random.default_rng(0)
Xr = rng.normal(size=Xb.shape)


def cv(X, y):
    try:
        s = cross_val_score(LogisticRegression(max_iter=3000, C=1.0), X, y, cv=5,
                            scoring="accuracy")
        return float(s.mean()), float(s.std())
    except Exception as e:                                   # noqa: BLE001
        return float("nan"), float("nan")


a_sig, a_sig_sd = cv(Xb, yb)
a_rnd, a_rnd_sd = cv(Xr, yb)
print(f"[Q1] base 众数类基线={best:.4f}({list(yb).count(best)/len(yb):.4f})  "
      f"trained base_emb 5折 acc={a_sig:.4f}±{a_sig_sd:.4f}  "
      f"随机同形状对照 acc={a_rnd:.4f}±{a_rnd_sd:.4f}")

# ── Q3 逐边配对
rows = {}
for tag in ("base", "baseemb"):
    d = torch.load(os.path.join(ARMS, f"{tag}.pt"), map_location="cpu")
    assert len(d["heads"]) == len(T), f"{tag} dump {len(d['heads'])} != {len(T)}"
    assert np.array_equal(d["heads"].numpy(), T.hd.values), f"{tag} 行序与 test.txt 不一致"
    rows[tag] = d["filt_rank"].numpy().astype(np.float64)

cohorts = {
    "全部 test 边": np.ones(len(T), bool),
    "全新 head": ~T.hd.isin(hist_heads).values,
    "边在 history(复发层)": np.array([(s, t) in hist_pairs for s, t in
                                  zip(T.hd.values, T["tail"].values)]),
}
print("\n[Q3] 同一把尺子、同一批行：")
for c, m in cohorts.items():
    h1b, h1e = (rows["base"][m] <= 1).mean(), (rows["baseemb"][m] <= 1).mean()
    only_e = int(((rows["baseemb"][m] <= 1) & (rows["base"][m] > 1)).sum())
    only_b = int(((rows["base"][m] <= 1) & (rows["baseemb"][m] > 1)).sum())
    med = np.median(rows["baseemb"][m]), np.median(rows["base"][m])
    print(f"  {c:22s} n={int(m.sum()):7d}  base H@1={h1b:.4f}  baseemb H@1={h1e:.4f}  "
          f"Δ={h1e-h1b:+.4f}  只有baseemb对={only_e:5d}  只有base对={only_b:5d}  "
          f"中位排名 {med[1]:.1f}->{med[0]:.1f}")

# 头表征里 base 项占多少（全新 vs 复发）
for c, m in (("全新 head", cohorts["全新 head"]), ("复发 head", cohorts["边在 history(复发层)"])):
    ids = T.hd.values[m]
    bn = np.linalg.norm(base_emb[base_of[ids]], axis=1)
    dn = np.linalg.norm(dyn[ids], axis=1)
    print(f"[占比] {c}: |base 项| 中位={np.median(bn):.3f}  |id 项| 中位={np.median(dn):.3f}"
          f"  base 占比={np.median(bn/(bn+dn)):.3f}")
