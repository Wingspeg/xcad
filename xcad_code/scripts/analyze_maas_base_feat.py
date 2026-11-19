#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_maas_base_feat.py -- 注入臂（--base-feat）的 0-GPU 机制检查。

要回答三件事，都是"下一步该做什么"的输入，不是自损读数：

Q1 注入的通道被读到了吗？
    把闭式 base_hist（同一张历史表、同一个生产排名函数）逐边和模型放一起看：
    在【base_hist 已经答对】的那些边上，模型答对多少？两臂对照（base / basefeat）。
Q2 差距集中在哪一类 base？
    按"该 head 的 base 在历史里有多少条边"分三分位，逐段看模型 vs base_hist。
    => 若高支持段已经贴近闭式、低支持段崩，问题在解码器怎么用这张表，不在表本身。
Q3 深层指标（H@3/H@10）为什么掉？
    排名分布（中位/p90/p99）+ 多答案格子占比分层：一个 (head, 快照) 在历史里有几种卡型。
    任务是先验多标签的（54.3% 的 (model,day) 格子多种卡型），尖锐化会牺牲深命中。
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

D = os.path.expanduser("~/xcad/xcad_code/xcad_model")
ARMS = os.path.join(D, "src/maas_arms")
DATA = os.path.join(D, "data/maas")
sys.path.insert(0, os.path.join(D, "src"))
import generic_strata as prod                                   # noqa: E402

n_ents = int(open(os.path.join(DATA, "stat.txt")).read().split()[0])
base_of = np.load(os.path.join(DATA, "base_of.npy"))
pad = int(base_of.max())

tails = set()
for fn in ("train.txt", "valid.txt", "test.txt"):
    for ln in open(os.path.join(DATA, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            tails.add(int(p[2]))
gpu_base = min(tails)
n_types = len(tails)

hrows, trows = [], []
for fn in ("train.txt", "valid.txt"):
    for ln in open(os.path.join(DATA, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            hrows.append((int(p[0]), int(p[2]) - gpu_base))
rows = []
for ln in open(os.path.join(DATA, "test.txt")):
    p = ln.split()
    if len(p) >= 4 and int(p[1]) == 0:
        rows.append((int(p[0]), int(p[2]) - gpu_base, int(p[3])))
T = pd.DataFrame(rows, columns=["hd", "tc", "tau"])
H = pd.DataFrame(hrows, columns=["hd", "tc"])
hist_pairs = set(map(tuple, H.values))
hist_heads = set(H.hd)

# 该 base 的历史边数（支持量）与卡型直方图
Hb = H.assign(b=base_of[H.hd.values])
sup = Hb.groupby("b").size()
cnt = Hb.groupby(["b", "tc"]).size().rename("n").reset_index()
tab = {int(b): {int(c): int(v) for c, v in sub.groupby("tc")["n"].sum().items()}
       for b, sub in cnt.groupby("b")}

# 与 dump 同序的行（test.txt rel0 行序 == main.py 收集顺序）
ranks = {}
for tag in ("base", "baseemb", "basefeat"):
    d = torch.load(os.path.join(ARMS, f"{tag}.pt"), map_location="cpu")
    assert np.array_equal(d["heads"].numpy(), T.hd.values), f"{tag} 行序不一致"
    ranks[tag] = d["filt_rank"].numpy().astype(np.float64)
n = len(T)

# 过滤集：同一 (head, 快照) 内其他答案
per = {}
for h, tau, t in zip(T.hd, T.tau, T.tc):
    per.setdefault((h, tau), set()).add(int(t))
gold = T.tc.to_numpy()
filt = [{x for x in per[(h, tau)] if x != int(g)} for h, tau, g in
        zip(T.hd, T.tau, gold)]

row_tab = [tab.get(int(b), {}) for b in base_of[T.hd.values]]
uniq, index_of, rk_id = [], {}, np.empty(n, dtype=np.int64)
for i, t in enumerate(row_tab):
    k = frozenset(t.items())
    j = index_of.get(k)
    if j is None:
        j = index_of[k] = len(uniq)
        uniq.append(t)
    rk_id[i] = j
bh_rank = prod._ranks_from_cols(rk_id, uniq, np.asarray(gold), filt, n_types,
                                np.random.default_rng(7), (0, n_types - 1)).astype(np.float64)

new_head = ~T.hd.isin(hist_heads).values
recur = np.array([(s, t) in hist_pairs for s, t in zip(T.hd.values, T.tc.values)])
bsup = sup.reindex(base_of[T.hd.values]).fillna(0).to_numpy()
q = np.quantile(bsup[new_head], [1/3, 2/3])
band = np.digitize(bsup, q)


def m(r, mask):
    x = r[mask]
    return (f"H@1={np.mean(x <= 1):.4f} H@3={np.mean(x <= 3):.4f} "
            f"H@10={np.mean(x <= 10):.4f} 中位={np.median(x):.1f} "
            f"p90={np.quantile(x, .9):.1f} p99={np.quantile(x, .99):.1f} n={int(mask.sum())}")


print(f"[setup] n={n} n_types={n_types} n_bases={pad} 全新 head={new_head.mean():.4f} "
      f"复发边={recur.mean():.4f}")
print("\n=== Q0 闭式 base_hist 自检（应与 §20 同尺对上）===")
print("  全新 head 层      ", m(bh_rank, new_head))
print("  复发边层          ", m(bh_rank, recur))

print("\n=== Q1 注入通道被读到了吗：base_hist 答对的边上，模型答对多少 ===")
for name, mask in (("全新 head", new_head), ("复发边", recur)):
    hit = mask & (bh_rank <= 1)
    for tag in ("base", "baseemb", "basefeat"):
        print(f"  {name:8s} base_hist对({hit.sum():6d}) -> 模型{tag:8s} 也答对 "
              f"{np.mean(ranks[tag][hit] <= 1):.4f}   | base_hist错 -> 模型答对 "
              f"{np.mean(ranks[tag][mask & (bh_rank > 1)] <= 1):.4f}")

print("\n=== Q2 差距集中在哪类 base（按 base 历史边数三分位，只看全新 head）===")
print(f"  三分位边界 = {list(np.round(q, 1))}")
for b, lab in ((0, "低支持"), (1, "中支持"), (2, "高支持")):
    sel = new_head & (band == b)
    print(f"  {lab} n={int(sel.sum()):6d}  base_hist {m(bh_rank, sel)[:9]}  "
          f"base {m(ranks['base'], sel)[:9]}  basefeat {m(ranks['basefeat'], sel)[:9]}"
          f"  |  base_hist 的 base 支持中位={np.median(bsup[sel]):.0f}")

print("\n=== Q3 深命中：排名分布与多答案格子 ===")
n_ans = np.array([len(per[(h, t)]) for h, t in zip(T.hd, T.tau)])
multi = n_ans >= 2
for name, mask in (("全部", np.ones(n, bool)), ("单答案格子", ~multi), ("多答案格子", multi)):
    print(f"  {name:12s} n={int(mask.sum()):6d}")
    for tag in ("base", "basefeat"):
        print(f"      {tag:8s} {m(ranks[tag], mask)}")

print("\n=== 附：三臂同表（便于抄进文档）===")
for name, mask in (("全部 test 边", np.ones(n, bool)), ("全新 head", new_head),
                   ("复发边(硬线层)", recur)):
    print(f"  {name:16s} n={int(mask.sum()):7d}")
    for tag in ("base", "baseemb", "basefeat"):
        print(f"      {tag:8s} {m(ranks[tag], mask)}")
    print(f"      base_hist{m(bh_rank, mask)}")
