#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_maas_prior_train_leak.py -- 注入的先验通道在【训练时】看到什么 vs 在【测试时】看到什么。

起因（§25 的 (F) 臂判线失败后的定位）：按行门控把 base 通道的信任度从 §23 的 1.5557 压到
约 1.15，并把 head 通道抬到约 2.2，结果复发层涨(0.7357->0.7450)而全新 head 层再跌
(0.2571->0.2365)。门控"方向对、水平错"，指向一个实现层的口径问题：两张偏置表都是
**train+valid 全历史**的静态表，于是训练时每一行的 head 通道都非空(它自己的那条边就在表里)，
而测试时只有 15.25% 的行有 head 历史 ⇒ 模型在训练里学到的"这条通道常开"在测试上不成立。

这个脚本只做四件事，全部 0 GPU：
  P1 两张表在 train/valid/test 三个流上的"行可用率"(该行偏置非零的比例)
  P2 同一批行上"表 argmax == gold 卡型"的比例(= 通道在那一批行上的可用正确率)
  P3 差值 = 训练期可见但测试期不可用的比例(= 泄漏量级)
  P4 训练期"只靠它自己那条边"的行数(该 head 在此之前没有任何快照)：这些行的 argmax
     必然等于 gold ⇒ 通道把答案直接递给 loss。
判读口径：测试行不含任何自泄漏(表里根本没有 test 边)，所以 P2/P3 的 test 列是真值；
train/valid 列是"模型训练时看到的假象"。
"""
import os

import numpy as np

D = os.path.expanduser("~/xcad/xcad_code/xcad_model/data/maas")
n_ents = int(open(os.path.join(D, "stat.txt")).read().split()[0])
base_of = np.load(os.path.join(D, "base_of.npy"))
head_prior = np.load(os.path.join(D, "head_prior.npy"))          # [n_ents, 30]
base_prior = np.load(os.path.join(D, "base_prior.npy"))          # [n_bases+1, 30]

tails = set()
rows = {"train": [], "valid": [], "test": []}
for fn in ("train.txt", "valid.txt", "test.txt"):
    split = fn.split(".")[0]
    for ln in open(os.path.join(D, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            h, t, tau = int(p[0]), int(p[2]), int(p[3])
            tails.add(t)
            rows[split].append((h, t, tau))
tail_lo = min(tails)
n_types = len(tails)
assert max(tails) == n_ents - 1 and n_types == n_ents - tail_lo
print(f"[setup] n_ents={n_ents} tail_lo={tail_lo} n_types={n_types} "
      f"n_train={len(rows['train'])} n_valid={len(rows['valid'])} n_test={len(rows['test'])}")

# head 在【更早快照】是否出现过（全时序口径：同一张表里所有 split 的首次快照）
hv = {}
for split in ("train", "valid", "test"):
    for h, t, tau in rows[split]:
        hv.setdefault(h, []).append(tau)
earlier = {h: min(taus) for h, taus in hv.items()}            # 首次出现的快照


def stats(split):
    hd = np.array([r[0] for r in rows[split]])
    tc = np.array([r[1] for r in rows[split]]) - tail_lo
    tau = np.array([r[2] for r in rows[split]])
    hb = head_prior[hd]                                       # [n, 30]
    bb = base_prior[base_of[hd]]
    h_avail = hb.max(axis=1) > 0
    b_avail = bb.max(axis=1) > 0
    h_argmax = hb.argmax(axis=1) == tc
    b_argmax = bb.argmax(axis=1) == tc
    # 该行在此快照之前有没有任何历史（严格早于 τ）
    has_past = tau > np.fromiter((earlier.get(h, 10 ** 9) for h in hd),
                                dtype=np.int64, count=len(hd))
    only_self = h_avail & ~has_past
    return dict(n=len(hd),
                h_avail=h_avail.mean(), b_avail=b_avail.mean(),
                h_hit=h_argmax[h_avail].mean() if h_avail.any() else float("nan"),
                b_hit=b_argmax[b_avail].mean() if b_avail.any() else float("nan"),
                only_self=only_self.mean(),
                only_self_hit=(hb[only_self].argmax(axis=1) == tc[only_self]).mean()
                if only_self.any() else float("nan"))


print("\n        流     n      head 可用率  head 可用时argmax命中   base 可用率  base 命中   "
      "\"无过去却有表\"比例  其中argmax==gold")
for split in ("train", "valid", "test"):
    s = stats(split)
    print(f"  {split:6s} {s['n']:8d}   {s['h_avail']:.4f}      {s['h_hit']:.4f}            "
          f"{s['b_avail']:.4f}      {s['b_hit']:.4f}      {s['only_self']:.4f}          {s['only_self_hit']:.4f}")

print("\n[P3] head 通道可用性 训练期 - 测试期 =",
      f"{stats('train')['h_avail'] - stats('test')['h_avail']:+.4f}")
print("[P3] head 通道命中   训练期 - 测试期 =",
      f"{stats('train')['h_hit'] - stats('test')['h_hit']:+.4f}")
print("[P3] base 通道可用性 训练期 - 测试期 =",
      f"{stats('train')['b_avail'] - stats('test')['b_avail']:+.4f}")
print("[P3] base 通道命中   训练期 - 测试期 =",
      f"{stats('train')['b_hit'] - stats('test')['b_hit']:+.4f}")
print("\n读法：train 行里 head 表非空 = 100% 是构造使然(表由 train+valid 全历史计数, 含该边自身)。"
      "\n     '无过去却有表'那列给出的是【只能靠自身那条边】的行占比 —— 这些行上通道等于把 gold 递过去。")
