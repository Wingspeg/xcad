#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_maas_base_prior.py -- 历史卡型计数 -> 可直接加在【输出层 logits】上的偏置表。

两张表（E1 = 双通道，§23.6 预先写下的做法）：
    base_prior[b, k] = log1p(该 base 历史上落在卡型 k 的边数)   行数 = n_bases + 1
    head_prior[v, k] = log1p(实体 v 自己历史上落在卡型 k 的边数) 行数 = n_ents
模型侧 score += alpha_b * base_prior[base(head), k] + alpha_m * head_prior[head, k]
两个标量各自可学习 => "父节点的构成"与"这个 model 自己的历史落点"是两条独立通道，
信任度由数据决定，不需要我们手工合并（§23 Q3 的 oracle 档就是手工合并的上界）。

为什么要换注入位置（§22 的读数给的指向）：
  --base-feat 把同一张统计表当【输入特征】灌进 head 表征，复发层第一次越过常数基线
  (0.2015 > tail_prior 0.1895)，但闭式通道本身在该层是 0.4448、在全新 head 层是 0.2981，
  模型只吃到 0.1551/0.2015。Q1 说得很直白：base_hist 答对的那些边，basefeat 也只答对
  29.3%；base_hist 答错的边上模型没变(0.0964 vs base 0.0966) —— 信息进来了，但经过
  "head 向量偏移 -> 与 30 个卡型向量做双线性"这条路，表里的【候选之间谁比谁强】几乎没被
  保留下来。统计表的自然形状是"每个候选一个分数"，就该加在每个候选的分数上。

列定义（与 §20 闭式通道、base_feat 同一张表，只用 train+valid 的 rel0 边，不碰 test）：
    bias[b, k] = log1p(count_{b,k})
  取 log 而不是占比：偏置要进 softmax 的 logits，log 计数在 logit 空间是"计数相乘"的正确
  尺度；同时 log1p 自动把"支持量小的父"压平(差异小 => 影响小)，不需要再引入置信度列。
  pad 行（第 n_bases 行）取全 0：无父实体/卡型实体加零偏置，等价于"不注入"。
  head 表同理：无历史边的 head 整行 0；30 个卡型实体自己当 head 时也不可能有历史，
  整行强制 0（它们不在 rel0 的 head 段里，置 0 只是让"零行 = 不注入"这条不变量全局成立）。

Usage (server):
    ~/xcad/xcad_code/.venv/bin/python xcad_code/src/data/export_maas_base_prior.py
"""
import json
import os

import numpy as np
import pandas as pd

D = os.path.expanduser("~/xcad/xcad_code/xcad_model/data/maas")
OUT_NPY = os.path.join(D, "base_prior.npy")
OUT_JSON = os.path.join(D, "base_prior_report.json")
HEAD_NPY = os.path.join(D, "head_prior.npy")
HEAD_JSON = os.path.join(D, "head_prior_report.json")

n_ents = int(open(os.path.join(D, "stat.txt")).read().split()[0])
base_of = np.load(os.path.join(D, "base_of.npy"))
assert base_of.shape == (n_ents,)
pad = int(base_of.max())
n_bases = pad

# tail 段 = 末尾连续段（与 export_maas_base_map.py / export_maas_base_feat.py 同法，再判一次）
tails = set()
for fn in ("train.txt", "valid.txt", "test.txt"):
    for ln in open(os.path.join(D, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            tails.add(int(p[2]))
tail_lo = min(tails)
assert max(tails) == n_ents - 1 and len(tails) == n_ents - tail_lo, "tail 段不是末尾连续段"
n_types = len(tails)
print(f"[id-space] n_ents={n_ents} tail_lo={tail_lo} n_types={n_types} n_bases={n_bases}")

rows = []
for fn in ("train.txt", "valid.txt"):
    for ln in open(os.path.join(D, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            rows.append((int(p[0]), int(p[2]) - tail_lo))
H = pd.DataFrame(rows, columns=["hd", "tc"])
assert H.tc.between(0, n_types - 1).all()
assert H.hd.between(0, tail_lo - 1).all(), "rel0 的 head 应全部落在 tail 段之前"
H["b"] = base_of[H.hd.values]
cnt = H.groupby(["b", "tc"]).size().unstack(fill_value=0).reindex(
    range(n_bases), fill_value=0).to_numpy(dtype=np.float64)
assert cnt.shape == (n_bases, n_types)
total = cnt.sum(axis=1)

# 第二张表：head 自己的历史卡型计数（= §17.3 entity_memory 通道的计数形式）
hcnt = (H.groupby(["hd", "tc"]).size().unstack(fill_value=0)
          .reindex(index=range(n_ents), columns=range(n_types), fill_value=0)
          .to_numpy(dtype=np.float64))
assert hcnt.shape == (n_ents, n_types)
assert not hcnt[tail_lo:].any(), "tail 段实体不该有历史出边计数"
htotal = hcnt.sum(axis=1)

prior = np.zeros((n_bases + 1, n_types), dtype=np.float32)
prior[:n_bases] = np.log1p(cnt)
assert np.allclose(prior[pad], 0.0), "pad 行应为全 0(无信息父 = 不注入)"

head_prior = np.zeros((n_ents, n_types), dtype=np.float32)
head_prior[:tail_lo] = np.log1p(hcnt[:tail_lo])
assert np.allclose(head_prior[tail_lo:], 0.0), "卡型实体行应为全 0(不注入)"

np.save(OUT_NPY, prior)
np.save(HEAD_NPY, head_prior)
rep = {
    "n_bases": n_bases, "pad_index": pad, "n_types": n_types,
    "tail_lo_entity_id": int(tail_lo),
    "columns": "bias[b, k] = log1p(count of history edges whose head's base == b and card type == k)",
    "history_only": "train.txt + valid.txt rel0 edges; test never touched",
    "n_bases_with_history": int((total > 0).sum()),
    "total_history_edges": int(total.sum()),
    "log1p_count_min": round(float(prior[:n_bases][total > 0].min()), 4),
    "log1p_count_max": round(float(prior[:n_bases].max()), 4),
    "pad_row_max_abs": round(float(np.abs(prior[pad]).max()), 8),
    "argmax_equals_modal_card_type": bool(
        (np.argmax(cnt, axis=1) == np.argmax(prior[:n_bases], axis=1)).all()),
    "head_table": {
        "shape": [int(n_ents), int(n_types)],
        "columns": "bias[v, k] = log1p(count of history edges whose head == v and card type == k)",
        "n_heads_with_history": int((htotal > 0).sum()),
        "frac_entities_with_history": round(float((htotal > 0).mean()), 4),
        "total_history_edges": int(htotal.sum()),
        "log1p_count_min": round(float(np.log1p(htotal[htotal > 0].min())), 4),
        "log1p_count_max": round(float(head_prior.max()), 4),
        "zero_rows": int(n_ents - int((htotal > 0).sum())),
        "tail_rows_all_zero": True,
        "argmax_equals_modal_card_type": bool(
            (np.argmax(hcnt[:tail_lo], axis=1) == np.argmax(head_prior[:tail_lo], axis=1)).all()),
    },
}
with open(OUT_JSON, "w") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
hrep = {k: v for k, v in rep.items() if k != "head_table"}
hrep.update(rep["head_table"])
with open(HEAD_JSON, "w") as f:
    json.dump(hrep, f, ensure_ascii=False, indent=2)
print(f"[write] {OUT_NPY} shape={prior.shape}  "
      f"log1p 计数范围=[{rep['log1p_count_min']}, {rep['log1p_count_max']}]")
print(f"[write] {HEAD_NPY} shape={head_prior.shape}  "
      f"有历史 head={rep['head_table']['n_heads_with_history']}/{n_ents} "
      f"({rep['head_table']['frac_entities_with_history']:.1%})  "
      f"log1p 计数范围=[{rep['head_table']['log1p_count_min']}, "
      f"{rep['head_table']['log1p_count_max']}]")
print(json.dumps(rep, ensure_ascii=False, indent=2))
