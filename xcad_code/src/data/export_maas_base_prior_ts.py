#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_maas_base_prior_ts.py -- (H) 时序口径的输出层先验表。

为什么要换口径（§25.6 的探针读数，`probe_maas_prior_train_leak.py`）：
    §23–§25 用的两张表是 **train+valid 全历史**的一张静态表，于是训练/验证时每一行的
    head 通道都非空（它自己那条边就在表里 —— 16.4% 的训练行在该 head 的任何更早快照
    上根本没有边，而这些行 79.8% 的 argmax 正是它自己这行的 gold）。模型在这上面学出
    alpha_m≈1.8~2.7 的"这条通道常开且可信"，而测试期该通道在 84.75% 的行上是 0
    （可用率 train 1.0000 vs test 0.1525）=> 信任度校准在错误的分布上，val 也失效。

本脚本产出的表把口径改成【严格更早】：
    base_prior_ts[t, b, k] = log1p( #{train+valid 里 rel0 且 ts < t 且 base(head)==b
                                      且卡型==k 的边} )
    head_prior_ts[t, v, k] = log1p( #{train+valid 里 rel0 且 ts < t 且 head==v
                                      且卡型==k 的边} )
下标 t 直接用数据文件第 4 列的 ts（maas: train 1..47 / valid 48..53 / test 54..65），
所以模型侧不需要知道"第几个快照"，只需要把当前 batch 的 ts 递进来。

两条必须先钉住的事实（脚本里都是断言，不是说明）：
  * **test 侧一个字都没变**：所有 t>=54 的表与 §23 的静态表逐位相同（test 行的条件不变，
    所以本轮任何 test 读数差异只能来自"模型学到了什么"，不可能来自"评测看到了什么"）。
  * **自泄漏按构造为 0**：某行 head 通道非空 <=> 该 head 在严格更早的快照里出现过。

test 的边永远不进任何表（只用 train.txt + valid.txt），与 §23/§24/§25 同一套泄漏纪律。

Usage (server):
    ~/xcad/xcad_code/.venv/bin/python xcad_code/src/data/export_maas_base_prior_ts.py
"""
import json
import os

import numpy as np

D = os.path.expanduser("~/xcad/xcad_code/xcad_model/data/maas")
OUT_BASE = os.path.join(D, "base_prior_ts.npy")
OUT_HEAD = os.path.join(D, "head_prior_ts.npy")
OUT_JSON = os.path.join(D, "prior_ts_report.json")
STATIC_BASE = os.path.join(D, "base_prior.npy")      # §23 的静态表，用来验 test 侧不变
STATIC_HEAD = os.path.join(D, "head_prior.npy")

n_ents = int(open(os.path.join(D, "stat.txt")).read().split()[0])
base_of = np.load(os.path.join(D, "base_of.npy"))
assert base_of.shape == (n_ents,)
pad = int(base_of.max())
n_bases = pad

# ── rel0 的 tail 段 = 末尾连续段（与 §23 同一判定） ──
tails = set()
for fn in ("train.txt", "valid.txt", "test.txt"):
    for ln in open(os.path.join(D, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            tails.add(int(p[2]))
tail_lo = min(tails)
assert max(tails) == n_ents - 1 and len(tails) == n_ents - tail_lo, "tail 段不是末尾连续段"
n_types = len(tails)

# ── 读边（带 ts）；hist = 进表的边（train+valid），test 只用于算可用率 ──
def read(fn):
    out = []
    for ln in open(os.path.join(D, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            out.append((int(p[0]), int(p[2]) - tail_lo, int(p[3])))
    return np.asarray(out, dtype=np.int64)

E = {s: read(f"{s}.txt") for s in ("train", "valid", "test")}
all_ts = np.concatenate([E[s][:, 2] for s in E])
T = int(all_ts.max()) + 1                       # 下标 = ts 原值（0 行恒为全 0）
for s in E:
    assert E[s][:, 0].min() >= 0 and E[s][:, 0].max() < tail_lo, "rel0 的 head 应在 tail 段之前"
    assert ((E[s][:, 1] >= 0) & (E[s][:, 1] <= n_types - 1)).all()
    assert E[s][:, 2].min() >= 1, "maas 的 ts 从 1 开始；出现 0 说明数据变了，口径要重推"
print(f"[id-space] n_ents={n_ents} tail_lo={tail_lo} n_types={n_types} n_bases={n_bases} "
      f"T={T} (ts 取值 1..{T - 1})")
print("[splits] " + "  ".join(
    f"{s}: n={len(E[s])} ts=[{E[s][:, 2].min()},{E[s][:, 2].max()}] 唯一={len(np.unique(E[s][:, 2]))}"
    for s in E))
assert E["train"][:, 2].max() < E["valid"][:, 2].min() < E["valid"][:, 2].max() < E["test"][:, 2].min(), \
    "三段 ts 不连续/有交叠 => 时序口径无法按 ts 直接建表"

HIST = np.concatenate([E["train"], E["valid"]])

# ── 建表：按天计数 -> 沿 ts 前缀和 -> 右移一位(= 严格更早) -> log1p ──
def build(idx_axis_size, head_to_row):
    day = np.zeros((T, idx_axis_size, n_types), dtype=np.float32)
    np.add.at(day, (HIST[:, 2], head_to_row(HIST[:, 0]), HIST[:, 1]), 1.0)
    np.cumsum(day, axis=0, out=day)                       # day[t] = ts<=t
    shifted = np.zeros_like(day)
    shifted[1:] = day[:-1]                                # shifted[t] = ts<t
    assert not shifted[0].any() and not day[0].any(), "ts=0 应该没有边"
    return shifted

head_cnt = build(n_ents, lambda h: h)
base_cnt = build(n_bases + 1, lambda h: base_of[h])
assert np.allclose(base_cnt[:, pad], 0.0), "pad 行不该有计数"
assert not head_cnt[:, tail_lo:].any(), "卡型实体不该有历史出边计数"

# 单调性（计数层面）
for nm, c in (("head", head_cnt), ("base", base_cnt)):
    assert (np.diff(c, axis=0) >= 0).all(), f"{nm}_prior_ts 沿 ts 出现倒退"

# log1p 必须走 float64 再落 float32 —— 与 §23 静态表的算法逐位一致（静态表就是对 float64
# 计数做 log1p 后写入 float32 数组的）。直接对 float32 取 log1p 会有 ~2e-7 的最后一位差。
def log1p_f32(cnt):
    out = np.empty(cnt.shape, dtype=np.float32)
    for t in range(cnt.shape[0]):
        out[t] = np.log1p(cnt[t].astype(np.float64))
    return out

head_prior = log1p_f32(head_cnt)
base_prior = log1p_f32(base_cnt)
del head_cnt, base_cnt

# ── C3: test 侧与 §23 静态表逐位相同（静态表没有 time 轴，所以要逐 t 比同一张 2-D 表） ──
t_test_lo = int(E["test"][:, 2].min())
sb = np.load(STATIC_BASE); sh = np.load(STATIC_HEAD)
assert sh.shape == (n_ents, n_types) and sb.shape[1] == n_types
same_head = all(np.array_equal(head_prior[t], sh) for t in range(t_test_lo, T))
same_base = all(np.array_equal(base_prior[t, :sb.shape[0]], sb) for t in range(t_test_lo, T))
if not (same_head and same_base):                  # 先把差量打出来再决定是不是真不一致
    _t = t_test_lo
    print(f"[C3-diff] t={_t} head maxdiff={np.abs(head_prior[_t] - sh).max():.3e} "
          f"不同元素={int((head_prior[_t] != sh).sum())}/{sh.size}  "
          f"base maxdiff={np.abs(base_prior[_t, :sb.shape[0]] - sb).max():.3e} "
          f"不同元素={int((base_prior[_t, :sb.shape[0]] != sb).sum())}/{sb.size}")
    _w = np.argwhere(head_prior[_t] != sh)[:3]
    for _i, _k in _w:
        print(f"           head[{_i},{_k}] 时序={head_prior[_t][_i, _k]!r} 静态={sh[_i, _k]!r}")
assert same_head and same_base, (
    f"test 段(t>={t_test_lo})与静态表不一致 head={same_head} base={same_base} "
    "=> 本轮改的应当只是训练/验证期的表，test 侧必须逐位不变")
assert np.array_equal(head_prior[t_test_lo], sh), "首个 test 快照的表必须等于静态表"
# 反向自检：训练/验证段的表【必须】与静态表不同，否则这次改动等于没改
t_first_valid = int(E["valid"][:, 2].min())
assert not np.array_equal(head_prior[t_first_valid], sh), \
    "valid 首快照的表与静态表相同 => 时序口径没有生效"
assert not np.array_equal(head_prior[int(E["train"][:, 2].max())], sh), \
    "train 末快照的表与静态表相同 => 时序口径没有生效"

# ── C4/C5: 分流的可用率 + 自泄漏（按构造必须为 0） ──
first_ts = np.full(n_ents, 10 ** 9, dtype=np.int64)
np.minimum.at(first_ts, (HIST[:, 0],), HIST[:, 2])
rep_split = {}
print("\n[availability] 分流可用率（时序口径）")
print("   split   n        head 可用率   base 可用率   非空表但 head 无更早快照(应=0)")
for s in ("train", "valid", "test"):
    hd, tc, ts = E[s][:, 0], E[s][:, 1], E[s][:, 2]
    hv = head_prior[ts, hd].max(axis=1) > 0
    bv = base_prior[ts, base_of[hd]].max(axis=1) > 0
    bad = int((hv & (ts <= first_ts[hd])).sum())           # C5 的硬检验
    hh = float((head_prior[ts, hd][hv].argmax(axis=1) == tc[hv]).mean()) if hv.any() else float("nan")
    print(f"   {s:6s} {len(hd):8d}   {hv.mean():.4f}        {bv.mean():.4f}        {bad}")
    rep_split[s] = {"n": int(len(hd)), "head_avail": round(float(hv.mean()), 4),
                    "base_avail": round(float(bv.mean()), 4),
                    "head_avail_hit": round(hh, 4),
                    "self_leak_rows": bad}
    assert bad == 0, f"{s} 行里有 {bad} 行在无更早快照时表非空 => 自泄漏没修掉"
gap = rep_split["train"]["head_avail"] - rep_split["test"]["head_avail"]
print(f"   >>> head 可用率 train - test = {gap:+.4f}   （静态口径下这个数是 +0.8475）")

# 全表统计
rep = {
    "convention": "bias[t, *, k] = log1p(count of train+valid rel0 edges with ts < t)",
    "index_axis": "dim0 == ts value from data file column 4 (maas: train 1..47, valid 48..53, test 54..65)",
    "T": T, "n_ents": n_ents, "n_bases": n_bases, "n_types": n_types,
    "head_table_shape": list(head_prior.shape), "base_table_shape": list(base_prior.shape),
    "head_table_mb": round(head_prior.nbytes / 1e6, 1),
    "test_side_identical_to_static": {"head": same_head, "base": same_base,
                                      "first_test_ts": t_test_lo},
    "static_convention_head_avail_gap_train_minus_test": 0.8475,
    "temporal_head_avail_gap_train_minus_test": round(gap, 4),
    "splits": rep_split,
    "log1p_max_head": round(float(head_prior.max()), 4),
    "log1p_max_base": round(float(base_prior.max()), 4),
    "n_heads_with_history_at_last_ts": int((sh[:tail_lo].sum(axis=1) > 0).sum()),
    "self_leak_rows_all_splits": 0,
    "history_edges_in_table": int(len(HIST)),
}
np.save(OUT_BASE, base_prior)
np.save(OUT_HEAD, head_prior)
with open(OUT_JSON, "w") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
print(f"\n[write] {OUT_HEAD} shape={head_prior.shape} ({rep['head_table_mb']} MB)  "
      f"log1p<= {rep['log1p_max_head']}")
print(f"[write] {OUT_BASE} shape={base_prior.shape}  log1p<= {rep['log1p_max_base']}")
print(json.dumps(rep, ensure_ascii=False, indent=2))
