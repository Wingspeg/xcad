#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_xcad_prior_bias_scale.py -- 0-GPU：先验项在两个数据集上到底"有多大"。

(D) 在 v2020 判负（复发层 0.7077，判线①要 0.90、停止线⑧ 0.8045），要区分两种解释：
  (a) 机制没用 —— 注入历史先验本身就翻不动模型的决定；
  (b) 量级没用 —— 同一套 `score += α·log1p(表[head])`，v2020 的表每行只有 1~2 条计数，
      乘上学到的 α≈1.55 后加成 ~1.0，而模型段内的分数差比它大得多；maas 的表每行动辄
      几十条，加成 ~α·4，所以那边能推 top-1。
本脚本只量表本身（不算模型分数，那需要 GPU 重跑），对两个数据集用同一把尺子：
在各自的 rel 全集 test 边上，取时序口径表在该快照·该 head 那一行（表存的就是 log1p(count)，
`rrgcn.apply_base_prior` 只做 `score[:, 卡型段] += α · 表[head]`），报
  行内是否非空、最大格反解出的真实计数 count1=expm1(表值)、次大格、
  先验项能拉开的 logit 差 = α·(表值1 − 表值2)，以及"该行全 0"的占比。
分层 = 复发 head（该行非空）/ 全新 head（该行全 0）。

用法(server): ~/xcad/xcad_code/.venv/bin/python xcad_code/scripts/probe_xcad_prior_bias_scale.py
产物: outputs/reports/probe_xcad_prior_bias_scale.{log,json}
"""
import json
import os

import numpy as np

D = os.path.expanduser("~/xcad/xcad_code/xcad_model/data")
REP = os.path.expanduser("~/xcad/xcad_code/outputs/reports")
ALPHA = {"xcad_v2": 1.5510, "maas": 1.5557}          # 各自学到的 head_alpha / prior_alpha
REL = {"xcad_v2": 1, "maas": 0}                       # v2020 的 r1_suits / maas 的唯一关系

CASES = {
    "xcad_v2": {"cube": "head_prior_ts.npy", "n_types": 5},
    "maas": {"cube": "head_prior_ts.npy", "n_types": 30},
}


def q(a, ps=(0, 1, 5, 25, 50, 75, 99, 100)):
    return {f"p{p}": (None if len(a) == 0 else float(np.percentile(a, p))) for p in ps}


out = {}
for ds, cfg in CASES.items():
    n_ents = int(open(os.path.join(D, ds, "stat.txt")).read().split()[0])
    rd = {s: np.loadtxt(os.path.join(D, ds, f"{s}.txt"), dtype=np.int64)
          for s in ("train", "valid", "test")}
    for s, a in rd.items():
        assert a.shape[1] == 4, f"{ds}/{s}.txt 不是 4 列 (head rel tail τ)"
    trv = np.concatenate([rd["train"], rd["valid"]])
    r = rd["test"][rd["test"][:, 1] == REL[ds]]                 # 本臂评测用的那批 test 边
    rr = trv[trv[:, 1] == REL[ds]]                              # 可查历史 = train+valid
    hd, tc, ts_raw = r[:, 0], r[:, 2], r[:, 3]
    # cube 的层轴 = 快照序号（原始 τ 排序后的下标 +1，第 0 层全 0），不是原始 τ
    snaps = np.unique(np.concatenate([rr[:, 3], r[:, 3]]))
    layer = np.searchsorted(snaps, ts_raw) + 1
    fl = np.searchsorted(snaps, rr[:, 3]) + 1
    first_layer = np.full(n_ents, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_layer, rr[:, 0], fl)
    honest = layer > first_layer[hd]
    cube = np.load(os.path.join(D, ds, cfg["cube"]), mmap_mode="r")
    assert cube.ndim == 3 and cube.shape[1] == n_ents and cube.shape[2] == cfg["n_types"], \
        f"{ds}: cube 形状 {cube.shape} 与预期不符"
    assert cube.shape[0] == len(snaps) + 1, \
        f"{ds}: cube 层数 {cube.shape[0]} != 快照数+1={len(snaps)+1} ⇒ 层轴口径不是'快照序号+1'"
    n_types = cube.shape[2]
    # 逐边取表行（按快照分组，一次只搬一层，避免 80k×30 全量复制）
    rows = np.zeros((len(hd), n_types), dtype=np.float32)
    for t in np.unique(layer):
        m = layer == t
        rows[m] = np.asarray(cube[t][hd[m]], dtype=np.float32)
    tot = rows.sum(axis=1)
    is_rec = tot > 0
    assert np.array_equal(is_rec, honest), \
        (f"{ds}: 表非空行 与 '有严格更早快照' 不一致 "
         f"only_table={int((is_rec & ~honest).sum())} only_honest={int((honest & ~is_rec).sum())}"
         " ⇒ 层轴映射错了，本脚本读数不可用")
    srt = -np.sort(-rows, axis=1)
    lp1, lp2 = srt[:, 0], srt[:, 1]          # 表存的就是 log1p(count)，这里不再套第二遍
    cnt1 = np.expm1(lp1)
    bad = np.abs(cnt1 - np.round(cnt1)).max() if len(cnt1) else 0.0
    assert bad < 1e-3, f"{ds}: 反解出的计数不是整数（最大偏差 {bad:.4f}）⇒ 表的口径不是 log1p(count)"
    al = ALPHA[ds]
    gap = al * (lp1 - lp2)                    # 先验项能把 top1 拉开多少 logit
    bias_max = al * lp1
    # 表 argmax 是否就是 gold（只算"表非空"的边；§2.4 那个 0.4780 还含"空行回退全局众数"，口径不同）
    # 列序 = tail 实体 id 升序、以【train∪valid∪test 的 tail 全集】为基准（与 exporter 同一约定；
    # maas 的 test 只出现 27/30 个卡型，只用 test 定列会把整行平移）
    tails = np.unique(np.concatenate([rr[:, 2], r[:, 2]]))
    tail_lo = int(tails.min())
    assert tails.tolist() == list(range(tail_lo, tail_lo + len(tails))), \
        f"{ds}: rel{REL[ds]} 的 tail 段不连续"
    assert len(tails) == n_types, f"{ds}: tail 全集 {len(tails)} != cube 列数 {n_types}"
    gold_col = tc - tail_lo
    tbl_hit = (lp1 > 0) & (rows.argmax(axis=1) == gold_col)
    print(f"[{ds}] cube={cube.shape} 快照数={len(snaps)} test 边={len(hd)}  "
          f"复发(tot>0)={int(is_rec.sum())} ({is_rec.mean()*100:.2f}%)  "
          f"全新(tot=0)={int((~is_rec).sum())}  [门禁] 表非空==有严格更早历史 OK  "
          f"[计数整数性 max|Δ|={bad:.2e}]")
    print(f"[{ds}] '表非空且答对' 全体={tbl_hit.mean():.6f} 复发="
          f"{(tbl_hit[is_rec].mean() if is_rec.any() else float('nan')):.6f}"
          f"（≠§2.4 的闭式查表 H@1，那个含空行回退全局众数）")
    out[ds] = {
        "cube_shape": list(cube.shape), "alpha": al, "n_test": int(len(hd)),
        "n_recurrent": int(is_rec.sum()), "rec_share": float(is_rec.mean()),
        "tbl_hit_all": float(tbl_hit.mean()),
        "tbl_hit_rec": float(tbl_hit[is_rec].mean()) if is_rec.any() else None,
        "row_total_rec": q(tot[is_rec]), "count_top1_rec": q(cnt1[is_rec]),
        "bias_max_rec": q(bias_max[is_rec]), "bias_gap_top2_rec": q(gap[is_rec]),
        "share_cnt_eq_1_rec": float((np.round(cnt1[is_rec]) == 1).mean()),
        "share_gap_le_1_rec": float((gap[is_rec] <= 1.0).mean()),
    }
    for nm in ("全体", "复发"):
        m = np.ones(len(hd), dtype=bool) if nm == "全体" else is_rec
        print(f"    {nm:3s} 行内总计数(log1p 求和, 仅示意) 中位={np.median(tot[m]):.2f} | "
              f"最大格反解计数 count1 中位={np.median(cnt1[m]):.1f} "
              f"p90={np.percentile(cnt1[m],90):.1f} 最大={cnt1[m].max():.0f} | "
              f"先验项 α·log1p(count1) 中位={np.median(bias_max[m]):.3f}")
    print(f"    复发层里 count1==1 的占比={out[ds]['share_cnt_eq_1_rec']:.4f}  "
          f"先验 top1-vs-top2 拉开 ≤1.0 logit 的占比={out[ds]['share_gap_le_1_rec']:.4f}")

os.makedirs(REP, exist_ok=True)
p = os.path.join(REP, "probe_xcad_prior_bias_scale.json")
with open(p, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n[written] {p}")
print("[done] 只量表本身的计数与加成量级；模型段内分数差需 GPU 重跑，本脚本不含。")
