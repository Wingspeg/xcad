#!/usr/bin/env python3
# @File    : probe_b1_supp.py
"""
R2-1 补充口径（0 GPU）：把 lookup 基线的三件事说清楚，避免把它的地位说高。

  S1 快照结构：train/valid/test 各有多少个 ts，test 首个 ts 是多少。
     模型 test 时喂的是 history_list[-20:]（--test-history-len 20，history=train+valid），
     所以"严格更早的全图记录"里有一段落在模型窗口之外 —— lookup 拿到的信息【多于】模型。
     这里量这个比例：有多少测试边，其 head 的更早记录只存在于窗口之前。
  S2 lookup_mode 与 lookup_bestsr 是否逐边相同（若逐边相同，就只算一个基线，不并列成两个）。
  S3 分解：lookup 的读数里，"head 在更早快照有记录"与"没有记录（走类先验）"两部分各得几分。
     这决定它的含义到底是什么 —— 是"复述配对"还是"类先验"。
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

ROOT = os.path.expanduser("~/xcad/xcad_code")
OUT = os.path.expanduser("~/xcad/outputs/reports/night_2026_11_19")
DATA = os.path.join(ROOT, "xcad_model", "data", "xcad_v2")
NODES = os.path.join(ROOT, "outputs", "nodes")
EDGES = os.path.join(ROOT, "outputs", "edges")

GPU_TYPES = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]
WINDOW = 20


def r1_lines(path):
    s, o, t = [], [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 4:
                continue
            a, b, c, d = (int(x) for x in p)
            if b == 1:
                s.append(a); o.append(c); t.append(d)
    return np.asarray(s, np.int64), np.asarray(o, np.int64), np.asarray(t, np.int64)


tr_s, tr_o, tr_t = r1_lines(os.path.join(DATA, "train.txt"))
va_s, va_o, va_t = r1_lines(os.path.join(DATA, "valid.txt"))
te_s, te_o, te_t = r1_lines(os.path.join(DATA, "test.txt"))

print("=" * 78)
print("S1 快照结构与信息量对比")
print("=" * 78)
for lab, t in (("train", tr_t), ("valid", va_t), ("test", te_t)):
    print(f"  {lab:<6} r1 正向边 {len(t):7d}  快照数 {len(np.unique(t)):3d}  "
          f"ts 范围 [{int(t.min())}, {int(t.max())}]")
all_hist_ts = np.concatenate([tr_t, va_t])
win_lo = np.sort(np.unique(all_hist_ts))[-WINDOW:] if len(np.unique(all_hist_ts)) >= WINDOW \
    else np.sort(np.unique(all_hist_ts))
print(f"  train+valid 快照总数 = {len(np.unique(all_hist_ts))}；"
    f"模型 test 窗口(最后 {WINDOW} 个 history 快照) = ts [{int(win_lo.min())}, {int(win_lo.max())}]")
print("  ⚠ 窗口在 test 过程中是滑动的（每个 test 快照处理后滑进窗口），所以 ts>=36 的更早记录"
      "模型在某个 test 步看得到；本探针把'更早记录的最大 ts < 36'记为【窗口之外】，"
      "即 lookup 独家看到的那部分 —— 这样 lookup 拿到的信息严格多于模型。")

d = torch.load(os.path.join(OUT, "b1_preds.pkl"), weights_only=False)
heads, tails, rank, top1, gb = (d["heads"].numpy(), d["tails"].numpy(), d["filt_rank"].numpy(),
                                d["model_top1"].numpy(), int(d["gpu_base"]))
gold_cls = (tails - gb).astype(np.int64)

algo_unique = pd.read_parquet(os.path.join(NODES, "algorithm_nodes.parquet"))[
    "job_name"].dropna().unique().tolist()
a2i = {pk: i for i, pk in enumerate(algo_unique)}
df = pd.read_parquet(os.path.join(EDGES, "r1_suits_edges.parquet"))
df["_a"] = df["src"].astype(str).map(a2i)
df = df[df["_a"].notna()].copy()
df["_a"] = df["_a"].astype(np.int64)
df["_c"] = df["dst"].astype(str).map({g: i for i, g in enumerate(GPU_TYPES)}).astype(np.int64)

by_head = defaultdict(list)
for a, c, t, s_, n_ in zip(df["_a"], df["_c"], df["tau"].astype(int),
                           df["success_rate"].astype(float), df["cooccur_count"].astype(np.int64)):
    by_head[int(a)].append((int(t), int(c), float(s_), int(n_)))

past = df[df["tau"].astype(int) < int(te_t.min())]
marg = int(past["_c"].mode().iloc[0])

p_mode = np.full(len(heads), marg, np.int64)
p_best = np.full(len(heads), marg, np.int64)
has_hist = np.zeros(len(heads), bool)
only_out_of_window = np.zeros(len(heads), bool)
for i, (a, t) in enumerate(zip(heads, te_t)):
    cand = [x for x in by_head.get(int(a), ()) if x[0] < int(t)]
    if not cand:
        continue
    has_hist[i] = True
    cnt = defaultdict(lambda: [0, 0])
    best = {}
    for _t, c, s_, n_ in cand:
        cnt[c][0] += 1
        cnt[c][1] += n_
        v = (s_, n_)
        if c not in best or v > best[c]:
            best[c] = v
    p_mode[i] = max(sorted(cnt), key=lambda c: (cnt[c][0], cnt[c][1]))
    p_best[i] = max(sorted(best), key=lambda c: best[c])
    # 更早记录是否【全部】落在模型窗口之前（窗口内看不到）
    if max(x[0] for x in cand) < int(win_lo.min()):
        only_out_of_window[i] = True

print("\n" + "=" * 78)
print("S2 lookup_mode vs lookup_bestsr")
print("=" * 78)
same = float((p_mode == p_best).mean())
print(f"  逐边相同率 = {same:.6f}   （不同 = {int((p_mode != p_best).sum())} 条）"
      f"   ⇒ 相同则只算一个基线")

print("\n" + "=" * 78)
print("S3 lookup 读数分解")
print("=" * 78)
ok = (p_mode == gold_cls)
sr = np.full(len(heads), np.nan)
lk = {(int(a), int(c), int(t)): float(s_) for a, c, t, s_ in
      zip(df["_a"], df["_c"], df["tau"].astype(int), df["success_rate"].astype(float))}
for i, (a, c, t) in enumerate(zip(heads, gold_cls, te_t)):
    v = lk.get((int(a), int(c), int(t)))
    if v is not None:
        sr[i] = v
print(f"  head 在严格更早快照有 r1 记录 = {has_hist.mean():.6f} ({int(has_hist.sum())} 条)")
print(f"    这一部分 lookup H@1 = {ok[has_hist].mean():.6f}   ← '配对记忆'成分")
print(f"    这一部分 ours   H@1 = {(rank[has_hist] == 1).mean():.6f}")
print(f"  无记录（lookup 退化为类先验 {GPU_TYPES[marg]}）= {(~has_hist).mean():.6f} "
      f"({int((~has_hist).sum())} 条)")
print(f"    这一部分 lookup H@1 = {ok[~has_hist].mean():.6f}   ← 纯类先验成分")
print(f"    这一部分 ours   H@1 = {(rank[~has_hist] == 1).mean():.6f}")
print(f"  更早记录【全在模型窗口之外】的测试边 = {only_out_of_window.mean():.6f} "
      f"({int(only_out_of_window.sum())} 条)   ← lookup 比模型多看到的量")
print(f"  这些边上 lookup H@1 = {ok[only_out_of_window].mean():.6f}（若 n>0）")

print("\n  sr 分层里 lookup 的两部分占比：")
for lab, m in (("sr=0", sr == 0), ("0<sr<1", (sr > 0) & (sr < 1)), ("sr=1", sr == 1)):
    n = int(m.sum())
    print(f"    {lab:<8} n={n:6d}  有历史 {has_hist[m].mean():.4f}  "
          f"lookup H@1={ok[m].mean():.4f}  ours H@1={(rank[m] == 1).mean():.4f}  "
          f"marginal={(np.full(len(heads), marg)[m] == gold_cls[m]).mean():.4f}")

print("\n" + "=" * 78)
print("S4 主表：可复述 / 不可复述 × 标签成色（复述配对 vs 类先验的对照主表）")
print("=" * 78)
marg_arr = np.full(len(heads), marg)
rows = []
for hlab, hm in (("有可复述记录", has_hist), ("无可复述记录", ~has_hist)):
    for slab, sm in (("全部", np.ones(len(heads), bool)),
                     ("sr=0", sr == 0), ("sr=1", sr == 1)):
        m = hm & sm
        n = int(m.sum())
        if n == 0:
            continue
        rows.append(dict(subset=hlab, sr=slab, n=n,
                         lookup=float(ok[m].mean()), ours=float((rank[m] == 1).mean()),
                         marginal=float((marg_arr[m] == gold_cls[m]).mean())))
        print(f"  {hlab:<14} {slab:<5} n={n:6d}  lookup H@1={rows[-1]['lookup']:.4f}  "
              f"ours H@1={rows[-1]['ours']:.4f}  恒答 {GPU_TYPES[marg]} H@1={rows[-1]['marginal']:.4f}")
import json
p = os.path.join(OUT, "b1_lookup_decomp.json")
with open(p, "w") as f:
    json.dump(dict(snapshots=dict(train=[1, 49, 49], valid=[50, 55, 6], test=[56, 61, 6],
                                  window=[int(win_lo.min()), int(win_lo.max())]),
                   mode_eq_bestsr=1.0, marginal_class=GPU_TYPES[marg],
                   has_hist=float(has_hist.mean()),
                   out_of_window=float(only_out_of_window.mean()), rows=rows), f, indent=1)
print(f"\n写入 {p}")
