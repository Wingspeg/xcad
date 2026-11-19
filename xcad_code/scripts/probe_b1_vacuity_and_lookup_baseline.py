#!/usr/bin/env python3
# @File    : probe_b1_vacuity_and_lookup_baseline.py
"""
B1 后续（0 GPU）：把分层 C 的 0.0000 读数【查死】，并给 R2-1 换一个不恒等的证据。

为什么要这个探针
----------------
analyze_b1_label_support.py 的分层 C 打印了"模型答错时，预测配对在 r1 图里出现过的
比例 = 0.0000"。它看起来像否证 R2-1 的决定性证据，但两条代码事实说明它是
构造必然，不是发现：
  (i)  查表键是 (head, gpu 类型, tau)，而分层 E 已测出全图 109,660 个 (head, tau) 键里
       109,653 个只有 1 个候选类型 ⇒ "该 head 在该快照声明过的类型"只有真值那一个；
  (ii) b1_preds.pkl 的 model_top1 来自 rrgcn.py:702 的 score.argmax(1)，而 score 在
       684 行已被 filter_score 就地改成 -inf ⇒ 答错(top1≠gold)的边，top1 不可能是
       任何"被声明且被过滤掉"的配对。
本探针 G1 段把这两条实测出来（含 rank>1 且 top1==gold 的边数应为 0）。

真正能回应 R2-1 的问题不是"错到哪条边"，而是【把答案拿掉之后，光背记录能得几分】。
所以 G2 段构造两个不学习的"复述历史"基线，用与模型完全相同的 11,836 条测试边、
相同的真值类打分，并按同样的 success_rate 分层重报：
    lookup_mode   : 该 head 在【严格更早】快照里声明过的类型里，出现次数最多的那个
    lookup_bestsr : 该 head 在严格更早快照里 success_rate 最高的那个类型
                    （并列时取 cooccur_count 大者）—— 这是"复述调度器成功配对"的最强形式
    class_marginal: 全体历史上边际最常见的类型（恒定量预测器）
关键口径：这两个基线看的是【全图严格更早的 r1 边】（含 train/valid 与更早的 test 快照），
比模型实际能看到的 20 快照窗口更宽松 —— 故意给"背记录"这个解释最大好处，再和模型比。

G3 段给类级成色：每种 GPU 类型在全图的 success_rate=1 占比，以及"答错时预测类的成色
是否高于真值类的成色"（= "标签偏向好配对"这一说法在类级上的可测形式）。

只读脚本：不训练、不写任何数据产物，只打印 + 落一份 json 到 outputs/reports/。
"""
import json
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
PKL = os.path.join(OUT, "b1_preds.pkl")

GPU_TYPES = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]
N_CLS = 5                     # r1 真值永不是 CPU ⇒ macro 在 T4..V100M32 上做
PUB = dict(mrr=0.927251, h1=0.884843, macro=0.5085)   # seed 42 门禁（同 B1）
THIN = 50


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def read_txt(path):
    """RE-GCN 格式 (s, r, o, ts)；返回 rel_id==1 的正向 r1_suits 边。"""
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


def acc_report(pred_cls, gold_cls, sr, tag, out):
    """pred/gold: 1..5（T4..V100M32），0 = 判不出来（CPU 段/无候选）。sr: 真值成色。"""
    row = {"tag": tag}
    ok = (pred_cls == gold_cls)
    line = f"  {tag:<16} 全体 H@1={ok.mean():.6f}"
    row["all"] = float(ok.mean())
    for lab, msk in (("sr=0", sr == 0), ("0<sr<1", (sr > 0) & (sr < 1)), ("sr=1", sr == 1)):
        n = int(msk.sum())
        v = float(ok[msk].mean()) if n else float("nan")
        line += f"   {lab} H@1={v:.4f} (n={n})"
        row[lab] = v
        row[lab + "_n"] = n
    # macro over the 5 gold classes (occurrence classes only)
    hits, ns = [], []
    for k in range(1, 6):
        m = gold_cls == k
        if m.sum() == 0:
            continue
        hits.append(float(ok[m].mean())); ns.append(int(m.sum()))
    line += f"   macro{len(hits)}={np.mean(hits):.4f}" if hits else "   macro=n/a"
    row["macro"] = float(np.mean(hits)) if hits else float("nan")
    row["n_classes"] = len(hits)
    print(line + ("   [段样本 < 50 只报数]" if any(row[k + "_n"] < THIN for k in ("sr=0", "0<sr<1")) else ""))
    out.append(row)
    return row


def main():
    d = torch.load(PKL, weights_only=False)
    heads = d["heads"].numpy().astype(np.int64)
    tails = d["tails"].numpy().astype(np.int64)
    rank = d["filt_rank"].numpy().astype(np.int64)
    top1 = d["model_top1"].numpy().astype(np.int64)
    gb = int(d["gpu_base"])
    gold_cls = (tails - gb)                       # 1..5
    pred_cls = np.where((top1 >= gb) & (top1 < gb + 6), top1 - gb, 0)

    algo_unique = pd.read_parquet(os.path.join(NODES, "algorithm_nodes.parquet"))[
        "job_name"].dropna().unique().tolist()
    a2i = {pk: i for i, pk in enumerate(algo_unique)}

    df = pd.read_parquet(os.path.join(EDGES, "r1_suits_edges.parquet"))
    df["_a"] = df["src"].astype(str).map(a2i)
    df = df[df["_a"].notna()].copy()
    df["_a"] = df["_a"].astype(np.int64)
    df["_c"] = df["dst"].astype(str).map({g: i for i, g in enumerate(GPU_TYPES)}).astype(np.int64)

    lk = {}
    for a, c, t, sr_, cc_ in zip(df["_a"], df["_c"], df["tau"].astype(int),
                               df["success_rate"].astype(float), df["cooccur_count"].astype(np.int64)):
        lk[(int(a), int(c), int(t))] = (float(sr_), int(cc_))

    # ---- 测试边的 tau 从 test.txt 读（pkl 不带 ts），契约：行序与 pkl 逐位相同 ----
    ts_s, ts_o, ts_t = read_txt(os.path.join(DATA, "test.txt"))
    assert len(ts_s) == len(heads) and bool((ts_s == heads).all()) and bool((ts_o == tails).all()), \
        "test.txt 的 r1 行序与 b1_preds.pkl 不一致，本探针作废"
    tt = ts_t
    tau_min_test = int(tt.min())
    sr = np.array([lk.get((int(a), int(c), int(t)), (np.nan, -1))[0]
                   for a, c, t in zip(heads, gold_cls, tt)])
    ccnt = np.array([lk.get((int(a), int(c), int(t)), (np.nan, -1))[1]
                     for a, c, t in zip(heads, gold_cls, tt)])
    known = ccnt >= 0
    print(f"契约：n={len(heads)}  支持度查表命中={known.mean():.6f}  gpu_base={gb}  "
          f"test 首个快照 ts={int(tt.min())}")

    # ────────────────────────── G1 恒等性检查 ──────────────────────────
    sec("G1 分层 C 的 0.0000 是不是构造必然")
    wrong = rank > 1
    print(f"  rank>1（答错）的边 n={int(wrong.sum())}；其中 model_top1 == gold 的边数 = "
          f"{int((wrong & (top1 == tails)).sum())}   ← 必须为 0，否则 filtered 口径有问题")
    ncan = df.groupby(["_a", "tau"], sort=False)["_c"].nunique()
    key = pd.MultiIndex.from_arrays([heads, tt], names=["_a", "tau"])
    per_edge = ncan.reindex(key).to_numpy().astype(float)
    print(f"  测试边所在 (head, ts) 键的候选类型数分布 = "
          f"{pd.Series(per_edge).value_counts().to_dict()}   ← 1 个 ⇒ 该键唯一合法答案就是真值")
    in_graph_key = np.array([(int(a), int(p), int(t)) in lk for a, p, t in zip(heads, pred_cls, tt)])
    print(f"  同键口径（分层 C 用的口径）：答错边上'预测配对在图里出现过' = "
          f"{in_graph_key[wrong].mean():.6f}   答对边上 = {in_graph_key[~wrong].mean():.6f}")
    # 去掉 ts 的非恒等版本：该 head 在【别的快照】有没有用过这个类型
    any_of_head = defaultdict(set)
    for a, c in zip(df["_a"], df["_c"]):
        any_of_head[int(a)].add(int(c))
    ever = np.array([int(p) in any_of_head.get(int(a), set()) for a, p in zip(heads, pred_cls)])
    ever_gold = np.array([int(g) in any_of_head.get(int(a), set()) for a, g in zip(heads, gold_cls)])
    n_head_multi = float(np.mean([len(any_of_head.get(int(a), set())) > 1 for a in heads]))
    print(f"  跨快照口径（唯一非恒等的问法）：答错边上'预测类型在该 head 用过的类型集合里' = "
          f"{ever[wrong].mean():.6f}（该集合含 >1 个类型的 head 占 {n_head_multi:.6f}，"
          f"gold 永远在里面：{ever_gold.mean():.6f}）")
    print("  ⇒ 分层 C 的两条读数都由'每键单候选'决定，不足以作为否证证据。")

    # ────────────────────────── G2 复述历史基线 ──────────────────────────
    sec("G2 把答案拿掉之后，光背记录能得几分（与模型同一批 11,836 条边）")
    by_head = defaultdict(list)
    for a, c, t, s_, n_ in zip(df["_a"], df["_c"], df["tau"].astype(int),
                               df["success_rate"].astype(float), df["cooccur_count"].astype(np.int64)):
        by_head[int(a)].append((int(t), int(c), float(s_), int(n_)))
    # 类先验：严格早于 test 首快照的边里最常见的类型
    past = df[df["tau"].astype(int) < tau_min_test]
    marg = int(past["_c"].mode().iloc[0]) if len(past) else 1
    past_sr1 = float((past["success_rate"].astype(float) == 1).mean()) if len(past) else float("nan")
    print(f"  类先验（train+valid+更早快照里最常见的 r1 tail 类型）= {GPU_TYPES[marg]}，"
          f"其中 success_rate==1 的边占 {past_sr1:.4f}")

    p_mode = np.zeros(len(heads), np.int64)
    p_best = np.zeros(len(heads), np.int64)
    has_hist = np.zeros(len(heads), bool)
    for i, (a, t) in enumerate(zip(heads, tt)):
        cand = [x for x in by_head.get(int(a), ()) if x[0] < int(t)]
        if not cand:
            p_mode[i] = marg; p_best[i] = marg
            continue
        has_hist[i] = True
        cnt = defaultdict(lambda: [0, 0.0])
        for _t, c, s_, n_ in cand:
            cnt[c][0] += 1
            cnt[c][1] += n_
        p_mode[i] = max(sorted(cnt), key=lambda c: (cnt[c][0], cnt[c][1]))
        best_sr = {}
        for _t, c, s_, n_ in cand:
            k = c
            v = (s_, n_)
            if k not in best_sr or v > best_sr[k]:
                best_sr[k] = v
        p_best[i] = max(sorted(best_sr), key=lambda c: best_sr[c])
    print(f"  该 head 在严格更早快照里有 r1 记录 = {has_hist.mean():.6f}"
          f"（无记录的走类先验）；模型 H@1 = {float((rank == 1).mean()):.6f}")
    rows = []
    acc_report(np.full(len(heads), marg), gold_cls, sr, "class_marginal", rows)
    acc_report(p_mode, gold_cls, sr, "lookup_mode", rows)
    acc_report(p_best, gold_cls, sr, "lookup_bestsr", rows)
    acc_report(pred_cls, gold_cls, sr, "ours(seed42)", rows)
    print(f"  注：lookup_bestsr 在'sr=0'层天然答不对——该层真值本身就是零成功配对，"
          f"而它只挑有成功记录的类型；这正是'复述调度器'能解释到哪里的边界。")

    # ────────────────────────── G3 类级成色 ──────────────────────────
    sec("G3 类级成色：答错时是否偏向历史上更成功的类型")
    g = df.groupby("_c").agg(n=("_c", "size"),
                             frac_sr1=("success_rate", lambda x: float((x == 1).mean())),
                             med_cc=("cooccur_count", "median"))
    test_cnt = pd.Series(gold_cls).value_counts()
    pred_cnt = pd.Series(pred_cls[wrong]).value_counts()
    print(f"  {'类型':>8} {'图内边数':>10} {'sr=1 占比':>10} {'cc 中位':>8} "
          f"{'test gold':>10} {'答错时预测':>10}")
    for k in range(1, 6):
        r_ = g.loc[k] if k in g.index else None
        print(f"  {GPU_TYPES[k]:>8} {int(r_['n']) if r_ is not None else 0:>10} "
              f"{float(r_['frac_sr1']) if r_ is not None else float('nan'):>10.4f} "
              f"{int(r_['med_cc']) if r_ is not None else 0:>8} "
              f"{int(test_cnt.get(k, 0)):>10} {int(pred_cnt.get(k, 0)):>10}")
    cls_sr = {k: float(g.loc[k, "frac_sr1"]) for k in g.index}
    w = wrong & (pred_cls > 0)
    drift = float(np.mean([cls_sr.get(int(p), np.nan) - cls_sr.get(int(gc), np.nan)
                           for p, gc in zip(pred_cls[w], gold_cls[w])]))
    print(f"  答错边上：预测类的历史 sr=1 占比 − 真值类的 = {drift:+.4f}"
          f"   (>0 才支持'偏向好配对'，且这是类级、非逐边)")
    # 模型 vs 各复述基线在 sr=0 层的差
    ours0 = float((rank == 1)[sr == 0].mean())
    print(f"  sr=0 层：ours H@1={ours0:.4f}  "
          f"lookup_mode={float((p_mode == gold_cls)[sr == 0].mean()):.4f}  "
          f"lookup_bestsr={float((p_best == gold_cls)[sr == 0].mean()):.4f}  "
          f"marginal={float((np.full(len(heads), marg) == gold_cls)[sr == 0].mean()):.4f}")

    out = dict(g1=dict(n_wrong=int(wrong.sum()),
                       wrong_with_top1_eq_gold=int((wrong & (top1 == tails)).sum()),
                       cand_per_key=pd.Series(per_edge).value_counts().to_dict(),
                       pair_in_graph_same_key_wrong=float(in_graph_key[wrong].mean()),
                       type_ever_used_wrong=float(ever[wrong].mean()),
                       heads_with_multi_types=float(n_head_multi)),
               g2=rows, g3=dict(cls_table=g.to_dict("index"), drift=drift,
                                marginal=GPU_TYPES[marg]),
               meta=dict(n=len(heads), gpu_base=gb, tau_min_test=int(tau_min_test),
                         has_hist=float(has_hist.mean())))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "b1_vacuity_and_lookup_baseline.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\n写入 {p}")


if __name__ == "__main__":
    main()
