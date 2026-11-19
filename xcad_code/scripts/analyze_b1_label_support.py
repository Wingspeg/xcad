#!/usr/bin/env python3
# @File    : analyze_b1_label_support.py
"""
B1 / R2-1：把 11,836 条 r1 测试边按【标签支持度】分层重报指标。

R2-1 提出的替代解释是：r1_suits 的 "suits" 标签其实是历史调度器的偏置
（成功次数少 / 从没成功过的配对也进了图）。要检验这句，需要知道
  (a) 测试边里有多少条来自零成功历史（success_rate=0）、低支持度（cooccur=1）；
  (b) 模型在这些边上是不是"照样答对"或"明显更差"。
两者都只需要一次 test-only + --dump-preds（已跑，见 b1_preds.pkl）＋ CPU 后处理，
不用训练、不用新 GPU。

口径契约（先过门禁再谈分层）
    1) b1_preds.pkl 的 (heads, tails) 序列必须与 test.txt 里 rel_id==1 行的读序
       逐位相同 —— 这是"边身份"的来源，不过就整份分析作废。
    2) 全量 11,836 条重算的 MRR / H@1 / macro5 必须逐位复现手稿读数
       0.927251 / 0.884843 / 0.5085（seed 42, b1_dump.log）—— 不过说明本脚本
       与评测器不同口径，不许报分层。
    3) 支持度查表的键用 export_to_regcn_format.build_entity_space 的同一句
       `algo_df["job_name"].dropna().unique().tolist()` + enumerate 得到的局部 idx，
       与 export_edge_weights.build_algo_ids 严格一致；算法段 offset=0，
       故 head 的全局 id 就等于这个局部 idx，可以直接查。

预注册判线（在看到数字前写死）
    L-B1-a：主口径改为"仅在 success_rate=1 的边上评估"当且仅当
             |macro5(sr=0) - macro5(sr=1)| >= 0.02。
             差 < 0.02 ⇒ 写"标签成色不改变结论（实测 10.38% 的测试边来自零成功历史）"，
             保留全量口径。
    L-B1-b：0.02 这个阈值来自 v2020 全量 r1 H@1 的 run-to-run 噪声 0.0172；
             分层后 n 变小、抽样噪声变大，所以同时给 macro 差的 bootstrap 95% CI，
             并用 CI 而不是点估计说话。
    L-B1-c：支持度侧看 cooccur_count 是否单调：若 支持度=1 段与 支持度>=10 段的
             H@1 差 >= 0.02，则"模型只是在背历史配对"这句有正面证据；否则没有。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.expanduser("~/xcad/xcad_code")
# 注意两个 outputs：代码树里是中间产物（nodes/edges parquet），
# 报表在仓库根的 ~/xcad/outputs/reports/ 下。
OUT = os.path.expanduser("~/xcad/outputs/reports/night_2026_11_19")
DATA = os.path.join(ROOT, "xcad_model", "data", "xcad_v2")
NODES = os.path.join(ROOT, "outputs", "nodes")
EDGES = os.path.join(ROOT, "outputs", "edges")

GPU_TYPES = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]
CLS = GPU_TYPES[1:]                    # r1 真值永不是 CPU ⇒ macro 在 5 类上做
N_TEST_R1_EXPECT = 11836
PUB = dict(mrr=0.927251, h1=0.884843, macro=0.5085)   # seed 42, b1_dump.log
NOISE_FLOOR = 0.02
THIN = 50                              # 段内样本量低于此只报数不作结论
SEED = 20261119
NBOOT = 2000


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ─────────────────────────── 边身份 ───────────────────────────
def parse_test_r1(path):
    s, o, t = [], [], []
    n_lines = 0
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 4:
                continue
            n_lines += 1
            a, b, c, d = (int(x) for x in p)
            if b == 1:                      # R1_REL_ID == 1 == r1_suits
                s.append(a); o.append(c); t.append(d)
    return (np.asarray(s, dtype=np.int64), np.asarray(o, dtype=np.int64),
            np.asarray(t, dtype=np.int64), n_lines)


def build_algo_ids():
    """与 export_to_regcn_format.py:90 完全一致的那一句。"""
    df = pd.read_parquet(os.path.join(NODES, "algorithm_nodes.parquet"))
    return df["job_name"].dropna().unique().tolist()


def build_support_lookup(algo_unique):
    """(algo_idx, gpu_name, tau) -> (success_rate, cooccur_count)。"""
    df = pd.read_parquet(os.path.join(EDGES, "r1_suits_edges.parquet"))
    a2i = {pk: i for i, pk in enumerate(algo_unique)}
    aa = df["src"].astype(str).map(a2i).to_numpy()
    bad = int(np.isnan(aa.astype(float)).sum())
    kk = list(zip(df["src"].astype(str), df["dst"].astype(str), df["tau"].astype(int)))
    lk = {}
    for key, sr, cc, ai in zip(kk, df["success_rate"].astype(float).values,
                               df["cooccur_count"].astype(np.int64).values, aa):
        lk[(int(ai), key[1], key[2])] = (float(sr), int(cc))
    return lk, len(df), bad, dict(
        n_rows=len(df), n_keys=len(lk),
        n_zero_sr=int((df.success_rate == 0).sum()),
        n_open_sr=int(((df.success_rate > 0) & (df.success_rate < 1)).sum()),
        n_one_sr=int((df.success_rate == 1).sum()),
        frac_zero_sr=float((df.success_rate == 0).mean()),
        frac_cc1=float((df.cooccur_count == 1).mean()),
        cc_p50=int(df.cooccur_count.quantile(.5)), cc_p90=int(df.cooccur_count.quantile(.9)),
        cc_max=int(df.cooccur_count.max()))


# ─────────────────────────── 指标 ───────────────────────────
def metrics(rank, cls):
    """rank: 1-based filtered rank; cls: 0..4 = T4..V100M32。"""
    n = int(len(rank))
    if n == 0:
        return None
    r = np.asarray(rank); c = np.asarray(cls)
    out = dict(n=n, mrr=float((1.0 / r).mean()),
               h1=float((r <= 1).mean()), h3=float((r <= 3).mean()),
               h10=float((r <= 10).mean()))
    per = {}
    cnt = np.bincount(c, minlength=5)
    hit = np.bincount(c, weights=(r <= 1).astype(float), minlength=5)
    for k, name in enumerate(CLS):
        per[name] = (float(hit[k] / cnt[k]) if cnt[k] > 0 else None, int(cnt[k]))
    out["per"] = per
    occ = [v for v, nn in per.values() if nn > 0]
    out["macro_occ"] = float(np.mean(occ)) if occ else None
    out["c_occ"] = len(occ)
    out["macro5"] = float(np.mean([v for v, nn in per.values()])) if all(
        nn > 0 for _, nn in per.values()) else None
    out["best_const_h1"] = float(cnt.max() / n)
    out["maj_class"] = CLS[int(np.argmax(cnt))]
    return out


def macro_occ_np(rank, cls):
    cnt = np.bincount(cls, minlength=5)
    hit = np.bincount(cls, weights=(rank <= 1).astype(float), minlength=5)
    m = cnt > 0
    return float((hit[m] / cnt[m]).mean()) if m.any() else np.nan


def boot_diff(r0, c0, r1, c1, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n0, n1 = len(r0), len(r1)
    d = np.empty(nboot)
    for b in range(nboot):
        i = rng.integers(0, n0, n0); j = rng.integers(0, n1, n1)
        d[b] = macro_occ_np(r0[i], c0[i]) - macro_occ_np(r1[j], c1[j])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(d.std(ddof=1))


def fmt(m):
    if m is None:
        return "        -"
    return (f"n={m['n']:6d}  MRR={m['mrr']:.6f}  H@1={m['h1']:.6f}  H@3={m['h3']:.4f} "
            f"H@10={m['h10']:.4f}  macro5={m['macro5'] if m['macro5'] is not None else float('nan'):.4f}"
            f"  macro_occ={m['macro_occ']:.4f}(c={m['c_occ']})  best_const={m['best_const_h1']:.4f}"
            f"[{m['maj_class']}]")


def print_per(m):
    if m is None:
        return
    print("            " + "  ".join(
        (f"{k}={v:.3f}/{nn}" if v is not None else f"{k}=NA/{nn}") for (v, nn), k in
        [(m['per'][k], k) for k in CLS]))


def band_table(title, rank, cls, bands):
    """bands: [(label, mask), ...]"""
    sec(title)
    res = {}
    for lab, msk in bands:
        m = metrics(rank[msk], cls[msk])
        res[lab] = m
        print(f"  {lab:<28} {fmt(m)}")
        print_per(m)
    return res


def build_head_tau_best(algo_unique):
    """(algo_idx, tau) -> 该 head 在该快照下【成功率先、支持度次】的 GPU 类型。
    用来把"trace 记录的是调度器实际做了什么"与"历史上最说得过去的配对"分开。"""
    df = pd.read_parquet(os.path.join(EDGES, "r1_suits_edges.parquet"))
    a2i = {pk: i for i, pk in enumerate(algo_unique)}
    ai = df["src"].astype(str).map(a2i).to_numpy()
    best = {}
    for a, g, t, sr, cc in zip(ai, df["dst"].astype(str), df["tau"].astype(int),
                               df["success_rate"].astype(float),
                               df["cooccur_count"].astype(np.int64)):
        if pd.isna(a):
            continue
        k = (int(a), int(t))
        cur = best.get(k)
        if cur is None or (float(sr), int(cc)) > (cur[1], cur[2]):
            best[k] = (str(g), float(sr), int(cc))
    return best


def main():
    print(f"python={sys.version.split()[0]}  pandas={pd.__version__}  torch={torch.__version__}")

    # ---- 契约 1：边身份 ----
    d = torch.load(os.path.join(OUT, "b1_preds.pkl"), map_location="cpu")
    heads = d["heads"].numpy().astype(np.int64)
    tails = d["tails"].numpy().astype(np.int64)
    rank = d["filt_rank"].numpy().astype(np.int64)
    gb = int(d["gpu_base"])
    ts, to, tt, n_lines = parse_test_r1(os.path.join(DATA, "test.txt"))
    sec("契约检查")
    print(f"b1_preds.pkl  keys={sorted(d.keys())}  n={len(heads)}  gpu_base={gb}")
    print(f"test.txt      总行数={n_lines}  rel_id==1 行数={len(ts)}")
    same = len(ts) == len(heads) and bool((ts == heads).all()) and bool((to == tails).all())
    print(f"[C1] (heads,tails) 与 test.txt r1 行逐位相同 = {same}")
    if not same:
        print("     heads[:10] =", heads[:10].tolist(), " test[:10] =", ts[:10].tolist())
        print("     tails[:10] =", tails[:10].tolist(), " test[:10] =", to[:10].tolist())
        sys.exit("C1 FAIL：边身份不成立，分层作废")
    cls = tails - gb
    print(f"[C1b] tail-gpu 类分布 = {np.bincount(cls, minlength=5).tolist()}"
          f"（期望 5032/5246/552/811/195）")
    assert cls.min() >= 1 and cls.max() <= 5, "有 r1 真值落在 CPU 或段外"
    cls5 = cls - 1                                        # T4..V100M32 -> 0..4

    # ---- 契约 2：全量口径复现手稿 ----
    allm = metrics(rank, cls5)
    ok = (abs(allm["mrr"] - PUB["mrr"]) <= 5e-7 and abs(allm["h1"] - PUB["h1"]) <= 5e-7
          and abs(allm["macro_occ"] - PUB["macro"]) <= 5e-5)
    print(f"[C2] 全量重算 MRR={allm['mrr']:.6f} H@1={allm['h1']:.6f} "
          f"macro_occ={allm['macro_occ']:.4f}  vs 手稿 {PUB['mrr']}/{PUB['h1']}/{PUB['macro']}"
          f"  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit("C2 FAIL：与评测器不同口径，不许报分层")

    # ---- 契约 3：支持度查表 ----
    algo_unique = build_algo_ids()
    lk, n_rows, n_bad, gstat = build_support_lookup(algo_unique)
    print(f"[C3] algo ids = {len(algo_unique)}（期望 102610）  heads 落段内 = "
          f"{bool(heads.max() < len(algo_unique))}")
    names = np.asarray([algo_unique[h] for h in heads])   # 只为人工抽查留底
    gpu_names = np.asarray(GPU_TYPES)[cls]
    sr = np.full(len(rank), np.nan); cc = np.full(len(rank), -1, dtype=np.int64)
    # lk 的键是 (局部 algo idx, GPU 名, tau)；算法段 offset=0 ⇒ head 全局 id == 局部 idx
    for i, (a, g, t) in enumerate(zip(heads, gpu_names, tt)):
        v = lk.get((int(a), str(g), int(t)))
        if v is not None:
            sr[i], cc[i] = v[0], v[1]
    hit = int((cc >= 0).sum())
    print(f"[C3] 全图 r1 边 {n_rows} 条：success_rate==0 占 {gstat['frac_zero_sr']:.4f}"
          f"、cooccur==1 占 {gstat['frac_cc1']:.4f}、cc p50/p90/max="
          f"{gstat['cc_p50']}/{gstat['cc_p90']}/{gstat['cc_max']}")
    print(f"[C3] 测试边命中查表 = {hit}/{len(rank)} ({hit/len(rank):.4f})，"
          f"未命中 {len(rank)-hit} 条（未命中者不进分层，单独报）")
    rep = {"contract": {"n_test_r1": int(len(rank)), "lookup_hit": hit,
                        "lookup_miss": int(len(rank) - hit), "graph_stats": gstat},
           "full": allm}

    unk = cc < 0
    if unk.any():
        m = metrics(rank[unk], cls5[unk])
        print(f"  [未命中查表]           {fmt(m)}")
        rep["lookup_miss_stratum"] = m

    # ---- 分层 1：success_rate ----
    b_sr = [("success_rate == 0", (sr == 0)),
            ("0 < success_rate < 1", (sr > 0) & (sr < 1)),
            ("success_rate == 1", (sr == 1))]
    rep["sr_strata"] = band_table("分层 A：success_rate（标签成色）", rank, cls5, b_sr)

    # ---- 分层 2：cooccur_count ----
    b_cc = [("cooccur == 1", cc == 1),
            ("cooccur 2-4", (cc >= 2) & (cc <= 4)),
            ("cooccur 5-9", (cc >= 5) & (cc <= 9)),
            ("cooccur 10-49", (cc >= 10) & (cc <= 49)),
            ("cooccur >= 50", cc >= 50)]
    rep["cc_strata"] = band_table("分层 B：cooccur_count（观测充分性）", rank, cls5, b_cc)

    # ---- 分层 C：模型在 sr=0 边上到底答成了什么 ----
    # 只看"记录的那次放置历史上从没成功过"的边：模型若系统性地把答案指向
    # 历史上成功过的另一个类型，那就是"模型学的是适配性、标签记的是调度行为"，
    # 正面回应 R2-1；若预测配对的成色并不更高，就只是普通错误，不能这么写。
    sec("分层 C：预测配对的标签成色 vs 真值配对的成色")
    top1 = d["model_top1"].numpy().astype(np.int64)
    in_seg = (top1 >= gb) & (top1 < gb + 6)
    print(f"无约束 top-1 落在 GPU 段内 = {float(in_seg.mean()):.6f}"
          f"（sr=0 段 {float(in_seg[sr == 0].mean()):.4f} / sr=1 段 {float(in_seg[sr == 1].mean()):.4f}）")
    pred_name = np.asarray(GPU_TYPES)[(top1 - gb).clip(0, 5)]
    pred_sr = np.full(len(rank), np.nan); pred_cc = np.full(len(rank), -1, dtype=np.int64)
    for i, (a, g, t) in enumerate(zip(heads, pred_name, tt)):
        v = lk.get((int(a), str(g), int(t)))
        if v is not None:
            pred_sr[i], pred_cc[i] = v[0], v[1]
    rep["C"] = {"in_seg_rate": float(in_seg.mean())}
    for lab, msk in [("sr=0 且答错", (sr == 0) & (rank > 1) & in_seg),
                     ("sr=0 且答对", (sr == 0) & (rank == 1) & in_seg),
                     ("sr=1 且答错", (sr == 1) & (rank > 1) & in_seg),
                     ("sr=1 且答对", (sr == 1) & (rank == 1) & in_seg)]:
        n = int(msk.sum())
        if n == 0:
            print(f"  {lab:<14} n=0")
            continue
        ps = pred_sr[msk]; pc = pred_cc[msk]
        known = pc >= 0
        frac_pair_in_graph = float(known.mean())
        frac_sr1 = float((ps == 1)[known].mean()) if known.any() else float("nan")
        better = float((ps > sr[msk]).mean()) if lab.startswith("sr=0") else float("nan")
        head = (f"  {lab:<14} n={n:6d}  预测配对在 r1 图里出现过={frac_pair_in_graph:.4f}"
                f"  预测配对 success_rate==1 占={frac_sr1:.4f}")
        if lab.startswith("sr=0"):
            head += f"  预测配对成色优于真值={better:.4f}"
        print(head)
        rep["C"][lab] = dict(n=n, pair_in_graph=frac_pair_in_graph, pred_sr_eq1=frac_sr1,
                             pred_sr_better=better)

    # ---- 分层 D：真值 vs「同快照历史最优配对」 ----
    # 若两者经常不一致，说明 r1 边本身不是"最优适配"的标注；再看模型在
    # 两者一致 / 不一致的边上分别表现如何（一致段=两种解释同向，不一致段=可区分）。
    sec("分层 D：真值 vs 同快照历史成色最优的 GPU 类型")
    best = build_head_tau_best(algo_unique)
    bl = [best.get((int(a), int(t))) for a, t in zip(heads, tt)]
    has = np.asarray([b is not None for b in bl])
    bn = [b[0] if b is not None else "" for b in bl]
    agree = np.asarray([g is not None and b == g for b, g in zip(bn, gpu_names.tolist())])
    print(f"head×快照在 r1 图里有候选 = {float(has.mean()):.4f}"
          f"  真值 == 成色最优候选 = {float(agree[has].mean()):.4f}（在有声明的边上）")
    for lab, msk in [("真值=成色最优", agree), ("真值≠成色最优", (~agree) & has)]:
        mm = msk & (cc >= 0)
        s = metrics(rank[mm], cls5[mm])
        print(f"  {lab:<16} n={s['n']:6d}  H@1={s['h1']:.6f}  MRR={s['mrr']:.6f} "
              f"macro_occ={s['macro_occ']:.4f}(c={s['c_occ']})")
        print_per(s)
        rep["D_" + lab] = s
    rep["D"] = dict(has_best=float(has.mean()), agree_rate=float(agree[has].mean()),
                    agree_by_sr={f"{lab}": float(agree[msk].mean()) for lab, msk in
                                 [("sr=0", sr == 0), ("0<sr<1", (sr > 0) & (sr < 1)), ("sr=1", sr == 1)]})
    print("  按真值成色分层的 agree 率：" + "  ".join(
        f"{k}={v:.4f}" for k, v in rep["D"]["agree_by_sr"].items()))

    # ---- 分层 E：D 段是不是恒等变换（每个 head×快照到底有几个候选类型） ----
    # 若绝大多数 (head, tau) 在 r1 图里只声明了一个 GPU 类型，那"真值 vs 成色最优"
    # 逐位相同是结构必然而非发现，D 段只用作一致性核对，不作为发现引用。
    sec("分层 E：多候选检查（决定 D 段是不是恒等变换）")
    df = pd.read_parquet(os.path.join(EDGES, "r1_suits_edges.parquet"))
    a2i = {pk: i for i, pk in enumerate(algo_unique)}
    df["_a"] = df["src"].astype(str).map(a2i)
    df = df[df["_a"].notna()]
    df["_a"] = df["_a"].astype(np.int64)
    ncan = df.groupby(["_a", "tau"], sort=False)["dst"].nunique()
    key = pd.MultiIndex.from_arrays([heads, tt], names=["_a", "tau"])
    per_edge = ncan.reindex(key).to_numpy().astype(float)
    rep["E"] = {"keys": int(len(ncan)),
                "can_dist": {str(k): int(v) for k, v in
                             ncan.value_counts().sort_index().items()},
                "frac_multi": float((per_edge > 1).mean()),
                "frac_multi_sr0": float((per_edge[sr == 0] > 1).mean())}
    print(f"(head,tau) 键数={rep['E']['keys']}  每键候选类型数分布={rep['E']['can_dist']}")
    print(f"测试边里 (head,tau) 有 ≥2 个已声明 GPU 类型的占比 = "
          f"{rep['E']['frac_multi']:.6f}（sr=0 段 {rep['E']['frac_multi_sr0']:.6f}）")
    multi = (per_edge > 1) & has
    if multi.sum() >= THIN:
        ag = float(agree[multi].mean())
        s_ag = metrics(rank[multi & agree], cls5[multi & agree])
        s_dg = metrics(rank[multi & ~agree], cls5[multi & ~agree])
        print(f"  多候选子集 n={int(multi.sum())}  agree 率={ag:.4f}")
        print(f"    agree   {fmt(s_ag)}")
        print(f"    disagree{fmt(s_dg) if s_dg else '   (n=0)'}")
        rep["E"]["multi_subset"] = dict(n=int(multi.sum()), agree=ag,
                                        agree_metrics=s_ag, disagree_metrics=s_dg)
        print("  ⇒ D 段在多候选子集上不是恒等变换，可单独报；但只覆盖全体 "
              f"{float(multi.mean()):.4f} 的测试边。")
    else:
        print(f"  ⇒ 多候选测试边只有 {int(multi.sum())} 条（<{THIN}）：D 段的 0.9997"
              f" 是结构必然（每个 head×快照基本只声明一个类型），不作为发现引用。")
        rep["E"]["verdict"] = "D is vacuous: single-candidate by construction"

    # ---- 判线 ----
    sec("预注册判线")
    A = rep["sr_strata"]
    m0, m1 = A.get("success_rate == 0"), A.get("success_rate == 1")
    ok30 = m0 is not None and m0["n"] >= THIN
    ok31 = m1 is not None and m1["n"] >= THIN
    if ok30 and ok31:
        dm = m1["macro_occ"] - m0["macro_occ"]
        dh1 = m1["h1"] - m0["h1"]
        msk0 = (sr == 0); msk1 = (sr == 1)
        lo, hi, sd = boot_diff(rank[msk0], cls5[msk0], rank[msk1], cls5[msk1])
        print(f"L-B1-a  macro_occ(sr=1) - macro_occ(sr=0) = {dm:+.4f}   "
              f"(sd_boot={sd:.4f}, 95%CI [{lo:+.4f}, {hi:+.4f}])")
        print(f"        H@1(sr=1) - H@1(sr=0) = {dh1:+.4f}")
        verdict_a = "换主口径" if abs(dm) >= NOISE_FLOOR else "保留全量口径"
        print(f"        |Δ|{'≥' if abs(dm) >= NOISE_FLOOR else '<'}{NOISE_FLOOR} ⇒ {verdict_a}")
        print(f"L-B1-a(CI)  CI 是否含 0 = {lo <= 0 <= hi}")
        rep["line_lb1a"] = dict(delta_macro=dm, delta_h1=dh1, boot_sd=sd, ci95=[lo, hi],
                                verdict=verdict_a, ci_crosses_zero=bool(lo <= 0 <= hi))
    else:
        print(f"L-B1-a  sr=0 段 n={None if m0 is None else m0['n']} 太薄（<{THIN}），不判线")
        rep["line_lb1a"] = {"skipped": "thin stratum"}

    C = rep["cc_strata"]
    kl, kh = "cooccur == 1", "cooccur >= 50"
    if C.get(kl) and C.get(kl)["n"] >= THIN and C.get(kh) and C[kh]["n"] >= THIN:
        d_h1 = C[kh]["h1"] - C[kl]["h1"]
        d_mc = C[kh]["macro_occ"] - C[kl]["macro_occ"]
        print(f"L-B1-c  H@1(≥50) - H@1(=1) = {d_h1:+.4f} | macro_occ 差 = {d_mc:+.4f}")
        print(f"        |ΔH@1|{'≥' if abs(d_h1) >= NOISE_FLOOR else '<'}{NOISE_FLOOR} ⇒ "
              f"{'“模型只是在背历史配对”有正面证据' if abs(d_h1) >= NOISE_FLOOR else '无正面证据'}")
        print("        单调性：" + "  ".join(f"{lab}:{C[lab]['h1']:.4f}" for lab, _ in b_cc
                                            if C.get(lab) and C[lab]["n"] > 0))
        rep["line_lb1c"] = dict(delta_h1=d_h1, delta_macro=d_mc,
                                verdict="有正面证据" if abs(d_h1) >= NOISE_FLOOR else "无正面证据",
                                curve={lab: (C[lab]["n"], C[lab]["h1"], C[lab]["macro_occ"])
                                       for lab, _ in b_cc if C.get(lab)})
    else:
        print(f"L-B1-c  低/高支持段样本量不足（{kl} n={C.get(kl) and C[kl]['n']}, "
              f"{kh} n={C.get(kh) and C[kh]['n']}），不判线")
        rep["line_lb1c"] = {"skipped": "thin strata"}

    p = os.path.join(OUT, "b1_label_support.json")
    json.dump(rep, open(p, "w"), indent=2, ensure_ascii=False, default=float)
    print(f"\n[写出] {p}")


if __name__ == "__main__":
    main()
