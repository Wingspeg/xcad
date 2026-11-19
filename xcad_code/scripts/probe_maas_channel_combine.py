#!/usr/bin/env python
"""probe_maas_channel_combine.py -- 字段判别力测量（v2 同尺）允许的第二次测量：两通道合并的天花板。

v2 的读数（probe_maas_head_fields_v2.py，12:53 wall / RSS 15.0 GB，0 GPU）：
  全新 head 层 n=102,867  base_hist 0.2980 / tail_prior 0.1651 / 模型 base 0.1233
  复发边层     n=17,525   entity_memory 0.7504 / base_hist 0.4354 / 模型 base 0.1472
  判据落在中间区（增益 +0.1329，G3 线 0.15）=> 按预注册只允许再来一次测量，不许直接开三臂。

这一量的是什么：**一个"两条通道都用对了"的打分函数能到多少**，也就是模型修好之后的天花板。
模型未来的形状就是"复发 head 查自己的历史（entity_memory），全新 head 查同 base 的编队
（base_hist）"，所以把这两张表按权重合成，看每一层能爬到哪：
  mix(w)  : score = (1-w)·归一化(entity_memory) + w·归一化(base_hist)
  switch  : head 在历史出现过就用 entity_memory，否则用 base_hist（模型该学到的门控）
  switchP : switch 之上，空表时退回 tail_prior
只读，不训练，0 GPU；用的尺子和 v2 完全相同（generic_strata._ranks_from_cols，候选 = 30 个卡型）。

顺带把 §17.3 的 fleet（该 head 历史所在集群的最热卡型）在同一把尺子下重算一遍，
并修掉 v2 里那一行覆盖率统计的缺陷（对 pandas Series 用了 set(ser)，取的是值不是键）。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_maas_head_fields_v2 as P                                  # noqa: E402

CAT = P.CAT
NEW = P.NEW


def norm(tab):
    if not tab:
        return {}
    s = float(sum(tab.values()))
    return {int(k): v / s for k, v in tab.items()} if s > 0 else {}


def mix(a, b, w):
    if w <= 0.0:
        return dict(a)
    if w >= 1.0:
        return dict(b)
    out = {k: (1.0 - w) * v for k, v in a.items()}
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + w * v
    return {k: v for k, v in out.items() if v > 1e-12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="~/xcad/datasets/maas/maas_head_edges.parquet")
    ap.add_argument("--data-dir", default="~/xcad/xcad_code/xcad_model/data/maas")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sys.path.insert(0, os.path.expanduser("~/xcad/xcad_code/xcad_model/src"))
    import generic_strata as prod

    c = pd.read_parquet(os.path.expanduser(args.cache))
    P.log(f"[cache] {len(c)} edges")
    dd = os.path.expanduser(args.data_dir)
    ent = {}
    with open(os.path.join(dd, "entity2id.txt")) as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            ent[p[0]] = int(p[1])

    tr = P.load_graph_txt(os.path.join(dd, "train.txt"))
    va = P.load_graph_txt(os.path.join(dd, "valid.txt"))
    te = P.load_graph_txt(os.path.join(dd, "test.txt"))
    ed = pd.concat([tr.assign(split="train"), va.assign(split="valid"),
                    te.assign(split="test")], ignore_index=True)
    gpu_base = int(ed.o.min())
    n_types = int(ed.o.max()) - gpu_base + 1
    ed["tail_idx"] = ed.o - gpu_base
    ed = ed.drop(columns="o")
    c = c.assign(s=c.head_name.map(ent), tail_idx=c.tail_name.map(ent) - gpu_base)
    g_edges = set(map(tuple, ed[["s", "tail_idx", "tau"]].to_numpy(np.int64)))
    c_edges = set(map(tuple, c[["s", "tail_idx", "tau"]].to_numpy(np.int64)))
    assert g_edges == c_edges, "cache 与图文件边集不一致"
    att = c.drop_duplicates(["s", "tau"])[
        ["s", "tau"] + CAT + ["params_b", "gpu_count", "base_id", "clu_modal"]]
    ed = ed.merge(att, on=["s", "tau"], how="left")
    hist = ed[ed.split != "test"].copy().reset_index(drop=True)
    T = ed[ed.split == "test"].copy().reset_index(drop=True)

    rows, base = prod.baseline_ranks(0, P.snaps_from(T), P.snaps_from(hist), n_types,
                                     universe=(0, n_types - 1))
    gold = np.array([int(r[2]) for r in rows], dtype=np.int64)
    n = len(gold)
    assert len(rows) == len(T) and all(
        (int(a[0]), int(a[2])) == (int(b), int(d))
        for a, (b, d) in zip(rows, zip(T.s, T.tail_idx))), "行序与 test.txt 不一致"
    per = {}
    for s, tau, t in zip(T.s.to_numpy(), T.tau.to_numpy(), T.tail_idx.to_numpy()):
        per.setdefault((int(s), int(tau)), set()).add(int(t))
    filt = [{x for x in per[(int(s), int(tau))] if x != int(t)}
            for s, tau, t in zip(T.s.to_numpy(), T.tau.to_numpy(), T.tail_idx.to_numpy())]

    hs_any, hs_rel, hs_days, hs_pairs, hs_cnt, tail_cnt = prod.history_stats(
        P.snaps_from(hist), 0)
    new_mask = ~np.isin(T.s.to_numpy(), list(hs_any))
    recur_edge = np.array([(int(s), int(t)) in hs_pairs
                           for s, t in zip(T.s.to_numpy(), T.tail_idx.to_numpy())])
    cohorts = {"全部 test 边": np.ones(n, bool), NEW: new_mask,
               "head 在 history 复发": np.isin(T.s.to_numpy(), list(hs_rel)),
               "边 (head,tail) 在 history": recur_edge}
    P.log("[cohorts] " + "  ".join(f"{k}={int(v.sum())}" for k, v in cohorts.items()))

    # ---- 三张历史表（键空间都是局部 tail_idx）
    # ---- entity_memory 的键是 head id（整数），不能走 hist_tables —— 它会 stringify 键，
    # 用 int 查就全空（第一版就踩了这个：entity_memory 读数与 uniform 逐位相同）。
    dm = hist.groupby(["s", "tail_idx"]).size()
    mem_cnt = {k: {int(t): int(v) for t, v in sub.droplevel(0).items()}
               for k, sub in dm.groupby(level=0)}
    base_cnt = P.hist_tables(hist, "base_id")
    tc = hist.groupby("tail_idx").size()
    prior = norm({int(t): int(v) for t, v in tc.items()})
    # fleet：head -> 它历史上最常出现的集群 -> 该集群历史最热卡型（§17.3 那条线）
    hc_series = hist.dropna(subset=["clu_modal"]).groupby("s").clu_modal.agg(
        lambda v: v.value_counts().idxmax())
    hc = hc_series.to_dict()
    cl_tab = {}
    for clu, sub in hist.dropna(subset=["clu_modal"]).groupby("clu_modal"):
        cnt = sub.groupby("tail_idx").size()
        cl_tab[clu] = int(cnt.idxmax())
    seen_clu = set(cl_tab)
    fleet_cov = float(P.key_series(T, "clu_modal").isin(seen_clu).mean())
    P.log(f"[fleet] head->集群 覆盖 {float(T.s.isin(set(hc)).mean()):.4f}  "
          f"集群表覆盖(修掉 v2 那行的键/值 bug) {fleet_cov:.4f}  集群数={len(cl_tab)}")

    mem_n = [norm(mem_cnt.get(s, {})) for s in T.s]
    base_n = [norm(base_cnt.get(k, {})) for k in P.key_series(T, "base_id")]
    in_hist = T.s.isin(set(mem_cnt)).to_numpy()

    tabs = {}
    tabs["entity_memory"] = mem_n
    tabs["base_hist"] = base_n
    tabs["tail_prior"] = [prior] * n
    for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        tabs[f"mix(w={w:.1f})"] = [mix(a, b, w) for a, b in zip(mem_n, base_n)]
    tabs["switch(有记忆用记忆, 否则用base)"] = [
        m if in_hist[i] else b for i, (m, b) in enumerate(zip(mem_n, base_n))]
    sw = [m if in_hist[i] else (b or prior) for i, (m, b) in enumerate(zip(mem_n, base_n))]
    tabs["switchP(空表退先验)"] = [s or prior for s in sw]
    tabs["fleet(§17.3 那条, 同尺重算)"] = [
        {cl_tab[hc[s]]: 1.0} if (s in hc and hc[s] in cl_tab) else dict(prior)
        for s in T.s]

    table = {}
    for name, t in tabs.items():
        r = P.ranks_from_tables(prod, t, gold, filt, n_types)
        table[name] = {c: P.metrics(r, m) for c, m in cohorts.items()}
    rep = {"n_test_rows": int(n), "n_card_types": int(n_types),
           "cohorts_n": {k: int(v.sum()) for k, v in cohorts.items()},
           "fleet_cluster_coverage_fixed": round(fleet_cov, 4),
           "head_to_cluster_coverage": round(float(T.s.isin(set(hc)).mean()), 4),
           "table": table}
    cols = list(cohorts)
    P.log("\n===== 两通道合并的天花板（同一把尺子：过滤式排名，候选 = %d 卡型）=====" % n_types)
    P.log(f"{'通道':34s}" + "".join(f"{c:>24s}" for c in cols))
    for name in table:
        P.log(f"{name:34s}" + "".join(
            "%10.4f/%6.3f/%7d" % (table[name][c]["H@1"], table[name][c]["H@3"],
                                  table[name][c]["n"]) for c in cols))
    P.log("  (每格 = H@1 / H@3 / n)")
    best = max((k for k in table if k.startswith(("mix", "switch"))),
               key=lambda k: table[k]["全部 test 边"]["H@1"])
    rep["best_combined_overall"] = best
    rep["best_combined_H1_by_cohort"] = {c: table[best][c]["H@1"] for c in cols}
    P.log(f"\n[合并天花板] 最优合成通道 = {best}  全体 H@1 "
          f"{table[best]['全部 test 边']['H@1']:.4f}  全新 head "
          f"{table[best][NEW]['H@1']:.4f}  复发边 "
          f"{table[best]['边 (head,tail) 在 history']['H@1']:.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
        P.log(f"[out] {args.out}")


if __name__ == "__main__":
    main()
