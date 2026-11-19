#!/usr/bin/env python
"""probe_maas_head_fields_v2.py -- 全新 head 的"可观测信号"，改用生产那把尺子重量。

v1（probe_maas_head_fields.py，13 分钟跑完）的缺陷：它算的是【直查命中率】
("该 key 的历史最热卡型 == 真值" 的边占比)，而 §17.3/§18.1 那张表算的是【过滤式排名
H@1】(同一快照内该 head 的其他答案先从候选里扣掉，再排名)。两者不是一个口径，差 2 倍多：
v1 的 tail_prior=0.0676，而 §17.3/§18.1 的 tail_prior=0.1651/0.1896。判据 G2/G3 是拿
"相对 tail_prior 的增益"写的，跨口径比出来的结论不能用 => 这一版把每个通道都送进
generic_strata._ranks_from_cols（就是产出 §18.1 那张表的同一个函数），并第一次把模型
自己的读数（seg_*.pt，类型段口径）和所有闭式通道放进同一张表。

对齐方式（比 v1 硬）：行序、gold、过滤集都不自己拼，直接来自磁盘上的
data/maas/{train,valid,test}.txt —— main.py / rrgcn.predict 吃的就是这三份文件；
tail id 平移到 [0, n_types) 的局部空间后用 universe=(0, n_types-1) 打分，与
--ent-segment 给模型的那把尺子只差一个常数平移，排名逐位相同（再用 dump 里的
heads/tails 逐行断言对齐）。parquet 侧只负责"每条边的请求侧属性"，并且逐边断言
它重建出的边集合与图文件完全一致。

判据（与 v1 同一条，只是现在两边同尺）：
  G1 覆盖面：全新 head 的 base_id 在历史出现率 < 0.20        => 属性条件化没有受众；
  G2 无信号：最佳请求侧通道对 tail_prior 的过滤式 H@1 增益 < 0.05 => 换数据，不要调参；
  G3 承重：  增益 >= 0.15                                     => 改打分条件+输出层，值得重训；
  中间区：只允许再来一次 15 分钟级测量，不许直接开三臂。

泄漏纪律不变：允许 = 请求侧字段(workload/model_arch/pd_role/params_b/gpu_count) +
base_id 的【历史】统计；禁止 = gpu_type、gpu_mem_alloc、当日 cluster/site。
当日集群那条只作【参照·含放置结果】打印，不进判定。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

CAT = ["workload", "model_arch", "pd_role"]
NUM = ["params_b", "gpu_count"]
NEW = "head 在 history 从未出现"


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------- 请求侧属性：归约 + 缓存
def modal_frame(df, keys, cols):
    """每个 (head, day) 取该属性的众数（当天实例段里出现最多的取值）。"""
    out = None
    for c in cols:
        m = (df.groupby(keys + [c], dropna=False).size().rename("n").reset_index()
             .sort_values("n", ascending=False).drop_duplicates(keys)[keys + [c]])
        out = m if out is None else out.merge(m, on=keys, how="outer")
    return out


def build_cache(shards_glob, cache_path):
    """每条 (model_id, gpu_type, day) 边 + 该 (model, day) 的 modal/median 属性。"""
    paths = [p for p in sorted(glob.glob(os.path.expanduser(shards_glob)))
             if "slim" not in os.path.basename(p)]
    assert paths, f"no shards under {shards_glob}"
    use = ["day", "model_id", "gpu_type", "cluster_id", "base_id"] + CAT + NUM
    raw = pd.concat([pd.read_parquet(p, columns=use) for p in paths], ignore_index=True)
    log(f"[raw] rows={len(raw)} shards={len(paths)}")

    # 与 export_maas_graph.py::load_edges 同法：去重 + 去空名 + 按天数编号算 tau
    q = raw[["model_id", "gpu_type", "day"]].dropna().drop_duplicates()
    blank = q[["model_id", "gpu_type"]].apply(
        lambda c: c.astype("string").str.strip().eq(""))
    log(f"[raw] blank-name rows dropped={int(blank.any(axis=1).sum())}")
    q = q[~blank.any(axis=1)].rename(
        columns={"model_id": "head_name", "gpu_type": "tail_name"}).copy()
    days = np.sort(q.day.unique())
    q["tau"] = q.day.map({int(d): i + 1 for i, d in enumerate(days)}).astype(int)
    q = q[["head_name", "tail_name", "day", "tau"]].drop_duplicates()

    att = raw.rename(columns={"model_id": "head_name"})[
        ["head_name", "day"] + CAT + NUM + ["base_id", "cluster_id"]]
    modal = modal_frame(att, ["head_name", "day"], CAT + ["base_id", "cluster_id"])
    med = att.groupby(["head_name", "day"])[NUM].median().reset_index()
    att = (modal.merge(med, on=["head_name", "day"], how="outer")
                .rename(columns={"cluster_id": "clu_modal"}))
    out = q.merge(att, on=["head_name", "day"], how="left")
    log(f"[cache] edges={len(out)} heads={out.head_name.nunique()} "
        f"tails={out.tail_name.nunique()} taus={out.tau.nunique()}")
    out.to_parquet(cache_path)
    return out


# ---------------------------------------------------------------- 图文件加载
def load_graph_txt(path):
    """-> DataFrame(s, o, tau)，只留 rel0 正向行，保持文件行序。"""
    d = pd.read_csv(path, sep="\t", header=None, names=["s", "r", "o", "tau"])
    d = d[d.r == 0].copy()
    assert d.tau.is_monotonic_increasing, f"{path}: tau 非递增，split_by_time 假设不成立"
    return d[["s", "o", "tau"]].reset_index(drop=True)


def snaps_from(df):
    """与 utils.split_by_time 同法切快照：列 = [s, rel, tail_idx]（tau 被剥掉）。"""
    a = df[["s", "tau", "tail_idx"]].to_numpy(np.int64)
    out, cur, latest = [], [], None
    for s, tau, t in a:
        if latest is not None and int(tau) != latest:
            if len(cur):
                out.append(np.asarray(cur, dtype=np.int64))
            cur = []
        latest = int(tau)
        cur.append([int(s), 0, int(t)])
    if len(cur):
        out.append(np.asarray(cur, dtype=np.int64))
    return out


# ---------------------------------------------------------------------- 打分
def ranks_from_tables(prod, row_tables, gold, filt, n_types, rng_seed=7):
    """每行一张分数表 -> 过滤式排名，走生产实现 generic_strata._ranks_from_cols。

    相同的表复用同一个 row_key（12 万行绝大多数共享同一张历史直方图）；键用
    frozenset 本身而不是它的 hash，无碰撞风险。universe=(0, n_types-1) 就是类型段尺子。
    """
    uniq, index_of = [], {}
    rid = np.empty(len(row_tables), dtype=np.int64)
    for i, t in enumerate(row_tables):
        k = frozenset(t.items())
        j = index_of.get(k)
        if j is None:
            j = index_of[k] = len(uniq)
            uniq.append(t)
        rid[i] = j
    rng = np.random.default_rng(rng_seed)
    return prod._ranks_from_cols(rid, uniq, np.asarray(gold), filt, n_types, rng,
                                 (0, n_types - 1))


def metrics(rank, mask=None):
    r = np.asarray(rank, dtype=np.float64)
    if mask is not None:
        r = r[mask]
    if not len(r):
        return {}
    return {"n": int(len(r)), "H@1": round(float((r <= 1).mean()), 4),
            "H@3": round(float((r <= 3).mean()), 4),
            "H@10": round(float((r <= 10).mean()), 4),
            "MRR": round(float((1.0 / r).mean()), 4),
            "median_rank": int(np.median(r))}


def direct_lookup(tables, gold):
    """v1 那把旧尺子（直查：表内最大分的列命中真值，并列算对），留着量两口径差多少。"""
    hit = np.zeros(len(gold), dtype=bool)
    for i, t in enumerate(tables):
        if not t:
            continue
        mx = max(t.values())
        if int(gold[i]) in [c for c, v in t.items() if v == mx]:
            hit[i] = True
    return round(float(hit.mean()), 4)


def key_series(df, key):
    """把（可能是组合）键取成可 hash 的 string Series。"""
    if isinstance(key, str):
        return df[key].astype("string")
    return df[key[0]].astype("string") + "|" + df[key[1]].astype("string")


def hist_tables(hist, key):
    """key -> {tail_idx: count}；与 entity_memory 的构造同法（只用历史边）。"""
    cols = [key] if isinstance(key, str) else list(key)
    h = hist[cols + ["tail_idx"]].copy()
    h["_k"] = key_series(h, key)
    h = h[h["_k"].notna()]
    cnt = h.groupby(["_k", "tail_idx"]).size()
    return {k: {int(t): int(c) for t, c in sub.droplevel(0).items()}
            for k, sub in cnt.groupby(level=0)}


def add_channel(hist, T, key, name, rep, tabs, new_mask):
    """历史直方图通道 + 覆盖率（全部 test 边 / 全新 head 层）+ 每键候选卡型数。"""
    tab = hist_tables(hist, key)
    ks = key_series(T, key)
    cov = ks.isin(set(tab))
    widths = [len(tab[k]) for k in tab]
    tabs[name] = [tab.get(k, {}) for k in ks]
    rep.setdefault("coverage", {})[name] = {
        "在全部test边": round(float(cov.mean()), 4),
        "在全新head边": round(float(cov[new_mask].mean()), 4),
        "历史里不同键个数": len(tab),
        "每键卡型数中位": float(np.median(widths)) if widths else None,
        "每键卡型数最大": int(max(widths)) if widths else None}


def logistic_tables(hist, T, cols, bincols):
    from sklearn.linear_model import LogisticRegression

    def design(df):
        parts = [pd.get_dummies(df[c].astype("string"), prefix=c, dummy_na=True)
                 for c in cols]
        parts += [pd.get_dummies(df[c].astype("string"), prefix="b_" + c, dummy_na=True)
                  for c in bincols]
        return pd.concat(parts, axis=1).astype(float)

    Xh = design(hist)
    Xt = design(T).reindex(columns=Xh.columns, fill_value=0.0)
    clf = LogisticRegression(max_iter=1500, C=1.0, solver="lbfgs")
    clf.fit(Xh.to_numpy(), hist["tail_idx"].to_numpy())
    P = clf.predict_proba(Xt.to_numpy())
    log(f"[logit] 特征={Xh.shape[1]} 训练行={Xh.shape[0]} 类={len(clf.classes_)}")
    return [{int(c): float(v) for c, v in zip(clf.classes_, row) if v > 0.0} for row in P]


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="~/xcad/datasets/maas/instance_daily/part_*.parquet")
    ap.add_argument("--cache", default="~/xcad/datasets/maas/maas_head_edges.parquet")
    ap.add_argument("--data-dir", default="~/xcad/xcad_code/xcad_model/data/maas")
    ap.add_argument("--arms-dir", default="~/xcad/xcad_code/xcad_model/src/maas_arms")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sys.path.insert(0, os.path.expanduser("~/xcad/xcad_code/xcad_model/src"))
    import generic_strata as prod
    import torch

    cache = os.path.expanduser(args.cache)
    if os.path.exists(cache):
        c = pd.read_parquet(cache)
        log(f"[cache] loaded {len(c)} edges from {cache}")
    else:
        c = build_cache(args.shards, cache)

    dd = os.path.expanduser(args.data_dir)
    ent = {}
    with open(os.path.join(dd, "entity2id.txt")) as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            assert len(p) == 2, ln
            ent[p[0]] = int(p[1])

    # ---- 图文件：train+valid = history，test = 评测行（与 main.py 的 train_list+valid_list 同集合）
    tr = load_graph_txt(os.path.join(dd, "train.txt"))
    va = load_graph_txt(os.path.join(dd, "valid.txt"))
    te = load_graph_txt(os.path.join(dd, "test.txt"))
    ed = pd.concat([tr.assign(split="train"), va.assign(split="valid"),
                    te.assign(split="test")], ignore_index=True)
    gpu_base = int(ed.o.min())
    n_types = int(ed.o.max()) - gpu_base + 1
    assert ed.o.nunique() == n_types, "tail 段不连续"
    assert int(ed.o.max()) == gpu_base + n_types - 1
    assert (ed.s < gpu_base).all(), "head id 落在 tail 段里"
    ed["tail_idx"] = ed.o - gpu_base
    ed = ed.drop(columns="o")
    log(f"[graph] train={len(tr)} valid={len(va)} test={len(te)} "
        f"heads={ed.s.nunique()} gpu_base={gpu_base} n_types={n_types} "
        f"snapshots={ed.tau.nunique()}")

    # ---- 属性合并到图行；逐边断言 parquet 重建的边集合 == 图文件的边集合
    c = c.assign(s=c.head_name.map(ent), tail_idx=c.tail_name.map(ent) - gpu_base)
    assert not c.s.isna().any() and not c.tail_idx.isna().any(), "实体名不在 entity2id 里"
    g_edges = set(map(tuple, ed[["s", "tail_idx", "tau"]].to_numpy(np.int64)))
    c_edges = set(map(tuple, c[["s", "tail_idx", "tau"]].to_numpy(np.int64)))
    log(f"[align] 图边={len(g_edges)} cache边={len(c_edges)} 对称差={len(g_edges ^ c_edges)}")
    assert g_edges == c_edges, "parquet 重建的边与图文件不是同一批，属性不可用"
    att = c.drop_duplicates(["s", "tau"])[["s", "tau"] + CAT + NUM + ["base_id", "clu_modal"]]
    ed = ed.merge(att, on=["s", "tau"], how="left")
    miss = int(ed.workload.isna().sum())
    log(f"[merge] 边={len(ed)} 属性缺失(workload)={miss} ({miss / max(len(ed), 1):.4f})")

    hist = ed[ed.split != "test"].copy().reset_index(drop=True)
    T = ed[ed.split == "test"].copy().reset_index(drop=True)
    assert len(T) == len(te), f"test 行数 {len(T)} != test.txt rel0 {len(te)}"

    # ---- 行序 / gold / 过滤集：走生产实现，保证与模型 dump 同一批行
    hist_snaps = snaps_from(hist)
    test_snaps = snaps_from(T)
    rows, base = prod.baseline_ranks(0, test_snaps, hist_snaps, n_types,
                                     universe=(0, n_types - 1))
    assert len(rows) == len(T), f"baseline_ranks 行数 {len(rows)} != {len(T)}"
    for i, ((rh, _, ro, _), (ms, mt)) in enumerate(zip(rows, zip(T.s, T.tail_idx))):
        assert (rh, ro) == (int(ms), int(mt)), (i, rh, ro, ms, mt)
    gold = np.array([int(r[2]) for r in rows], dtype=np.int64)
    n = len(gold)

    # 过滤集 = 同一 test 快照内、同 head 的其他答案（与 load_all_answers_for_time_filter 同法）
    per = {}
    for s, tau, t in zip(T.s.to_numpy(), T.tau.to_numpy(), T.tail_idx.to_numpy()):
        per.setdefault((int(s), int(tau)), set()).add(int(t))
    filt = [{x for x in per[(int(s), int(tau))] if x != int(t)}
            for s, tau, t in zip(T.s.to_numpy(), T.tau.to_numpy(), T.tail_idx.to_numpy())]

    hs_any, hs_rel, hs_days, hs_pairs, hs_cnt, tail_cnt = prod.history_stats(hist_snaps, 0)
    new_mask = ~np.isin(T.s.to_numpy(), list(hs_any))
    # hist_snaps 的 tail 列已是局部 id，所以 hs_pairs 也是 (s, tail_idx)
    recur_edge = np.array([(int(s), int(t)) in hs_pairs
                           for s, t in zip(T.s.to_numpy(), T.tail_idx.to_numpy())])
    cohorts = {"全部 test 边": np.ones(n, bool),
               NEW: new_mask,
               "head 在 history 复发": np.isin(T.s.to_numpy(), list(hs_rel)),
               "head 复发 >=2 快照": np.isin(T.s.to_numpy(),
                                            [k for k, v in hs_days.items() if v >= 2]),
               "边 (head,tail) 在 history": recur_edge}
    log("[cohorts] " + "  ".join(f"{k}={int(v.sum())}({v.mean() * 100:.1f}%)"
                                 for k, v in cohorts.items()))

    rep = {"n_test_rows": int(n), "n_card_types": int(n_types), "gpu_base": int(gpu_base),
           "snapshots": {"history": len(hist_snaps), "test": len(test_snaps)},
           "cohorts_n": {k: int(v.sum()) for k, v in cohorts.items()},
           "new_head_share": round(float(new_mask.mean()), 4),
           "ruler": "过滤式排名 generic_strata._ranks_from_cols, 候选=类型段, 与 18/19 节同尺"}

    # ---- 分箱（边界只从历史取；缺失单独成一档）—— 必须在建通道之前
    for col in NUM:
        v = pd.to_numeric(hist[col], errors="coerce")
        bs = np.unique(v.dropna().quantile([0.2, 0.4, 0.6, 0.8]).to_numpy())
        rep.setdefault("bins", {})[col] = [round(float(b), 3) for b in bs]
        rep.setdefault("missing_rate", {})[col] = {
            "history": round(float(v.isna().mean()), 4),
            "test": round(float(pd.to_numeric(T[col], errors="coerce").isna().mean()), 4)}
        for df in (hist, T):
            x = pd.to_numeric(df[col], errors="coerce")
            df[col + "_bin"] = pd.Series(
                pd.cut(x, bins=[-np.inf] + list(bs) + [np.inf], labels=False),
                index=df.index).astype("Int64").astype("string")

    tabs = {}
    # 无信息 / 纯先验 / 记忆（后两条是对照，不进 G2/G3 判定）
    tabs["uniform(全 0 分)"] = [{}] * n
    # 注意：prod.history_stats 返回的 tail_cnt 键是【快照里的 tail 列】= 已平移的局部 id；
    # 这里仍显式从 hist 自己数一遍，避免两处口径悄悄不一致（下面 pipe_check 会对着比）。
    tc = hist.groupby("tail_idx").size()
    tabs["tail_prior(全局最热卡型)"] = [{int(t): int(v) for t, v in tc.items()}] * n
    d = hist.groupby(["s", "tail_idx"]).size()
    mem = {k: {int(t): int(v) for t, v in sub.droplevel(0).items()}
           for k, sub in d.groupby(level=0)}
    tabs["entity_memory(head×卡型历史计数)"] = [mem.get(s, {}) for s in T.s]
    rep["coverage"] = {"entity_memory(head×卡型历史计数)": {
        "在全部test边": round(float(T.s.isin(set(mem)).mean()), 4),
        "在全新head边": 0.0}}
    # 请求侧属性通道：键是属性值而不是 head id，所以能推广到没见过的 head
    for key, name in [("base_id", "base_hist(该 base 历史卡型直方图)"),
                      ("params_b_bin", "params_bin_hist"),
                      ("gpu_count_bin", "gpu_count_hist"),
                      ("workload", "workload_hist"),
                      ("model_arch", "arch_hist"),
                      ("pd_role", "pd_role_hist"),
                      (("base_id", "params_b_bin"), "base_hist × params_bin")]:
        add_channel(hist, T, key, name, rep, tabs, new_mask)
    # 组合逻辑回归：请求侧字段合起来能不能判别
    for cols, bins, name in [(CAT, [], "logit(三个请求类别)"),
                             (CAT, ["params_b_bin", "gpu_count_bin"], "logit(全部请求侧字段)"),
                             (CAT + ["base_id"], ["params_b_bin", "gpu_count_bin"],
                              "logit(请求侧字段 + base_id)")]:
        try:
            tabs[name] = logistic_tables(hist, T, cols, bins)
        except Exception as ex:                                  # noqa: BLE001
            rep.setdefault("errors", {})[name] = repr(ex)
            log(f"[logit] {name} 失败: {ex!r}")

    table = {}
    for name, t in tabs.items():
        r = ranks_from_tables(prod, t, gold, filt, n_types)
        table[name] = {c: metrics(r, m) for c, m in cohorts.items()}
        table[name]["_直查命中率(v1旧尺子)"] = direct_lookup(t, gold)
    # 生产实现自己算的两条闭式基线：校验这份管道没歪（同尺下应逐位相同）
    for k in ("entity_memory", "tail_prior"):
        r = np.asarray(base[k], dtype=np.float64)
        table["prod." + k] = {c: metrics(r, m) for c, m in cohorts.items()}
    rep["prod.lookup_hit_by_cohort"] = {
        c: round(float(base["lookup_hit"][m].mean()), 4) for c, m in cohorts.items()}
    # 管道自检：我这份通道表与生产实现算的同一条通道，逐层 H@1 必须贴在一起
    # （并列抖动 seed 不同，允许 0.01 以内差别；超出说明我拼错了键空间）
    chk = {}
    for mine, theirs in (("tail_prior(全局最热卡型)", "prod.tail_prior"),
                         ("entity_memory(head×卡型历史计数)", "prod.entity_memory")):
        chk[mine] = {c: [table[mine][c]["H@1"], table[theirs][c]["H@1"]] for c in cohorts}
        for c in cohorts:
            d0 = abs(chk[mine][c][0] - chk[mine][c][1])
            assert d0 <= 0.01, f"管道自检失败 {mine} @{c}: {chk[mine][c]} 差 {d0:.4f}"
    rep["pipe_check_H@1_mine_vs_prod"] = chk

    # ---- 模型自己的读数（类型段口径 dump）放同一张表
    for arm in ["seg_base", "seg_hetero", "seg_static"]:
        p = os.path.expanduser(f"{args.arms_dir}/{arm}.pt")
        if not os.path.exists(p):
            log(f"[dump] 缺 {p}")
            continue
        dm = torch.load(p, map_location="cpu")
        r = np.asarray(dm["filt_rank"]).astype(np.float64)
        assert len(r) == n, f"{arm} dump 行数 {len(r)} != {n}"
        assert np.array_equal(np.asarray(dm["heads"]), T.s.to_numpy()), f"{arm} head 行序对不上"
        assert np.array_equal(np.asarray(dm["tails"]) - gpu_base,
                              T.tail_idx.to_numpy()), f"{arm} gold 对不上"
        table["模型 " + arm.replace("seg_", "")] = {
            c: metrics(r, m) for c, m in cohorts.items()}

    # ---- 【参照·含当日放置结果】不进判定
    cl = hist.dropna(subset=["clu_modal"]).groupby("clu_modal").tail_idx.agg(
        lambda s: int(s.mode().iat[0]))
    leak = [{int(cl[k]): 1} if k in cl else {} for k in T.clu_modal]
    r = ranks_from_tables(prod, leak, gold, filt, n_types)
    rep["REFERENCE_当日cluster_含泄漏_不进判定"] = {c: metrics(r, m) for c, m in cohorts.items()}
    rep["REFERENCE_当日cluster_覆盖率"] = round(float(T.clu_modal.isin(set(cl)).mean()), 4)

    rep["channel_table_filtered_ruler"] = table
    cols = list(cohorts)
    log("\n===== 同一把尺子（过滤式排名，候选 = %d 个卡型，与 18/19 节口径一致）=====" % n_types)
    log(f"{'通道':34s}" + "".join(f"{c:>24s}" for c in cols) + f"{'直查(v1)':>12s}")
    for name in table:
        cells = "".join(
            "%10.4f/%6.3f/%7d" % (table[name][c].get("H@1", float("nan")),
                                  table[name][c].get("H@3", float("nan")),
                                  table[name][c].get("n", 0)) for c in cols)
        dl = table[name].get("_直查命中率(v1旧尺子)")
        dl = "%12.4f" % dl if dl is not None else "%12s" % "-"
        log(f"{name:34s}{cells}{dl}")
    log("  (每格 = H@1 / H@3 / n)")

    # ---- 判定
    prior = table["tail_prior(全局最热卡型)"][NEW]["H@1"]
    attr_names = [k for k in table
                  if k.startswith(("logit", "base_hist", "params_bin", "gpu_count_hist",
                                   "workload_hist", "arch_hist", "pd_role_hist"))]
    attr = {k: table[k][NEW]["H@1"] for k in attr_names}
    best = max(attr, key=attr.get)
    gain = attr[best] - prior
    cov = rep["coverage"]["base_hist(该 base 历史卡型直方图)"]["在全新head边"]
    model_new = {k: table[k][NEW]["H@1"] for k in table if k.startswith("模型 ")}
    log(f"\n[G1] 全新 head 的 base_id 在历史出现率 = {cov:.4f}  (线 0.20)")
    log(f"[G2/G3] 最佳请求侧通道 = {best} H@1 {attr[best]:.4f}  vs tail_prior {prior:.4f}"
        f"  增益 {gain:+.4f}  (线 <0.05 无可学判别 / >=0.15 承重)")
    log("[对照] 全新 head 层：模型 H@1 = " + json.dumps(model_new, ensure_ascii=False) +
        f"  uniform={table['uniform(全 0 分)'][NEW]['H@1']:.4f}"
        f"  tail_prior={prior:.4f}"
        f"  entity_memory={table['entity_memory(head×卡型历史计数)'][NEW]['H@1']:.4f}")
    rep["G1_base_coverage_new_head"] = round(float(cov), 4)
    rep["G2G3_best_channel"] = best
    rep["G2G3_best_channel_H1_new_head"] = round(float(attr[best]), 4)
    rep["G2G3_tail_prior_H1_new_head"] = round(float(prior), 4)
    rep["G2G3_gain_vs_tail_prior"] = round(float(gain), 4)
    rep["model_H1_new_head"] = model_new
    rep["attr_channels_H1_new_head"] = {k: round(float(v), 4) for k, v in attr.items()}
    if cov < 0.20:
        v = "G1 不过 => 属性条件化没有受众：放弃 maas 作第二数据集，不再重训"
    elif gain < 0.05:
        v = "G2 命中 => 请求侧字段没有可学判别：换数据，不要调参"
    elif gain >= 0.15:
        v = "G3 => 信号在：把打分条件从 head id 换成请求侧属性 + 类型段输出层，值得重训"
    else:
        v = "中间区(0.05~0.15) => 只允许再来一次 15 分钟级测量，不许直接开三臂"
    rep["verdict"] = v
    log("[verdict] " + v)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
        log(f"[out] {args.out}")


if __name__ == "__main__":
    main()
