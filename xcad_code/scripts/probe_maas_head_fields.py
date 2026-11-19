#!/usr/bin/env python
"""probe_maas_head_fields.py -- 全新 head 上到底有没有"可观测信号"（0 GPU，纯 CPU）。

背景（screening §18/§19）：maas 测试边里 84.8%（102,867 条）的 head（model_id）在历史里
从未出现。§19 已排除"评测口径不公平"：候选从 80,287 缩到 30 个卡型，三臂读数逐位不动，
模型 0.1472 < 恒定答最热卡型 0.1896。剩下的解释是结构性的：打分是 head embedding · tail
embedding，全新 head 的 embedding 是初始化噪声 ⇒ 分数对 h 无依赖，期望上退化成 tail 边际
分布。调参救不了这个。

本脚本量一件事：**把"head 的身份"换成"head 的可观测上下文"之后，同一层还能不能学到东西。**
建图只读了 5 列（day/model_id/gpu_type/cluster_id/site_id），原始 trace 还有 base_id（这个
变体基于哪个基座模型）、app_id、workload、model_arch、pd_role、gpu_count、params_b、
gpu_mem_alloc。若这些字段在全新 head 层显著越过 tail_prior，"修模型"就有靶子；越不过，
说明 maas 对全新 head 结构上没给可学信号 ⇒ 换数据，别再调参。

判据（跑之前写死）：
  G1 覆盖面：全新 head 中 base_id 在历史里出现过的比例 < 0.20  => 属性条件化没有受众，放弃 maas；
  G2 无信号：最佳"请求侧"通道 H@1 相对 tail_prior 增益 < 0.05  => 结论同上（换数据）；
  G3 承重：  增益 >= 0.15                                       => 信号在，改打分条件+输出层。
  0.05 <= 增益 < 0.15：中间区，只允许再花一次 15 分钟级的验证，不许直接开三臂。

泄漏纪律（比读数重要）：
  允许 = 决策时刻请求自带：workload / model_arch / pd_role / params_b / gpu_count，
         以及 base_id / app_id 的**历史**统计。
  禁止 = gpu_type、gpu_mem_alloc（放置结果）、当日 cluster_id/site_id（同一批放置答案的另一半）。
         集群相关只保留"历史 cluster→卡型 × 该 head 的历史集群"这一条（= §17 的 fleet 通道），
         另把"历史 cluster→卡型 × 该 head **当天**集群"标成【参照·含当日放置结果】，不进判定。

切点与 §17 建成图逐条同构（(model_id,gpu_type,day) 去重、丢空名、按快照边界后缀切
test 0.2 / valid 0.1），开头拿 graph_report.json 对账，对不上直接退出。
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

CAT = ["workload", "model_arch", "pd_role"]
NUM = ["params_b", "gpu_count"]
KEYS = ["head_id", "day"]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------- 建图复现 + 对账
def load_raw(shards_glob, rep):
    paths = [p for p in sorted(glob.glob(os.path.expanduser(shards_glob)))
             if "slim" not in os.path.basename(p)]
    assert paths, f"no shards under {shards_glob}"
    use = ["day", "model_id", "gpu_type", "cluster_id", "site_id", "app_id",
           "base_id"] + CAT + NUM
    raw = pd.concat([pd.read_parquet(p, columns=use) for p in paths], ignore_index=True)
    rep["shards"] = [os.path.basename(p) for p in paths]
    rep["raw_rows"] = int(len(raw))
    rep["field_missing_share"] = {c: round(float(raw[c].isna().mean()), 6)
                                  for c in use if c != "day"}
    return raw


def build_edges(raw, rep):
    q = raw[["model_id", "gpu_type", "day"]].dropna().drop_duplicates()
    blank = q[["model_id", "gpu_type"]].apply(
        lambda c: c.astype("string").str.strip().eq(""))
    q = q[~blank.any(axis=1)].rename(
        columns={"model_id": "head_id", "gpu_type": "tail_id"}).copy()
    days = np.sort(q.day.unique())
    q["tau"] = q.day.map({int(d): i + 1 for i, d in enumerate(days)}).astype(int)
    q = q[KEYS + ["tail_id", "tau"]].drop_duplicates()

    per = q.groupby("tau").size().sort_index()
    tot = int(per.sum())

    def suffix_start(counts, frac):
        cum = counts.sort_index(ascending=False).cumsum()
        ok = cum[cum >= frac * tot]
        return int(ok.index.max()) if len(ok) else int(counts.index.min())

    test_lo = suffix_start(per, 0.2)
    valid_lo = suffix_start(per[per.index < test_lo], 0.1)
    q["split"] = np.where(q.tau >= test_lo, "test",
                          np.where(q.tau >= valid_lo, "valid", "train"))
    rep.update(edges_after_dedup=int(len(q)),
               split_cutpoints={"train_tau_max": valid_lo - 1,
                                "valid": [valid_lo, test_lo - 1],
                                "test": [test_lo, int(per.index.max())]},
               train_edges=int((q.split == "train").sum()),
               valid_edges=int((q.split == "valid").sum()),
               test_edges=int((q.split == "test").sum()),
               heads=int(q.head_id.nunique()), tails=int(q.tail_id.nunique()))
    return q


def reconcile(rep, path):
    if not os.path.exists(path):
        rep["reconcile"] = "SKIPPED: graph_report.json 不存在"
        return
    g = json.load(open(path))
    diff = {k: (g.get(k), rep.get(k)) for k in
            ("edges_after_dedup", "train_edges", "valid_edges", "test_edges",
             "heads", "tails") if g.get(k) != rep.get(k)}
    rep["reconcile"] = "OK" if not diff else diff
    assert not diff, f"与建成图对不上，读数作废: {diff}"


# ------------------------------------------------------------------ 属性归约
def modal_frame(df, cols):
    """每 (head_id, day) 取该列众数（向量化，不用 lambda 遍历分组）。"""
    out = None
    for c in cols:
        m = (df.groupby(KEYS + [c], dropna=False).size().rename("n")
               .reset_index().sort_values("n", ascending=False)
               .drop_duplicates(KEYS)[KEYS + [c]])
        out = m if out is None else out.merge(m, on=KEYS, how="outer")
    return out


def head_constancy(raw, cols):
    r = {}
    g = raw.groupby("model_id")
    for c in cols:
        nu = g[c].nunique()
        r[c] = {"models": int(len(nu)), "single_value_share": round(float((nu == 1).mean()), 4),
                "median_values": int(nu.median()), "max_values": int(nu.max())}
    return r


# -------------------------------------------------------------------- 通道
def h1_by_lookup(hist, test_df, key):
    """hist 里该 key 的众数卡型 -> test_df 的 top-1 命中（覆盖不到的算错）。"""
    cnt = hist.groupby([key, "tail_id"]).size().rename("n").reset_index()
    best = cnt.sort_values("n", ascending=False).drop_duplicates(key).set_index(key).tail_id
    nuniq = cnt.groupby(key).tail_id.nunique()
    have = test_df[key].isin(best.index)
    pred = test_df[key].map(best)
    return {
        "n_test": int(len(test_df)),
        "keys_in_hist": int(len(best)),
        "mean_types_per_key": round(float(nuniq.mean()), 2) if len(nuniq) else None,
        "coverage": round(float(have.mean()), 4),
        "h1_all": round(float((pred == test_df.tail_id)[have].sum() / max(len(test_df), 1)), 4),
        "h1_on_covered": round(float((pred[have] == test_df.tail_id[have]).mean()), 4)
        if have.any() else 0.0,
    }


def onehot(df, cols, ref_cols):
    X = pd.concat([pd.get_dummies(df[c].astype("string"), prefix=c, dummy_na=True)
                   for c in cols], axis=1).astype(float)
    return X.reindex(columns=ref_cols, fill_value=0.0)


def logit_channel(hist, test_df, cols, tails, t2i, rep):
    from sklearn.linear_model import LogisticRegression
    Xh_full = pd.concat([pd.get_dummies(hist[c].astype("string"), prefix=c, dummy_na=True)
                         for c in cols], axis=1).astype(float)
    Xt = onehot(test_df, cols, Xh_full.columns)
    yh = hist.tail_id.map(t2i).to_numpy()
    if len(np.unique(yh)) < 2:
        return {"skipped": "单类"}
    clf = LogisticRegression(max_iter=1200, C=1.0, solver="lbfgs")
    clf.fit(Xh_full.values, yh)
    inv = np.array(tails)
    pred = inv[clf.predict(Xt.values)]
    return {"n_features": int(Xh_full.shape[1]),
            "h1_all": round(float((pred == test_df.tail_id.to_numpy()).mean()), 4)}


def cond_entropy_bits(df, col):
    d = df[[col, "tail_id"]].dropna()
    if not len(d) or d.tail_id.nunique() < 2:
        return None
    cnt = d.groupby([col, "tail_id"]).size().rename("n").reset_index()
    p = cnt.n / cnt.groupby(col).n.transform("sum")
    h = -(p * np.log2(p)).groupby(cnt[col]).sum()
    w = cnt.groupby(col).n.sum() / len(d)
    return round(float((h * w).sum()), 4)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="~/xcad/datasets/maas/instance_daily/part_*.parquet")
    ap.add_argument("--graph-report", default="~/xcad/xcad_model/data/maas/graph_report.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rep = {}
    raw = load_raw(args.shards, rep)
    q = build_edges(raw, rep)
    reconcile(rep, os.path.expanduser(args.graph_report))
    rep["id_level_constancy_per_model"] = head_constancy(raw, CAT + NUM + ["base_id", "app_id"])

    att_keys = CAT + NUM + ["base_id", "app_id", "cluster_id"]
    att = raw.rename(columns={"model_id": "head_id"})[KEYS + att_keys]
    modal = modal_frame(att, CAT + ["base_id", "app_id", "cluster_id"])
    med = att.groupby(KEYS)[NUM].median().reset_index()
    att = modal.merge(med, on=KEYS, how="outer").rename(columns={"cluster_id": "clu_modal"})
    q = q.merge(att, on=KEYS, how="left")

    hist = q[q.split != "test"].copy()
    test = q[q.split == "test"].copy()
    del raw, att, modal, med

    tails = sorted(q.tail_id.unique())
    t2i = {t: i for i, t in enumerate(tails)}
    rep["n_card_types"] = len(tails)

    hist_heads = set(hist.head_id.unique())
    test["is_new_head"] = ~test.head_id.isin(hist_heads)
    nw = test[test.is_new_head].copy()
    rc = test[~test.is_new_head].copy()
    rep["test_new_head"] = {"edges": int(len(nw)), "share": round(float(len(nw) / len(test)), 4),
                            "heads": int(nw.head_id.nunique()),
                            "recurrent_edges": int(len(rc))}

    # 分箱（边界取自历史分位，避免用 test 日信息）
    for c in NUM:
        qs = np.unique(np.nanquantile(hist[c].astype(float), np.linspace(0, 1, 9)))
        for df in (hist, nw, rc):
            df[c + "_bin"] = pd.cut(df[c].astype(float), bins=qs, include_lowest=True,
                                    labels=False).astype("Int64").astype("string")
    rep["param_bins"] = {"edges": [round(float(x), 3) for x in qs], "n_bins": int(len(qs) - 1)}

    hist_base, hist_app = set(hist.base_id.dropna()), set(hist.app_id.dropna())
    rep["new_head_field_presence"] = {
        "base_id_seen_in_history": round(float(nw.base_id.isin(hist_base).mean()), 4),
        "app_id_seen_in_history": round(float(nw.app_id.isin(hist_app).mean()), 4),
        "params_b_present": round(float(nw.params_b.notna().mean()), 4),
        "gpu_count_present": round(float(nw.gpu_count.notna().mean()), 4),
        "distinct_params_b_values_on_new_heads": int(nw.params_b.nunique()),
        "distinct_base_id_on_new_heads": int(nw.base_id.nunique()),
    }
    tail_mode = hist.tail_id.value_counts().idxmax()
    rep["channels_new_head_layer"] = {
        "uniform": {"h1_all": round(1.0 / len(tails), 4)},
        "tail_prior(全局最热卡型)": {"h1_all": round(float((nw.tail_id == tail_mode).mean()), 4),
                                    "coverage": 1.0},
    }
    for key, name in [("base_id", "base_modal(该 base 历史最热卡型)"),
                      ("app_id", "app_modal(该 app 历史最热卡型)"),
                      ("params_b_bin", "params_bin_modal"),
                      ("gpu_count_bin", "gpu_count_bin_modal"),
                      ("workload", "workload_modal"), ("model_arch", "model_arch_modal"),
                      ("pd_role", "pd_role_modal")]:
        rep["channels_new_head_layer"][name] = h1_by_lookup(hist, nw, key)
    rep["channels_new_head_layer"]["joint(base_id x params_bin) 查表"] = h1_by_lookup(
        hist.assign(joint=hist.base_id.astype("string") + "|" + hist.params_b_bin.astype("string")),
        nw.assign(joint=nw.base_id.astype("string") + "|" + nw.params_b_bin.astype("string")),
        "joint")
    for cols, name in [(["params_b_bin"], "logit(params_b 分箱)"),
                       (["gpu_count_bin"], "logit(gpu_count 分箱)"),
                       (CAT, "logit(workload+arch+pd_role)"),
                       (CAT + ["params_b_bin", "gpu_count_bin"], "logit(全部请求侧字段)")]:
        rep["channels_new_head_layer"][name] = logit_channel(hist, nw, cols, tails, t2i, rep)
    # base_id 不进 one-hot（历史里几万个键会撑出 460k x 50k 稠密阵）；用查表/联合查表代表它。

    rep["new_head_cond_entropy_bits"] = {
        "H(gpu_type)": round(float(-sum((v / len(nw)) * np.log2(v / len(nw))
                                        for v in nw.tail_id.value_counts())), 4),
        **{f"H(gpu_type | {c})": cond_entropy_bits(nw, c)
           for c in CAT + ["params_b_bin", "gpu_count_bin", "base_id", "app_id"]}}

    # 复发层对照：属性通道会不会把记忆的活抢走（抢得动就说明它是承重特征）
    rep["channels_recurrent_layer"] = {
        "tail_prior": round(float((rc.tail_id == tail_mode).mean()), 4),
        "head_modal(该 head 历史最热卡型)": h1_by_lookup(hist, rc, "head_id"),
        "base_modal": h1_by_lookup(hist, rc, "base_id"),
        "logit(全部请求侧字段)": logit_channel(hist, rc, CAT + ["params_b_bin", "gpu_count_bin"],
                                              tails, t2i, rep),
        "logit(全部请求侧字段+base_id)": logit_channel(
            hist, rc, CAT + ["params_b_bin", "gpu_count_bin", "base_id"], tails, t2i, rep),
        "entity_memory 见 §18.1": 0.7515, "fleet 见 §17.3": 0.3473,
    }

    # 【参照·含当日放置结果】不要用：当天集群是同一批放置答案的另一半，只交代量级
    cl_map = hist.groupby("clu_modal").tail_id.agg(
        lambda s: s.mode().iat[0] if len(s.mode()) else None)
    rep["REFERENCE_with_same_day_placement_LEAKAGE"] = {
        "note": "该 head 当天的集群(取自 test 日实例记录, 属放置结果) x 历史 cluster->卡型众数；"
                "不进任何判定",
        "h1_all": round(float((nw.clu_modal.map(cl_map) == nw.tail_id).fillna(False).mean()), 4),
        "coverage": round(float(nw.clu_modal.isin(cl_map.index).mean()), 4)}

    log(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2, default=str)

    # ------------------------------------------------------------------ 判定
    cov = rep["new_head_field_presence"]["base_id_seen_in_history"]
    attr_ch = {k: v["h1_all"] for k, v in rep["channels_new_head_layer"].items()
               if "h1_all" in v and not k.startswith(("uniform", "tail_prior"))}
    best_name = max(attr_ch, key=attr_ch.get)
    best, prior = attr_ch[best_name], rep["channels_new_head_layer"]["tail_prior(全局最热卡型)"]["h1_all"]
    gain = best - prior
    log(f"\n[G1] 全新 head 的 base_id 在历史出现率 = {cov:.4f}   (线 0.20)")
    log(f"[G2/G3] 最佳请求侧通道 = {best_name} H@1 {best:.4f}  vs tail_prior {prior:.4f}  "
        f"增益 {gain:+.4f}   (线: <0.05 无可学判别 / >=0.15 承重)")
    if cov < 0.20:
        verdict = "G1 不过 => 属性条件化没有受众：放弃 maas 作为第二数据集，不再重训"
    elif gain < 0.05:
        verdict = "G2 命中 => 请求侧字段没有可学判别：换数据，不要调参"
    elif gain >= 0.15:
        verdict = "G3 => 信号在：把打分条件从 head id 换成请求侧属性 + 类型段输出层，值得重训"
    else:
        verdict = "中间区(0.05~0.15) => 只允许再花一次 15 分钟级验证，不许直接开三臂"
    log("[verdict] " + verdict)


if __name__ == "__main__":
    main()
