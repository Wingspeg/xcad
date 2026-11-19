#!/usr/bin/env python3
"""Alibaba maas (cluster-trace-v2026-maas) screening -- F2/F4 closed-form channels.

Reads the already-sliced instance_daily shard(s) and, for the (model_id -> gpu_type)
edge relation, scores every channel that could explain away a gain:

  class-prior     : most frequent gpu_type in history              (tail popularity)
  repeat-prior    : per model, its most frequent gpu_type in history (algorithm->type profile)
  memory          : per model, its gpu_type on the most recent history day
  cluster / site  : the modal gpu_type of the model's ... n/a at edge level, so we
                    score the *inventory* version: modal type of each cluster_id /
                    site_id, then H@1 against the gold type  (criterion F2 proxy)

Pass criteria were fixed in docs/reviews/2026-11-16_second_dataset_screening.md
section 15.4 before any number was seen.  0 GPU.
"""
import sys

import numpy as np
import pandas as pd

TRAIN_MIN_DAYS = 2


def entropy(s: pd.Series) -> float:
    p = s.value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())


def channels(e: pd.DataFrame, test_days: int) -> None:
    days = np.sort(e.day.unique())
    lo = days[-test_days]
    hist, test = e[e.day < lo], e[e.day >= lo]
    if hist.empty or test.empty:
        print(f"  test=last {test_days}d: empty split, skipped")
        return

    gold = test.gpu_type.to_numpy()
    cls_prior = hist.gpu_type.value_counts().idxmax()
    prof = hist.groupby("model_id").gpu_type.agg(lambda s: s.value_counts().idxmax())
    mem = hist.sort_values("day").groupby("model_id").tail(1).set_index("model_id").gpu_type
    n_h2 = hist.groupby("model_id").day.nunique()
    # F2 decisive baseline: the same prediction made WITHOUT the model id at all --
    # "whatever card this cluster mostly runs".  If this matches memory/prior, the
    # r1 edge is just a projection of model->cluster and there is nothing to learn.
    clus = hist.groupby("cluster_id").gpu_type.agg(lambda s: s.value_counts().idxmax())
    cmod = test.cluster_id.map(clus)

    def rate(pred):
        v = pred.reindex(test.model_id).to_numpy()
        ok = v == gold
        return float(np.nanmean(ok.astype(float))), int(np.sum(ok & ~pd.isna(v)))

    print(f"\n  === test = last {test_days} days ({lo}..{days.max()}), "
          f"n_edges={len(test):,}, n_models={test.model_id.nunique():,}")
    print(f"      class-prior  H@1={np.mean(gold == cls_prior):.4f}")
    r, c = rate(prof)
    print(f"      repeat-prior H@1={r:.4f} (covered {c:,}/{len(test):,})")
    r, c = rate(mem)
    print(f"      memory       H@1={r:.4f} (covered {c:,})")
    ok_c = cmod.to_numpy() == gold
    print(f"      cluster-modal H@1={np.nanmean(ok_c.astype(float)):.4f} "
          f"(covered {int(np.sum(ok_c & ~pd.isna(cmod.to_numpy()))):,}, "
          f"clusters seen in hist={test.cluster_id.isin(clus.index).mean():.3f})")

    ok_p = prof.reindex(test.model_id).to_numpy() == gold
    ok_m = mem.reindex(test.model_id).to_numpy() == gold
    only_p = int(np.nansum(ok_p & ~ok_m))
    only_m = int(np.nansum(ok_m & ~ok_p))
    print(f"      onlyPRIOR={only_p:,}  onlyMEM={only_m:,}  "
          f"both={int(np.nansum(ok_p & ok_m)):,}  neither={int(np.nansum(~ok_p & ~ok_m)):,}")

    hpairs = set(zip(hist.model_id, hist.gpu_type))
    recur = sum(1 for m, g in zip(test.model_id, test.gpu_type) if (m, g) in hpairs)
    rec_mask = np.array([(m, g) in hpairs for m, g in zip(test.model_id, test.gpu_type)])

    # the R1-1 stratum: edges whose (model, type) pair was already seen
    rm, rg = ok_m[rec_mask], gold[rec_mask]
    print(f"      [复发层 n={int(rec_mask.sum()):,}] memory H@1={np.nanmean(rm.astype(float)):.4f}"
          f"  class-prior H@1={np.mean(rg == cls_prior):.4f}"
          f"  repeat-prior H@1={np.nanmean(ok_p[rec_mask].astype(float)):.4f}")
    # do the two channels even give different answers on this layer?
    pv = prof.reindex(test.model_id).to_numpy()
    mv = mem.reindex(test.model_id).to_numpy()
    both_seen = rec_mask & ~pd.isna(pv) & ~pd.isna(mv)
    diff = int(np.sum(both_seen & (pv != mv)))
    o_m = int(np.sum(both_seen & (mv == gold) & (pv != gold)))
    o_p = int(np.sum(both_seen & (pv == gold) & (mv != gold)))
    agree = float(np.mean(pv[both_seen] == mv[both_seen])) if both_seen.any() else float("nan")
    print(f"      [复发层 记忆 vs 画像] 同点率={agree:.6f}  分歧边={diff:,}"
          f"  仅记忆对={o_m:,}  仅画像对={o_p:,}")
    seen = n_h2[n_h2 >= TRAIN_MIN_DAYS].index
    cm = np.isin(test.model_id, list(seen))
    print(f"      [head>=2历史天 n={int(cm.sum()):,}] memory H@1={np.nanmean(ok_m[cm].astype(float)):.4f}"
          f"  class-prior H@1={np.mean(gold[cm] == cls_prior):.4f}"
          f"  repeat-prior H@1={np.nanmean(ok_p[cm].astype(float)):.4f}")
    # --- F2 decisive: on the cohort stratum, does anything beat a plain cluster lookup?
    both = rec_mask & ~pd.isna(mv) & ~pd.isna(cmod.to_numpy())
    if both.any():
        r_m = float(np.mean(ok_m[both]))
        r_c = float(np.mean(ok_c[both]))
        o_mc = int(np.sum(ok_m[both] & ~ok_c[both]))
        o_cm = int(np.sum(ok_c[both] & ~ok_m[both]))
        print(f"      [复发层∩集群可见 n={int(both.sum()):,}] memory H@1={r_m:.4f}"
          f"  cluster-modal H@1={r_c:.4f}  仅记忆对={o_mc:,}  仅集群对={o_cm:,}")
    if cm.any():
        b2 = cm & ~pd.isna(cmod.to_numpy())
        print(f"      [head>=2天∩集群可见 n={int(b2.sum()):,}] memory H@1={np.nanmean(ok_m[b2].astype(float)):.4f}"
          f"  repeat-prior H@1={np.nanmean(ok_p[b2].astype(float)):.4f}"
          f"  cluster-modal H@1={np.mean(ok_c[b2]):.4f}")
    rec_head = int(test.model_id.isin(n_h2[n_h2 >= TRAIN_MIN_DAYS].index).sum())
    print(f"      recurrent edges={recur:,}  edges with head in >=2 history days={rec_head:,}"
          f"   (criterion (i) bar = 500)")

    # F2 proxy: how much of the type label is already implied by the inventory?
    for col in ("cluster_id", "site_id"):
        mode = e.groupby(col).gpu_type.agg(lambda s: s.value_counts().idxmax())
        r2 = float(np.mean(e[col].map(mode).to_numpy() == e.gpu_type.to_numpy()))
        ent = round(1 - sum(len(g) / len(e) * entropy(g.gpu_type)
                            for _, g in e.groupby(col)) / entropy(e.gpu_type), 4)
        ntpc = int(e.groupby(col).gpu_type.nunique().median())
        print(f"      {col}: [in-sample inventory ceiling] modal-type H@1={r2:.4f}  "
              f"explains {ent:.4f} of type entropy"
              f"  median #types per {col}={ntpc}")


def main(paths, head="model_id"):
    for p in paths:
        t = pd.read_parquet(p)
        e = t[[head, "gpu_type", "day", "cluster_id", "site_id"]].drop_duplicates()
        if head != "model_id":            # downstream code speaks one column name
            e = e.rename(columns={head: "model_id"})
        print("=" * 24, p, f"HEAD={head}")
        print(f"  rows={len(t):,}  days={t.day.nunique()} "
              f"[{t.day.min()}..{t.day.max()}]  {head}={t[head].nunique():,} "
              f"models={t.model_id.nunique():,} types={t.gpu_type.nunique()}")
        unk = t.gpu_type.astype("string").str.startswith("unknown_").mean()
        print(f"  F1: unknown_ gpu_type share={unk:.4%}  distinct types={t.gpu_type.nunique()}")
        for k in (3, 5, 7):
            channels(e, k)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--head=")]
    hd = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--head=")),
              "model_id")
    main(args, hd)
