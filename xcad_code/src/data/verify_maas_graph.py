#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_maas_graph.py -- invariants of xcad_model/data/maas + a correctness check
of the closed-form ranking shortcut.

Two things are verified, because both were written from scratch:
  A. disk-format invariants: id-space segments, no endpoint outside entity2id,
     (s,r,o,tau) unique, whole-snapshot split boundaries, no snapshot shared.
  B. ranks_from_scores() in export_maas_graph.py is an exact closed form over the
     30 card types; export_mit_graph.py's filtered_ranks() builds the dense
     [n_test, n_entities] matrix.  On a random subsample both are run with the
     SAME seed and the ranks must agree -- otherwise every number in the
     closed-form table is unverified.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.export_maas_graph import (REL_ID, _load_txt, channel_matrices,  # noqa: E402
                                    ranks_from_scores)


def check_format(d, rep):
    ent = {}
    with open(os.path.join(d, "entity2id.txt")) as f:
        for line in f:
            name, i = line.rstrip("\n").split("\t")
            ent[name] = int(i)
    n_ent, n_rel = open(os.path.join(d, "stat.txt")).read().split("\t")
    rep["entity2id_lines"] = len(ent)
    rep["stat_declares"] = [int(n_ent), int(n_rel)]
    assert len(ent) == int(n_ent), "stat.txt disagrees with entity2id.txt"
    assert sorted(ent.values()) == list(range(len(ent))), "ids not 0..N-1"
    assert int(n_rel) == 1, "maas graph is one relation"

    q = pd.read_parquet(os.path.join(os.path.expanduser(
        "~/xcad/datasets/maas/instance_daily/six_shards_slim.parquet")),
        columns=["day", "model_id", "gpu_type"])
    # The exporter drops rows whose entity name is blank ("" is a legal value in
    # BOTH columns -- see export_maas_graph.load_edges), so the trace is compared
    # after the same filter.  Anything else would be a real inconsistency.
    blank = q[["model_id", "gpu_type"]].apply(
        lambda c: c.astype("string").str.strip().eq(""))
    rep["trace_blank_name_rows"] = {
        "model_id": int(blank.model_id.sum()), "gpu_type": int(blank.gpu_type.sum()),
        "either": int(blank.any(axis=1).sum())}
    assert rep["trace_blank_name_rows"]["either"] > 0, \
        "the blank-name filter is dead code: check the exporter"
    q = q[~blank.any(axis=1)]
    tails = set(q.gpu_type.unique())
    heads = set(q.model_id.unique())
    rep["n_tails_in_trace"] = len(tails)
    rep["n_heads_in_trace"] = len(heads)
    assert tails <= set(ent), "a gpu_type is missing from entity2id"
    assert heads <= set(ent), "a model_id is missing from entity2id"
    assert not (heads & tails), "head and tail name spaces overlap"
    assert len(ent) == len(heads) + len(tails), \
        "entity2id does not cover exactly heads + tails"
    gpu_base = len(heads)
    rep["gpu_segment_start"] = gpu_base
    assert all(ent[t] >= gpu_base for t in tails), "a tail sits in the head segment"
    assert all(ent[h] < gpu_base for h in heads), "a head sits in the tail segment"

    splits = {s: _load_txt(os.path.join(d, f"{s}.txt")) for s in
              ("train", "valid", "test")}
    allrows = pd.concat(splits.values())
    rep["rows_train_valid_test"] = {k: int(len(v)) for k, v in splits.items()}
    assert len(allrows) == len(allrows.drop_duplicates(["s", "r", "o", "tau"])), \
        "duplicate (s,r,o,tau)"
    assert set(allrows.r) == {REL_ID}
    for s, t in (("train", "valid"), ("valid", "test")):
        assert splits[s].tau.max() < splits[t].tau.min(), f"{s}/{t} snapshots overlap"
    rep["tau_bounds"] = {k: [int(v.tau.min()), int(v.tau.max())]
                         for k, v in splits.items()}
    # every edge in the trace appears exactly once, in exactly one split
    ded = q.drop_duplicates(["model_id", "gpu_type", "day"])
    assert len(ded) == len(allrows), f"dedup trace {len(ded)} != graph {len(allrows)}"
    rep["edges_match_dedup_trace"] = int(len(allrows))
    return ent, splits, tails


def filtered_ranks_reference(scores, gold, filt_mask, jitter):
    """Dense reference = export_mit_graph.filtered_ranks(), transcribed line for
    line (float32 cast after the jitter, -inf for the filtered set with gold put
    back, rank = 1 + #{strictly greater}).  scores is already [n, n_entities]."""
    n = scores.shape[0]
    sc = (scores + jitter).astype(np.float32)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        g = gold[i]
        s = sc[i].copy()
        f = np.flatnonzero(filt_mask[i])
        if f.size:
            s[f] = -np.inf
            s[g] = sc[i, g]
        out[i] = 1 + int((s > s[g]).sum())
    return out


def check_ranks(d, ent, splits, tails, rep, n_sub=2000):
    """Compare export_maas_graph.ranks_from_scores with a dense [n, n_entities]
    reference built from the same candidate jitter.

    Rows whose gold card type IS a candidate must agree exactly: a positive score
    always beats a jitter-only entity, so both methods put gold right after the
    higher-scored candidates.  Rows whose gold is NOT a candidate can only agree
    in distribution -- the closed form draws one uniform position among the
    jitter-only peers, the reference permutes all of them -- so there we check the
    bounds and the mean."""
    tr, va, te = splits["train"], splits["valid"], splits["test"]
    rng = np.random.default_rng(7)
    sub = te.iloc[rng.choice(len(te), size=min(n_sub, len(te)),
                             replace=False)].reset_index(drop=True)
    raw = pd.read_parquet(os.path.join(os.path.expanduser(
        "~/xcad/datasets/maas/instance_daily/six_shards_slim.parquet")))
    days = np.sort(raw.day.unique())              # tau is the 1-based rank of day
    last_day = int(days[max(tr.tau.max(), va.tau.max()) - 1])
    eid2head = {i: n for i, n in enumerate(sorted(set(ent) - set(tails)))}
    info = {"_last_hist_day": last_day, "_eid2head_name": eid2head}

    chans, filt, gcol, hist = channel_matrices(tr, va, sub, raw, sorted(tails), ent,
                                               20, info)
    n, n_ent = len(sub), len(ent)
    tail_ids = [ent[t] for t in sorted(tails)]
    gold = sub.o.to_numpy()
    # filt / m live in the 30-column candidate BLOCK; the reference has to work in
    # the full entity space (that is what export_mit_graph.filtered_ranks gets from
    # disk), so both the filter mask and the scores are lifted once, here.
    jit = np.random.default_rng(1).random((n, n_ent)) * 1e-6
    filt_ent = np.zeros((n, n_ent), dtype=bool)
    filt_ent[:, tail_ids] = filt
    out = {}
    for name, m in chans.items():
        # ONE jitter draw, shared by both implementations: tie-breaking is random,
        # so bit-equality on candidate rows is only meaningful with identical
        # jitter.  Distinct counts differ by >= 1 >> 1e-6, so what this pins down is
        # the shortcut's bookkeeping (who counts as higher, -inf, the peer range).
        dense = np.zeros((n, n_ent), dtype=np.float32)
        for j, eid in enumerate(tail_ids):
            dense[:, eid] = m[:, j]
        ref = filtered_ranks_reference(dense, gold, filt_ent, jit)
        fast = ranks_from_scores(m, gcol, filt, n_ent, rng_seed=1,
                                 jitter=jit[:, tail_ids])
        is_cand = (m[np.arange(n), gcol] > 0) & ~filt[np.arange(n), gcol]
        exact = int(np.sum(fast[is_cand] == ref[is_cand]))
        peers = n_ent - (m > 0).sum(1)
        lo, hi = ((m > 0).sum(1) + 1, peers + 1)
        inb = int(np.sum((fast[~is_cand] >= lo[~is_cand])
                         & (fast[~is_cand] <= hi[~is_cand])))
        out[name] = {
            "candidate_rows": int(is_cand.sum()), "exact_match": exact,
            "noncandidate_rows": int((~is_cand).sum()), "within_bounds": inb,
            "mean_rank_cand_fast": round(float(np.mean(fast[is_cand])), 3),
            "mean_rank_cand_ref": round(float(np.mean(ref[is_cand])), 3),
            "mean_rank_noncand_fast": round(float(np.mean(fast[~is_cand])), 1)
            if (~is_cand).any() else None,
            "mean_rank_noncand_ref": round(float(np.mean(ref[~is_cand])), 1)
            if (~is_cand).any() else None}
        assert exact == int(is_cand.sum()), \
            f"{name}: closed form disagrees with dense on {is_cand.sum() - exact} " \
            f"candidate rows of {is_cand.sum()}"
        assert inb == int((~is_cand).sum()), f"{name}: non-candidate rank out of bounds"
    rep["rank_check"] = {"subsample": n, "n_entities": n_ent, "per_channel": out}


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/xcad/xcad_code/xcad_model/data/maas")
    rep = {}
    ent, splits, tails = check_format(d, rep)
    check_ranks(d, ent, splits, tails, rep)
    print(json.dumps(rep, indent=1, default=str))
    print("\nALL INVARIANTS OK")


if __name__ == "__main__":
    main()
