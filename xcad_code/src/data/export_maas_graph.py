#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_maas_graph.py -- Alibaba cluster-trace-v2026-maas -> RE-GCN disk format.

Why this file exists
--------------------
R1-1's named experiment (2) needs a *constructive* cross-time cohort AND two
separable competition channels.  On v2020 the cohort is 12 edges and the two
channels coincide edge by edge (agreement 0.999747), so no gain there can be
attributed; MIT has a cohort but its memory channel is near-deterministic (83.2%
verbatim repeats) and its card-type variance is 0
(docs/reviews/2026-11-16_second_dataset_screening.md S14, S16).

Head granularity was NOT assumed -- it was decided by a 0-GPU probe
(scripts/probe_maas_freq.py --head={app_id,model_id,base_id}, logs
outputs/reports/probe_maas_head_*.log).  Last-3-snapshot window, lookup accuracy:

    head      n_heads  test edges  cohort>=2d   memory  profile  agree  fleet-modal
    app_id    105,721     55,269       5,953      0.792    0.737   0.784      0.478
    model_id   80,258     51,382       8,008      0.471    0.497   0.509      0.447   <-- used
    base_id     1,076     16,988      15,408      0.169    0.247   0.237      0.385

  * app_id = MIT's failure mode (memory near-deterministic => nothing to attribute);
  * base_id = criterion F2's failure mode (the fleet lookup 0.385 beats BOTH
    head-specific channels 0.17/0.25: at algorithm-family granularity the card
    type is a property of the cluster, not of the algorithm);
  * model_id is the only level where both channels are non-trivial and sit above
    the fleet lookup, which is what criterion (ii) asks for.

What the graph is
-----------------
    head : model_id      (80,257 deployed model variants; b_/s_/x_/o_ = fine-tune flavour)
    rel  : 0  gpu_alloc  (observed model -> card type; single relation)
    tail : gpu_type      (30 anonymised card types; unknown_* rows = 0.0355%)
    tau  : 1-based index over the distinct days present in the sliced shards
           (65 of the trace's 181 days; 38 gaps > 1 day, so one tau step can cross
            a calendar gap -- the GRU sees snapshot order, not calendar time)

Rows whose entity name is BLANK are dropped before the id space is built: "" is a
legal value in model_id (70 dedup edges) and in gpu_type (54), so it would sit in
the head segment and the tail segment at once, shift every id by one and merge two
entities -- the same class of defect as the historical xCAD id collision.  verify_-
maas_graph.py asserts ids == 0..N-1 and head/tail segments disjoint.

Task caveat (also written into graph_report.json): this is an inference-serving
trace.  A (model, day) cell is multi-valued in card type on 54.3% of cells (max
25 types in one day) because a service fans out over a heterogeneous fleet, so
the task is intrinsically multi-label.

Ranking protocol -- same as export_mit_graph.py, which mirrors
src/rrgcn.py::predict:
  * forward triples only;
  * the other answers of the same (head, relation) inside the same test snapshot
    are filtered out (what utils.load_all_answers_for_time_filter does);
  * rank = 1 + #{strictly greater scores}, ties broken at RANDOM by an eps jitter.
Because only the 30 card-type entities can ever receive a positive score, the
full [n_test, n_entities] matrix of the MIT version is replaced by an exact
closed form over the 30 candidates plus the "unscored rest gets a random
position" tail -- identical semantics, ~1000x less memory (n_test is 1e5 and
n_entities 8e4 => 37 GB dense).  The equivalence is checked row by row in
src/data/verify_maas_graph.py, on the SAME jitter draw.

Usage (server):
    ~/xcad/xcad_code/.venv/bin/python xcad_code/src/data/export_maas_graph.py
"""
import argparse
import datetime as dt
import glob
import json
import os

import numpy as np
import pandas as pd

REL_NAME = "gpu_alloc"
REL_ID = 0
READ = ["day", "model_id", "gpu_type", "cluster_id", "site_id"]


# ---------------------------------------------------------------- graph build
def load_edges(shards, report):
    paths = [p for p in sorted(glob.glob(shards)) if "slim" not in os.path.basename(p)]
    assert paths, f"no shards match {shards}"
    raw = pd.concat([pd.read_parquet(p, columns=READ) for p in paths], ignore_index=True)
    report["shards"] = [os.path.basename(p) for p in paths]
    report["raw_rows"] = int(len(raw))
    report["raw_days"] = int(raw.day.nunique())
    report["raw_day_range"] = [int(raw.day.min()), int(raw.day.max())]
    report["unknown_gpu_type_share"] = round(
        float(raw.gpu_type.astype("string").str.startswith("unknown_").mean()), 6)

    # one edge per (model, card type, day): instance segments collapse.
    q = raw[["model_id", "gpu_type", "day"]].dropna().drop_duplicates()
    # Blank names are dropped, never renamed: "" is a legal value in BOTH columns
    # (model_id 6,543 rows / gpu_type 76 rows in the sliced shards), so it lands in
    # the head set and the tail set at once -> one dict key, two ids -> every id
    # after it shifts by one and two distinct entities silently merge.  That is the
    # same class of defect as the old xCAD id collision; see verify_maas_graph.py.
    blank = q[["model_id", "gpu_type"]].apply(
        lambda c: c.astype("string").str.strip().eq(""))
    report["dropped_blank_name_rows"] = int((blank.any(axis=1)).sum())
    report["blank_model_id_rows"] = int(blank.model_id.sum())
    report["blank_gpu_type_rows"] = int(blank.gpu_type.sum())
    q = q[~blank.any(axis=1)]
    days = np.sort(q.day.unique())
    tau_of = {int(d): i + 1 for i, d in enumerate(days)}
    q = q.rename(columns={"model_id": "head_id", "gpu_type": "tail_id"}).copy()
    q["tau"] = q.day.map(tau_of).astype(int)
    q["rel"] = REL_ID
    q = q[["head_id", "tail_id", "day", "rel", "tau"]].drop_duplicates()

    report["edges_after_dedup"] = int(len(q))
    report["snapshots_used"] = int(q.tau.nunique())
    report["heads"] = int(q.head_id.nunique())
    report["tails"] = int(q.tail_id.nunique())
    per = q.groupby("tau").size()
    report["edges_per_snapshot"] = {"min": int(per.min()), "median": int(per.median()),
                                    "max": int(per.max())}
    mt = q.groupby(["head_id", "tau"]).tail_id.nunique()
    report["head_day_multi_type_frac"] = round(float((mt > 1).mean()), 4)
    report["head_day_max_types"] = int(mt.max())
    hd = q.groupby("head_id").tau.nunique()
    report["head_distinct_days"] = {"eq1": round(float((hd == 1).mean()), 4),
                                    "ge2": round(float((hd >= 2).mean()), 4),
                                    "ge3": round(float((hd >= 3).mean()), 4),
                                    "max": int(hd.max())}
    type_drift = q.sort_values("day").groupby("head_id").tail_id.agg(
        lambda s: s.nunique() > 1)
    report["heads_changing_card_type_over_time"] = round(float(type_drift.mean()), 4)
    return q, raw


def time_split(q, test_frac, valid_frac, report):
    """Suffix-window split on tau; boundaries are whole snapshots (no leakage)."""
    per = q.groupby("tau").size().sort_index()
    tot = int(per.sum())

    def suffix_start(counts, frac):
        cum = counts.sort_index(ascending=False).cumsum()
        ok = cum[cum >= frac * tot]
        return int(ok.index.max()) if len(ok) else int(counts.index.min())

    test_lo = suffix_start(per, test_frac)
    rest = per[per.index < test_lo]
    assert len(rest), "test window swallowed the timeline: lower --test-frac"
    valid_lo = suffix_start(rest, valid_frac)
    assert valid_lo < test_lo, "valid window empty: lower --valid-frac"
    q = q.copy()
    q["split"] = np.where(q.tau >= test_lo, "test",
                          np.where(q.tau >= valid_lo, "valid", "train"))
    report["split_cutpoints"] = {"train_tau_max": valid_lo - 1,
                                 "valid": [valid_lo, test_lo - 1],
                                 "test": [test_lo, int(per.index.max())]}
    for s in ("train", "valid", "test"):
        sub = q[q.split == s]
        assert len(sub), f"empty split {s}"
        report[f"{s}_edges"] = int(len(sub))
        report[f"{s}_edge_frac"] = round(len(sub) / tot, 4)
        report[f"{s}_snapshots"] = int(sub.tau.nunique())
    # leakage invariant: a snapshot belongs to exactly one split
    assert q.groupby("tau").split.nunique().max() == 1, "snapshot shared between splits"
    return q


def entity_space(q, report):
    heads = sorted(q.head_id.unique().tolist())
    tails = sorted(q.tail_id.unique().tolist())
    for name in heads + tails:
        assert "\t" not in name and "\n" not in name, name
    ent = {e: i for i, e in enumerate(heads + tails)}
    report["n_head_entities"] = len(heads)
    report["n_gpu_entities"] = len(tails)
    report["n_entities"] = len(ent)
    report["gpu_segment"] = [len(heads), len(heads) + len(tails) - 1]
    assert set(q.tail_id) <= set(tails), "a tail is not in the gpu segment"
    # The head and tail id spaces must be disjoint and contiguous, or entity2id
    # silently merges a model with a card type and every downstream id shifts.
    assert not (set(heads) & set(tails)), f"shared entity name: {sorted(set(heads) & set(tails))[:3]}"
    assert len(ent) == len(heads) + len(tails), "entity name collision in entity2id"
    assert sorted(ent.values()) == list(range(len(ent))), "entity ids are not 0..N-1"
    report["heads_never_in_train"] = int(len(
        set(heads) - set(q[q.split == "train"].head_id)))
    return ent, heads, tails


def write_regcn(out, q, ent, report):
    os.makedirs(out, exist_ok=True)
    w = q.assign(s=q.head_id.map(ent), o=q.tail_id.map(ent))
    assert w[["s", "o"]].notna().all().all(), "edge endpoint missing from entity2id"
    with open(os.path.join(out, "entity2id.txt"), "w") as f:
        for name, i in ent.items():
            f.write(f"{name}\t{i}\n")
    with open(os.path.join(out, "relation2id.txt"), "w") as f:
        f.write(f"{REL_NAME}\t{REL_ID}\n")
    with open(os.path.join(out, "stat.txt"), "w") as f:
        f.write(f"{len(ent)}\t{1}")
    for split, fname in (("train", "train.txt"), ("valid", "valid.txt"),
                        ("test", "test.txt")):
        sub = (w[w.split == split].rename(columns={"rel": "r"})
               .sort_values(["tau", "s"], kind="mergesort"))
        sub[["s", "r", "o", "tau"]].astype(int).to_csv(
            os.path.join(out, fname), sep="\t", header=False, index=False)
        report[f"file_{fname}_rows"] = int(len(sub))


# --------------------------------------------------- closed-form channels
def _load_txt(path):
    d = pd.read_csv(path, sep="\t", header=None, names=["s", "r", "o", "tau"])
    return d[d.r == REL_ID].copy()          # forward rows only


def ranks_from_scores(mat, gold_col, filt_mask, n_ent, rng_seed, jitter=None):
    """mat / filt_mask: [n_rows, n_tails] (mat>0 means "this card type is a
    candidate for this edge").  Returns 1-based filtered ranks in the FULL entity
    space, with ties broken at random, exactly as the dense MIT implementation
    does -- but computed in closed form, because only the 30 card types can ever
    receive a positive score:

      gold scored   -> rank = 1 + #{kept candidates with a strictly greater
                                  jittered score}          (unscored entities are
                                  pure jitter and therefore always below)
      gold unscored -> rank = #{kept candidates} + a uniformly random position
                                  among the jitter-only peers

    `jitter` exists for verify_maas_graph.check_ranks: tie-breaking is random, so
    the dense reference can only be bit-compared when the two implementations draw
    the SAME jitter.  With jitter supplied, candidate rows must agree exactly.
    """
    n = mat.shape[0]
    rng = np.random.default_rng(rng_seed)
    if jitter is None:
        jitter = rng.random(mat.shape) * 1e-6
    # float32 on purpose: filtered_ranks() in export_mit_graph.py adds the jitter in
    # float64 and then casts to float32, which DESTROYS the jitter for large counts
    # (float32 spacing at 8,192 is ~1e-3 >> 1e-6).  Two candidates with the same
    # large count therefore stay tied there and rank optimistically.  Skipping this
    # cast is exactly what verify_maas_graph.check_ranks caught.
    sc = np.where(filt_mask, -np.inf, (mat + jitter).astype(np.float32))
    gsc = sc[np.arange(n), gold_col]
    higher = (sc > gsc[:, None]).sum(1)                        # excludes gold
    kept = (mat > 0) & ~filt_mask
    scored_kept = kept.sum(1)
    a = (mat > 0).sum(1)                                       # all candidates
    f_only = (filt_mask & (mat == 0)).sum(1)                   # filtered non-candidates
    peers = np.maximum(n_ent - a - f_only, 1)                  # jitter-only, incl. gold
    pos_unscored = scored_kept + rng.integers(1, peers + 1)
    is_scored = (mat > 0) & ~filt_mask
    return np.where(is_scored[np.arange(n), gold_col],
                    higher + 1, pos_unscored).astype(np.int64)


def metrics(ranks):
    r = ranks.astype(float)
    return {"n": int(len(r)), "mrr": round(float(np.mean(1 / r)), 6),
            "h1": round(float((r <= 1).mean()), 6),
            "h3": round(float((r <= 3).mean()), 6),
            "h10": round(float((r <= 10).mean()), 6)}


def channel_matrices(train, valid, test, raw, tails, ent, hist_snaps, report):
    """Every channel is scored over the 31 card types from HISTORY ONLY.

    raw (instance level) is needed by the fleet channel because cluster_id is a
    property of the instance segment, not of the collapsed graph edge.
    """
    hist = pd.concat([train, valid])
    col_of_name = {tid: j for j, tid in enumerate(tails)}      # name -> column
    col_of_eid = {ent[tid]: j for tid, j in col_of_name.items()}  # disk id -> column
    n_k = len(tails)
    gcol = np.array([col_of_eid[o] for o in test.o], dtype=np.int64)

    def head_pair_matrix(counts):
        """counts: Series indexed by (head entity, tail entity) -> dense matrix.
        Every candidate of the head is scored, not only the gold one: filling
        only gold would turn a ranking baseline into a lookup hit rate."""
        m = np.zeros((len(test), n_k), dtype=np.float32)
        idx = counts.reset_index()
        rows = {}
        for h, o, c in zip(idx.iloc[:, 0], idx.iloc[:, 1], idx.iloc[:, 2]):
            rows.setdefault(h, []).append((col_of_eid[o], float(c)))
        for k, h in enumerate(test.s):
            for j, c in rows.get(h, ()):
                m[k, j] = c
        return m

    pair_cnt = hist.groupby(["s", "o"]).size()
    tail_cnt = hist.groupby("o").size()
    win = hist[hist.tau > hist.tau.max() - hist_snaps]
    pair_cnt_win = win.groupby(["s", "o"]).size()

    chans = {}
    rng = np.random.default_rng(0)
    chans["uniform"] = rng.random((len(test), n_k), dtype=np.float32)
    tp = np.zeros(n_k, dtype=np.float32)
    for o, c in tail_cnt.items():
        tp[col_of_eid[o]] = c
    chans["tail_prior(history)"] = np.tile(tp, (len(test), 1))
    chans["entity_memory(head,tail)"] = head_pair_matrix(pair_cnt.rename("n"))
    chans[f"memory_last{hist_snaps}_snaps"] = head_pair_matrix(pair_cnt_win.rename("n"))

    # last-history-snapshot memory: what the head used most recently
    last_tau = hist.groupby("s").tau.max().rename("mx")
    ld = hist.merge(last_tau, on="s")
    ld = ld[ld.tau == ld.mx].groupby(["s", "o"]).size()
    chans["memory_last1_snapshot"] = head_pair_matrix(ld.rename("n"))

    # --- F2 as a baseline, not a disqualifier: "the card type of the clusters
    #     this model has already been running on".  History-only, and it uses the
    #     fleet's labels rather than the head's own.
    hraw = raw[raw.day <= report["_last_hist_day"]]
    modetype = hraw.groupby("cluster_id").gpu_type.agg(
        lambda s: s.value_counts().idxmax())
    eid2head = report["_eid2head_name"]
    hr = hraw[["model_id", "cluster_id"]].dropna().drop_duplicates()
    hr = hr.assign(t2=hr.cluster_id.map(modetype)).dropna(subset=["t2"])
    per_head = {}
    for (mid, t2), c in hr.groupby(["model_id", "t2"]).size().items():
        if t2 in col_of_name:
            per_head.setdefault(mid, {})[col_of_name[t2]] = float(c)
    fmap = np.zeros((len(test), n_k), dtype=np.float32)
    cov = 0
    for k, e in enumerate(test.s):
        cs = per_head.get(eid2head.get(e), {})
        for j, c in cs.items():
            fmap[k, j] = max(fmap[k, j], c)
        cov += bool(cs)
    report["fleet_channel_head_coverage"] = round(cov / len(test), 4)
    chans["fleet_prior(head's clusters)"] = fmap

    # --- filter: other answers of the same (head, relation) in the same test snapshot
    filt = np.zeros((len(test), n_k), dtype=bool)
    by_snap = {}
    for k, (s, tau) in enumerate(zip(test.s, test.tau)):
        by_snap.setdefault((s, tau), []).append(k)
    for (_s, _tau), ks in by_snap.items():
        cols = [gcol[k] for k in ks]
        for k in ks:
            filt[k, [c for c in cols if c != gcol[k]]] = True
    return chans, filt, gcol, hist


def strata(hist, test, chans, filt, gcol, n_ent, report):
    seen_days = hist.groupby("s").tau.nunique()
    hset = set(hist.s)
    pair_set = set(map(tuple, hist[["s", "o"]].to_numpy()))
    rec = np.array([(s, o) in pair_set for s, o in zip(test.s, test.o)])
    in_hist = test.s.isin(hset).to_numpy()
    strat = {
        "all": np.ones(len(test), dtype=bool),
        "head_seen_in_history": in_hist,
        "head_recurs_ge2_snaps": test.s.isin(
            set(seen_days[seen_days >= 2].index)).to_numpy(),
        "head_NEVER_in_history": ~in_hist,
        "edge_recurs_in_history": rec,
        "edge_NOT_in_history": ~rec,
    }
    out = {}
    for name, m in chans.items():
        ranks = ranks_from_scores(m, gcol, filt, n_ent, rng_seed=1)
        out[name] = {k: metrics(ranks[v]) for k, v in strat.items() if v.any()}
    report["baseline_strata_sizes"] = {k: int(v.sum()) for k, v in strat.items()}
    return out, rec


def separation_table(hist, test, rec, report):
    """Do the two channels answer the same edge the same way?  If they do, a gain
    cannot be attributed -- v2020's number is 0.999747.  These are LOOKUP
    accuracies on the cohort (0/1 against gold), not filtered ranks; the rank
    versions are in closed_form[edge_recurs_in_history]."""
    modal = hist.groupby("s").o.agg(lambda v: v.value_counts().idxmax())
    last = hist.sort_values("tau").groupby("s").tail(1).set_index("s").o
    pv = test.s.map(modal).to_numpy()
    mv = test.s.map(last).to_numpy()
    gv = test.o.to_numpy()
    both = rec & ~pd.isna(pv) & ~pd.isna(mv)
    out = {
        "cohort_edges_edge_recurs": int(rec.sum()),
        "cohort_edges_both_channels_defined": int(both.sum()),
        "agreement_memory_vs_profile": round(float(np.mean(pv[both] == mv[both])), 6),
        "lookup_memory_H@1_on_cohort": round(float(np.mean(mv[both] == gv[both])), 6),
        "lookup_profile_H@1_on_cohort": round(float(np.mean(pv[both] == gv[both])), 6),
        "only_memory_right": int(np.sum((mv[both] == gv[both]) & (pv[both] != gv[both]))),
        "only_profile_right": int(np.sum((pv[both] == gv[both]) & (mv[both] != gv[both]))),
        "v2020_reference_agreement": 0.999747,
    }
    report["cohort_channel_separation"] = out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=os.path.expanduser(
        "~/xcad/datasets/maas/instance_daily/part_[0-5].parquet"))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--history-snaps", type=int, default=20)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))            # .../xcad_code
    out_dir = args.out_dir or os.path.join(repo, "xcad_model", "data", "maas")

    rep = {"built_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "shards_glob": args.shards, "out_dir": out_dir,
           "relation": REL_NAME, "rel_id": REL_ID,
           "head": "model_id", "tail": "gpu_type",
           "history_snaps": args.history_snaps,
           "test_frac": args.test_frac, "valid_frac": args.valid_frac}
    q, raw = load_edges(args.shards, rep)
    q = time_split(q, args.test_frac, args.valid_frac, rep)
    ent, heads, tails = entity_space(q, rep)
    rep["_eid2head_name"] = {i: h for i, h in enumerate(heads)}
    write_regcn(out_dir, q, ent, rep)
    # last calendar day that is still history (train+valid), for the fleet channel
    hi_valid = rep["split_cutpoints"]["valid"][1]
    rep["_last_hist_day"] = int(q[q.tau <= hi_valid].day.max())

    tr = _load_txt(os.path.join(out_dir, "train.txt"))
    va = _load_txt(os.path.join(out_dir, "valid.txt"))
    te = _load_txt(os.path.join(out_dir, "test.txt"))
    chans, filt, gcol, hist = channel_matrices(tr, va, te, raw, tails, ent,
                                               args.history_snaps, rep)
    cf, rec = strata(hist, te, chans, filt, gcol, rep["n_entities"], rep)
    rep["closed_form"] = cf
    separation_table(hist, te, rec, rep)
    rep["test_head_coverage"] = {"heads_in_test": int(te.s.nunique()),
                                 "test_edges": int(len(te))}

    rep["task_caveat"] = (
        "maas is an inference-serving trace: an edge means 'this model variant was "
        "served on this card type on this day', decided by capacity planning / "
        f"serving orchestration rather than a per-job scheduler.  (model, day) cells "
        f"are multi-valued in card type on {rep['head_day_multi_type_frac']:.1%} of "
        f"cells (max {rep['head_day_max_types']} types in one day), so the task is "
        "multi-label and every number is a filtered rank against the head's other "
        "answers.  gpu_type is an anonymised id, so real hardware specifications "
        "cannot be cited from this dataset.")

    with open(os.path.join(out_dir, "graph_report.json"), "w") as f:
        json.dump({k: v for k, v in rep.items() if not k.startswith("_")},
                  f, indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("closed_form", "_eid2head_name", "_last_hist_day")},
                     indent=1, default=str))
    print("\n" + "=" * 92)
    print(f"CLOSED-FORM CHANNELS  (filtered, forward edges, random tie-break, "
          f"n_entities={rep['n_entities']}, n_tails={len(tails)})")
    print("=" * 92)
    for chan, strata_ in cf.items():
        print(f"\n  {chan}")
        for st, m in strata_.items():
            print(f"    {st:24s} n={m['n']:8d}  MRR={m['mrr']:.6f}  "
                  f"H@1={m['h1']:.4f}  H@3={m['h3']:.4f}  H@10={m['h10']:.4f}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
