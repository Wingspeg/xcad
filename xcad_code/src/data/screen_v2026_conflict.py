#!/usr/bin/env python3
"""Conflict-cell headroom on Alibaba cluster-trace-gpu-v2026.

R1-1 asks whether xCAD's gain is attributable to the *temporal* evolution of the
relation or to a *static* algorithm->GPU-type prior.  The decisive cohort is the
set of test edges where those two candidate explanations disagree.  On
cluster-trace-gpu-v2020 that cohort has 12 leak-free edges (see
r1_construct.py), so the cell cannot be populated there.  This script measures
how many such edges a dense v2026 window provides.

Signals compared, both estimated from history only (no test leakage):
  MEM  = per-head exact mode of the last K history snapshots (entity memory)
  PROF = per-profile-class mode of GPU type, profile = (job_type, model_type,
         priority_class) (the static attribute prior)
Truth = the GPU type of the head in the test snapshot.

Usage: screen_v2026_conflict.py [parquet_dir] [last]
       "last" evaluates only the final cut (fast check on a long window).
"""
import collections
import glob
import os
import re
import sys

import pyarrow.parquet as pq

COLS = ["workload_id", "server_id", "gpu_spec_public", "job_type_public",
        "model_type_public", "priority_class", "gpu_request"]
DEFAULT = "~/xcad/datasets/v2026/raw"
K_TAIL = 2          # snapshots counted by the MEM predictor


def key_of(path):
    m = re.search(r"day(\d+)_hour(\d+)", os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 99)


def load(files):
    """-> snapshots: list of dict wl -> (Counter(gpu), Counter(prof)) deduped per snapshot."""
    snaps = []
    for f in files:
        t = pq.read_table(f, columns=COLS).to_pydict()
        acc = collections.defaultdict(lambda: (collections.Counter(), collections.Counter()))
        n = len(t["workload_id"])
        for i in range(n):
            wl, gpu = t["workload_id"][i], t["gpu_spec_public"][i]
            g = t["gpu_request"][i]
            if wl is None or gpu is None or not g:
                continue
            prof = (t["job_type_public"][i], t["model_type_public"][i], t["priority_class"][i])
            cg, cp = acc[wl]
            cg[gpu] += 1
            cp[prof] += 1
        snaps.append(acc)
    return snaps


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    only_last = len(sys.argv) > 2 and sys.argv[2] == "last"
    files = [f for f in sorted(glob.glob(os.path.join(d, "*.parquet")), key=key_of)]
    print("window: %d snapshots  %s .. %s" % (len(files), key_of(files[0]), key_of(files[-1])))
    snaps = load(files)
    print("edges/snapshot: %s" % [len(s) for s in snaps])
    print()

    print("%-4s %8s %8s %8s %8s | %7s %7s %7s | %8s %8s %8s %8s" %
          ("cut", "test_e", "rec>=2", "span>=2", "rec%", "MEM", "PROF", "both",
           "M!P", "P!M", "M!P+w", "P!M+w"))
    tot = collections.Counter()
    for cut in ([len(snaps) - 1] if only_last else range(1, len(snaps))):
        hist = collections.defaultdict(list)           # wl -> [(si, gpu, prof)]
        for si in range(cut):
            for wl, (cg, cp) in snaps[si].items():
                hist[wl].append((si, cg.most_common(1)[0][0], cp.most_common(1)[0][0]))
        prof_cls = collections.defaultdict(collections.Counter)   # profile -> gpu counter
        prof_seen = collections.Counter()                          # profile -> snapshots seen
        for wl, h in hist.items():
            for (si, gpu, prof) in h:
                prof_cls[prof][gpu] += 1
            prof_seen[h[-1][2]] += 1
        test_e = rec2 = span2 = 0
        mem_hit = prof_hit = both_hit = 0
        m_not_p = p_not_m = m_not_p_win = p_not_m_win = 0
        for wl, (cg, cp) in snaps[cut].items():
            test_e += 1
            h = hist.get(wl)
            if not h or len(h) < 2:
                continue
            rec2 += 1
            if h[-1][0] - h[0][0] >= 1 and len(h) >= 2:
                span2 += 1
            truth = cg.most_common(1)[0][0]
            mem = [g for (_, g, _) in h][-K_TAIL:]
            mem_pred = collections.Counter(mem).most_common(1)[0][0]
            prof = cp.most_common(1)[0][0]
            prof_pred = prof_cls[prof].most_common(1)[0][0] if prof_cls.get(prof) else None
            m, p = mem_pred == truth, prof_pred == truth
            mem_hit += m
            prof_hit += p
            both_hit += (m and p)
            if mem_pred != prof_pred:
                m_not_p += 1
                if m and not p:
                    m_not_p_win += 1
            if prof_pred is not None and mem_pred != prof_pred:
                p_not_m += 1
                if p and not m:
                    p_not_m_win += 1
        n = max(1, span2)
        print("%-4d %8d %8d %8d %7.1f | %6.4f %6.4f %6.4f | %8d %8d %8d %8d" %
              (cut, test_e, rec2, span2, 100.0 * rec2 / max(1, test_e),
               mem_hit / n, prof_hit / n, both_hit / n,
               m_not_p, p_not_m, m_not_p_win, p_not_m_win))
        for k, v in [("test", test_e), ("rec2", rec2), ("mem", mem_hit), ("prof", prof_hit),
                     ("cell_mp", m_not_p), ("cell_pm", p_not_m),
                     ("cell_mp_win", m_not_p_win), ("cell_pm_win", p_not_m_win)]:
            tot[k] += v
    print()
    print("window totals: recurrent edges=%d  MEM right=%d (%.4f)  PROF right=%d (%.4f)" %
          (tot["rec2"], tot["mem"], tot["mem"] / max(1, tot["rec2"]),
           tot["prof"], tot["prof"] / max(1, tot["rec2"])))
    print("disagreement cells: MEM!=PROF on %d edges (MEM correct %d, PROF correct %d)" %
          (tot["cell_mp"], tot["cell_mp_win"], tot["cell_pm_win"]))
    print()

    # how much genuine temporal migration exists at all?
    mig = 0
    mig_wl = set()
    allw = set()
    for s in snaps:
        allw.update(s)
    seq = collections.defaultdict(list)
    for si, s in enumerate(snaps):
        for wl, (cg, _) in s.items():
            g = cg.most_common(1)[0][0]
            if not seq[wl] or seq[wl][-1] != g:
                seq[wl].append(g)
    for wl, g in seq.items():
        if len(g) >= 2:
            mig += 1
            mig_wl.add(wl)
    print("heads with >=2 distinct GPU types over the window (true migration): %d / %d = %.2f%%"
          % (mig, len(allw), 100.0 * mig / max(1, len(allw))))
    edge_mig = sum(len(snaps[si]) for si in range(1, len(snaps))
                   for wl in snaps[si] if wl in mig_wl)
    print("test edges belonging to those heads: %d" % edge_mig)
    dist = collections.Counter(len(g) for g in seq.values())
    print("distinct-GPU-type count histogram: %s" % sorted(dist.items()))


if __name__ == "__main__":
    main()
