#!/usr/bin/env python3
"""Screen Alibaba cluster-trace-gpu-v2026 as the second dataset for xCAD.

Pre-registered go/no-go criteria (set before looking at the numbers):

  (i)   leak-free cohort  |C(n_hist>=2, span>=2)| >= 500 test edges
  (ii)  pure-memory bound in the recurrent stratum must be *beatable* by a
        temporal model, i.e. the recurrent cohort must not be 100% memorised
        (we record the exact-tail-mode accuracy as the memory bound and the
        static-attribute profile accuracy as the "no-time" competitor)
  (iii) entities carry GPU-type labels (coverage + cardinality)
  (iv)  entity cardinality is trainable on one RTX 3090 next to the v2020 arm

Entity = workload_id (the job_name analogue); link = workload -> GPU type.
Snapshots = the sampled ``day=*/hour=*`` parquet partitions, in time order.
"""
import collections
import glob
import os
import re
import sys

import pyarrow.parquet as pq

COLS = ["pod_id", "workload_id", "server_id", "gpu_spec_public", "job_type_public",
        "model_type_public", "priority_class", "gpu_request", "state_public"]
RAW = sys.argv[1] if len(sys.argv) > 1 else "~/xcad/datasets/v2026/raw"


def key_of(path):
    m = re.search(r"day(\d+)_hour(\d+)", os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 99)


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.parquet")), key=key_of)
    if not files:
        raise SystemExit("no parquet under %s" % RAW)
    snaps = []            # list of (day, hour, rows)  rows = list[(wl, gpu, prof)]
    print("snapshots: %d  %s" % (len(files), [key_of(f) for f in files]))
    print()
    print("== per-snapshot inventory ==========================================")
    print("%-12s %10s %10s %10s %10s %10s" % ("snap", "rows", "gpu_rows", "wl", "pods", "server"))
    for f in files:
        t = pq.read_table(f, columns=COLS).to_pydict()
        n = len(t["workload_id"])
        rows = []
        wl_set, pod_set, srv_set = set(), set(), set()
        n_gpu = 0
        for i in range(n):
            wl = t["workload_id"][i]
            gpu = t["gpu_spec_public"][i]
            if wl is None or gpu is None:
                continue
            g = t["gpu_request"][i]
            if g is None or g <= 0:
                continue
            n_gpu += 1
            wl_set.add(wl); pod_set.add(t["pod_id"][i]); srv_set.add(t["server_id"][i])
            prof = (t["job_type_public"][i], t["model_type_public"][i], t["priority_class"][i])
            rows.append((wl, gpu, prof))
        d, h = key_of(f)
        snaps.append(("d%dh%02d" % (d, h), rows))
        print("%-12s %10d %10d %10d %10d %10d" % ("d%d/h%02d" % (d, h), n, n_gpu,
                                                  len(wl_set), len(pod_set), len(srv_set)))
    print()

    # ---------------------------------------------------------------- (iii)
    gpu_cnt = collections.Counter()
    for _, rows in snaps:
        for _, g, _ in rows:
            gpu_cnt[g] += 1
    tot = sum(gpu_cnt.values())
    print("== (iii) GPU-type labels ===========================================")
    print("distinct gpu_spec_public over gpu_request>0 rows: %d" % len(gpu_cnt))
    for g, c in gpu_cnt.most_common():
        print("   %-12s %10d  %5.2f%%" % (g, c, 100.0 * c / tot))
    print()

    # entity cardinality across the union of snapshots
    all_wl = set()
    for _, rows in snaps:
        all_wl.update(w for w, _, _ in rows)
    print("== (iv) entity cardinality =========================================")
    print("distinct workload_id with gpu edges: %d" % len(all_wl))
    print()

    # -------------------------------------------------------- recurrence
    seen = collections.Counter()
    first_last = {}
    for si, (_, rows) in enumerate(snaps):
        cur = set(w for w, _, _ in rows)
        seen.update(cur)
        for w in cur:
            a, b = first_last.get(w, (si, si))
            first_last[w] = (a, si)
    print("== recurrence across sampled snapshots =============================")
    hist = collections.Counter(seen.values())
    print("snapshot-count histogram for workloads: %s" % sorted(hist.items()))
    print("one-shot (appear in exactly 1 snapshot): %d / %d = %.2f%%"
          % (hist[1], len(all_wl), 100.0 * hist[1] / max(1, len(all_wl))))
    multi = sum(c for k, c in hist.items() if k >= 2)
    print("recur  (>=2 snapshots)                 : %d (%.2f%%)"
          % (multi, 100.0 * multi / max(1, len(all_wl))))
    span_ge2 = sum(1 for w, (a, b) in first_last.items() if b - a >= 1 and seen[w] >= 2)
    print("workloads with span>=2 and n_hist>=2   : %d" % span_ge2)
    print()

    # -------------------------------------------------- (i)/(ii) leak-free cut
    print("== (i)+(ii) leak-free cohort sweep =================================")
    print("%-6s %9s %9s %9s %9s %9s %9s" %
          ("cut", "test_ed", "C(n>=1)", "C(n>=2)", "C(n>=2,span>=2)", "mem_bound", "prof_acc"))
    for cut in range(1, len(snaps)):
        heads_hist = collections.defaultdict(list)     # wl -> [(si, gpu, prof)]
        for si in range(cut):
            for w, g, p in snaps[si][1]:
                heads_hist[w].append((si, g, p))
        test_edges = snaps[cut][1]
        c1 = c2 = c2s = 0
        mem_hit = prof_hit = n_rec = 0
        for w, g, p in test_edges:
            h = heads_hist.get(w)
            if not h:
                continue
            n_hist = len(set(x[0] for x in h))
            span = max(x[0] for x in h) - min(x[0] for x in h) + 1
            if n_hist >= 1:
                c1 += 1
            if n_hist >= 2:
                c2 += 1
                if span >= 2:
                    c2s += 1
                # memory bound: exact tail mode of the history sequence
                tail = collections.Counter(x[1] for x in h[-3:]).most_common(1)
                n_rec += 1
                if tail and tail[0][0] == g:
                    mem_hit += 1
                pc = collections.Counter(x[2] for x in h).most_common(1)
                if pc and pc[0][0] == p:
                    prof_hit += 1
        print("%-6s %9d %9d %9d %9d %9s %9s" %
              (snaps[cut][0], len(test_edges), c1, c2, c2s,
               ("%.4f" % (mem_hit / n_rec)) if n_rec else "-",
               ("%.4f" % (prof_hit / n_rec)) if n_rec else "-"))
    print()

    # drift: does a workload keep its GPU type?
    per_wl = collections.defaultdict(set)
    for _, rows in snaps:
        for w, g, _ in rows:
            per_wl[w].add(g)
    k = collections.Counter(len(v) for v in per_wl.values())
    print("== GPU-type stability per workload =================================")
    print("distinct gpu types seen per workload: %s" % sorted(k.items()))
    rec = [w for w, v in per_wl.items() if seen[w] >= 2]
    drift = sum(1 for w in rec if len(per_wl[w]) >= 2)
    print("among recurring workloads: %d / %d change GPU type (%.2f%%)"
          % (drift, len(rec), 100.0 * drift / max(1, len(rec))))


if __name__ == "__main__":
    main()
