#!/usr/bin/env python3
"""How far can a *purely static* predictor go on v2026, versus entity memory?

R1-1's alternative explanation is "the gain comes from the static
algorithm->GPU-type profile, not from history".  The falsifiable version on a
second dataset is: fit every static channel the trace offers on history only,
and see whether it can reach the entity-memory bound on the leak-free recurrent
cohort.  Channels compared on the same cut:

  MEM       per-head mode of its own history            (entity memory, upper bound)
  PROF      mode of the gpu type inside the head's categorical profile class
  LR-attr   multinomial logistic on resource requests + qos/job/model (+/- cluster)
  LR-attr+  LR-attr plus the head's own numeric history aggregate (mean gpu_request
            over its history) - still static in time, uses entity identity only
            through its attributes

Usage: screen_v2026_prior.py [parquet_dir] [cuts...]
"""
import collections
import glob
import os
import re
import sys

import numpy as np
import pyarrow.parquet as pq

COLS = ["workload_id", "gpu_spec_public", "job_type_public", "model_type_public",
        "priority_class", "cluster_id", "gpu_request", "gpu_mem_request",
        "cpu_request_cores", "server_gpu_count"]
NUM = ["gpu_request", "gpu_mem_request", "cpu_request_cores", "server_gpu_count"]


def key_of(path):
    m = re.search(r"day(\d+)_hour(\d+)", os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 99)


def head_rows(path):
    """One record per (head, gpu_type) present in the snapshot, attributes averaged."""
    t = pq.read_table(path, columns=COLS).to_pydict()
    acc = {}
    n = len(t["workload_id"])
    for i in range(n):
        wl, gpu = t["workload_id"][i], t["gpu_spec_public"][i]
        g = t["gpu_request"][i]
        if wl is None or gpu is None or not g:
            continue
        k = (wl, gpu)
        r = acc.get(k)
        feats = [t[c][i] for c in NUM]
        if r is None:
            acc[k] = [[f] for f in feats] + [(t["job_type_public"][i], t["model_type_public"][i],
                                              t["priority_class"][i], t["cluster_id"][i])]
        else:
            for j, f in enumerate(feats):
                r[j].append(f)
    out = []
    for (wl, gpu), v in acc.items():
        feats = [float(np.mean(x)) if x else 0.0 for x in v[:len(NUM)]]
        out.append((wl, gpu, feats, v[len(NUM)]))
    return out


def build_matrix(rows, vocab):
    X = np.zeros((len(rows), len(NUM) + sum(len(v) for v in vocab.values())), dtype=np.float32)
    for i, (_, _, feats, cat) in enumerate(rows):
        X[i, :len(NUM)] = feats
        off = len(NUM)
        for col, val in zip(["job", "model", "prio", "cluster"], cat):
            idx = vocab[col].get(val)
            if idx is not None:
                X[i, off + idx] = 1.0
            off += len(vocab[col])
    return X


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "~/xcad/datasets/v2026/day180"
    cuts = [int(c) for c in sys.argv[2:]] or None
    files = sorted(glob.glob(os.path.join(d, "*.parquet")), key=key_of)
    snaps = [head_rows(f) for f in files]
    print("window %s  snapshots=%d  heads/snapshot=%s" %
          (os.path.basename(d), len(snaps), [len(s) for s in snaps[:3]]))
    from sklearn.linear_model import LogisticRegression
    print()
    hdr = ("%-5s %8s %8s | %7s %7s %7s %7s | %9s %9s" %
           ("cut", "test_e", "rec_e", "MEM", "PROF", "LR", "LR+cl", "LRwrong", "onlyMEM"))
    print(hdr)
    for cut in (cuts or [6, 12, 18, 23]):
        if cut >= len(snaps):
            continue
        train = [r for si in range(cut) for r in snaps[si]]
        hist = collections.defaultdict(list)
        for si in range(cut):
            for (wl, gpu, feats, cat) in snaps[si]:
                hist[wl].append((si, gpu, feats, cat))
        vocab_nocluster = {c: {} for c in ["job", "model", "prio", "cluster"]}
        for (_, _, _, cat) in train:
            for col, val in zip(["job", "model", "prio", "cluster"], cat):
                v = vocab_nocluster[col]
                if val not in v and col != "cluster":
                    v[val] = len(v)
        vocab_full = {k: dict(v) for k, v in vocab_nocluster.items()}
        for (_, _, _, cat) in train:
            v = vocab_full["cluster"]
            if cat[3] not in v:
                v[cat[3]] = len(v)
        classes = sorted(set(g for (_, g, _, _) in train))
        cls_ix = {c: i for i, c in enumerate(classes)}
        ytr = np.array([cls_ix[r[1]] for r in train])
        Xtr_nocl = build_matrix(train, vocab_nocluster)
        Xtr_full = build_matrix(train, vocab_full)
        test = snaps[cut]
        Xte_nocl = build_matrix(test, vocab_nocluster)
        Xte_full = build_matrix(test, vocab_full)
        clf1 = LogisticRegression(max_iter=400, C=1.0, solver="lbfgs")
        clf1.fit(Xtr_nocl, ytr)
        p1 = clf1.predict(Xte_nocl)
        clf2 = LogisticRegression(max_iter=400, C=1.0, solver="lbfgs")
        clf2.fit(Xtr_full, ytr)
        p2 = clf2.predict(Xte_full)
        # categorical-profile prior per class
        prof_cls = collections.defaultdict(collections.Counter)
        for (_, g, _, cat) in train:
            prof_cls[cat[:3]][g] += 1
        truth = [cls_ix[r[1]] for r in test]
        rec = mem_hit = prof_hit = l1_hit = l2_hit = only_mem = lr_wrong = 0
        for i, (wl, g, feats, cat) in enumerate(test):
            h = hist.get(wl)
            if not h or len(h) < 2:
                continue
            rec += 1
            mp = collections.Counter(x[1] for x in h[-2:]).most_common(1)[0][0]
            m = cls_ix[mp]
            pp = prof_cls.get(cat[:3])
            pc = cls_ix[pp.most_common(1)[0][0]] if pp else -1
            t = truth[i]
            mem_hit += (m == t)
            prof_hit += (pc == t)
            l1_hit += (p1[i] == t)
            l2_hit += (p2[i] == t)
            lr_wrong += (p2[i] != t)
            if m == t and p2[i] != t:
                only_mem += 1
        print("%-5d %8d %8d | %7.4f %7.4f %7.4f %7.4f | %9d %9d" %
              (cut, len(test), rec, mem_hit / rec, prof_hit / rec, l1_hit / rec, l2_hit / rec,
               lr_wrong, only_mem))
    print()
    print("columns: MEM=entity history mode, PROF=categorical profile class mode,")
    print("         LR=static resource attributes, LR+cl=LR incl. cluster_id,")
    print("         onlyMEM=recurrent edges MEM gets right that LR+cl gets wrong")


if __name__ == "__main__":
    main()
