#!/usr/bin/env python3
"""One-off: explain the mean-rank gap on "gold not rated" rows between
export_maas_graph.ranks_from_scores (closed form) and the dense reference in
verify_maas_graph.py.  Decomposes the dense rank into kept candidates / block
peers / head-segment peers / filtered, and prints the closed form's own terms.
Not part of the evidence chain -- it exists so §17.2 can state the residual or
close it.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/xcad/xcad_code/src"))
from data.export_maas_graph import REL_ID, _load_txt, channel_matrices  # noqa: E402

D = os.path.expanduser("~/xcad/xcad_code/xcad_model/data/maas")
ent = {}
for line in open(os.path.join(D, "entity2id.txt")):
    name, i = line.rstrip("\n").split("\t")
    ent[name] = int(i)
n_ent = len(ent)
sp = {s: _load_txt(os.path.join(D, f"{s}.txt")) for s in ("train", "valid", "test")}
raw = pd.read_parquet(os.path.expanduser(
    "~/xcad/datasets/maas/instance_daily/six_shards_slim.parquet"))
tails = sorted(set(raw.gpu_type.unique()) - {""})
days = np.sort(raw.day.unique())
tr, va, te = sp["train"], sp["valid"], sp["test"]
last_day = int(days[max(tr.tau.max(), va.tau.max()) - 1])
eid2head = {i: nm for nm, i in ent.items() if nm not in set(tails)}
rng = np.random.default_rng(7)
sub = te.iloc[rng.choice(len(te), 2000, replace=False)].reset_index(drop=True)
chans, filt, gcol, hist = channel_matrices(
    tr, va, sub, raw, tails, ent, 20,
    {"_last_hist_day": last_day, "_eid2head_name": eid2head})

tail_ids = np.array([ent[t] for t in tails])
gold = sub.o.to_numpy()
ar = np.arange(len(sub))
m = chans["entity_memory(head,tail)"]
jit = np.random.default_rng(1).random((len(sub), n_ent)) * 1e-6
dense = np.zeros((len(sub), n_ent), dtype=np.float32)
for j, eid in enumerate(tail_ids):
    dense[:, eid] = m[:, j]
sc = (dense + jit).astype(np.float32)
filt_ent = np.zeros((len(sub), n_ent), dtype=bool)
filt_ent[:, tail_ids] = filt

nc = ~((m[ar, gcol] > 0) & ~filt[ar, gcol])
print(f"n_ent={n_ent} n_tail_block={len(tails)} noncandidate rows={nc.sum()}")
kept = ((m > 0) & ~filt).sum(1)
a = (m > 0).sum(1)
f_only = (filt & (m == 0)).sum(1)
peers = np.maximum(n_ent - a - f_only, 1)
print(f"closed form: mean kept={kept[nc].mean():.3f} mean a={a[nc].mean():.3f} "
      f"mean f_only={f_only[nc].mean():.3f} mean peers={peers[nc].mean():.1f} "
      f"=> E[rank]={np.mean(kept[nc] + (peers[nc] + 1) / 2):.1f}")

gs = sc[ar, gold]
gnc = gs[nc][:, None]
above_all = (sc[nc] > gnc).sum(1)
blk = np.zeros(n_ent, dtype=bool)
blk[tail_ids] = True
above_blk = (sc[nc] > gnc)[:, blk].sum(1)
is_filt = filt_ent[nc]
above_filt_cand = ((sc[nc] > gnc) & is_filt & blk[None, :]).sum(1)
print(f"dense: E[rank]={np.mean(above_all) + 1:.1f}  above(block)={np.mean(above_blk):.3f}"
      f"  above(head segment)={np.mean(above_all - above_blk):.1f}"
      f"  above&filtered={np.mean(above_filt_cand):.3f}")
print(f"dense ties with gold: {(sc[nc] == gnc).sum(1).mean():.3f} entities/row")
print(f"gold value distribution: median={np.median(gs[nc]):.3e} "
      f"min={gs[nc].min():.3e} max={gs[nc].max():.3e}")
nz = (sc[nc] > 0) & (~blk[None, :])
print(f"head-segment entries with score>0: mean per row={nz.sum(1).mean():.1f} of {n_ent - len(tails)}")

# Seed sweep: the harness draws `jit` from default_rng(1) AND feeds
# ranks_from_scores(rng_seed=1) -- if the peer draw and the jitter stream share a
# seed, the two "unbiased uniform" means can move together.  Re-draw the jitter
# only, and both draws independently, and watch the gap.
from data.export_maas_graph import ranks_from_scores  # noqa: E402
for js, rs in ((1, 1), (1, 11), (2, 1), (3, 3), (4, 5)):
    j2 = np.random.default_rng(js).random((len(sub), n_ent)) * 1e-6
    s2 = (dense + j2).astype(np.float32)
    g2 = s2[ar, gold]
    g2nc = s2[ar, gold][nc][:, None]
    ref2 = (s2[nc] > g2nc).sum(1) + 1
    fast2 = ranks_from_scores(m, gcol, filt, n_ent, rng_seed=rs,
                              jitter=j2[:, tail_ids])
    print(f"  jit_seed={js} fast_seed={rs}: dense mean={ref2.mean():9.1f} "
          f"closed mean={fast2[nc].mean():9.1f} gap={fast2[nc].mean() - ref2.mean():+8.1f}")

