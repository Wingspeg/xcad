#!/usr/bin/env python3
"""Why is entity2id.txt one line short?  Look for names shared between the head
and the tail space, and for empty/blank ids (the classic id-collision class)."""
import os

import pandas as pd

P = os.path.expanduser("~/xcad/datasets/maas/instance_daily/six_shards_slim.parquet")
t = pd.read_parquet(P, columns=["model_id", "gpu_type", "cluster_id"])
Q = ""
for c in ["model_id", "gpu_type", "cluster_id"]:
    s = t[c].astype("string")
    print(f"{c}: rows_eq_empty={int(s.eq(Q).sum()):,}  "
          f"rows_blank_after_strip={int(s.str.strip().eq(Q).sum()):,}  "
          f"null={int(t[c].isna().sum()):,}")

h = set(t.model_id.dropna().unique())
g = set(t.gpu_type.dropna().unique())
sh = sorted(h & g)
print(f"\ndistinct heads={len(h)}  distinct tails={len(g)}  union={len(h | g)}")
print(f"shared names between head and tail space: n={len(sh)}  {sh[:5]}")

# how many graph edges touch those names
e = t[["model_id", "gpu_type", "day"]].dropna().drop_duplicates()
bad = e[e.model_id.isin(sh) | e.gpu_type.isin(sh)]
print(f"edges touching a shared/empty name: {len(bad):,} of {len(e):,}")
print(f"  as head: {int(e.model_id.isin(sh).sum()):,}   as tail: "
      f"{int(e.gpu_type.isin(sh).sum()):,}")
print("\ntail names (sorted, first 3):", sorted(g)[:3])
print("head names (sorted, first 3):", sorted(h)[:3])
