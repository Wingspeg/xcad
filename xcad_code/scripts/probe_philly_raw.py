"""Independent raw re-parse of the Philly job log: machine list vs gpu indices.

The published tables say 238 of 321 machines declared with 2 GPUs are seen
hosting device indices up to 7.  Either the source contradicts itself or the
cleaner mis-pairs (ip, gpus).  Re-parse the raw JSON from scratch and answer
only that.

usage: ~/xcad/xcad_code/.venv/bin/python scripts/probe_philly_raw.py
"""
import json
import os
import re
from collections import defaultdict

import pandas as pd

RAW = os.path.expanduser("~/xcad/datasets/philly/trace-data/cluster_job_log")
LIST = os.path.expanduser("~/xcad/datasets/philly/trace-data/cluster_machine_list")

declared = {}
with open(LIST) as fh:
    for i, line in enumerate(fh):
        p = [x.strip() for x in line.rstrip("\n").split(",")]
        if i == 0 or len(p) < 3:
            continue
        declared[p[0]] = (int(p[1]), p[2])
print(f"machine list entries: {len(declared)}  "
      f"count vocab={sorted({c for c, _ in declared.values()})}  "
      f"mem vocab={sorted({m for _, m in declared.values()})}")

seen = defaultdict(set)          # ip -> gpu indices ever attached to that ip
pairings = 0
n_detail_multi = 0
n_rec = 0
buf = open(RAW).read().lstrip()
if buf.startswith("["):
    buf = buf[1:]
dec = json.JSONDecoder()
pos, n = 0, len(buf)
while pos < n:
    while pos < n and buf[pos] in " \t\r\n,":
        pos += 1
    if pos >= n or buf[pos] == "]":
        break
    rec, pos = dec.raw_decode(buf, pos)
    n_rec += 1
    for att in rec.get("attempts", []) or []:
        det = att.get("detail", []) or []
        if len(det) > 1:
            n_detail_multi += 1
        for d in det:
            ip, gs = d.get("ip"), d.get("gpus")
            if not ip or gs is None:
                continue
            pairings += 1
            for g in gs:
                m = re.match(r"gpu(\d+)", str(g))
                if m:
                    seen[ip].add(int(m.group(1)))
print(f"job records parsed={n_rec:,}")

print(f"parsed (ip,gpus) pairings={pairings:,}  distinct ips={len(seen):,}  "
      f"attempts with >1 detail entry={n_detail_multi:,}")

rows = []
for ip, idx in seen.items():
    d = declared.get(ip)
    rows.append(dict(ip=ip, n_obs=len(idx), imax=max(idx),
                     decl=d[0] if d else None, mem=d[1] if d else None))
df = pd.DataFrame(rows)
print(f"ips in log not in machine list: {int(df['decl'].isna().sum()):,}")
ok = df.dropna(subset=["decl"])
ok = ok.assign(too_many=ok["n_obs"] > ok["decl"], beyond=ok["imax"] >= ok["decl"])
print("\nobserved distinct indices vs declared GPU count:")
print(pd.crosstab(ok["decl"], ok["n_obs"]).to_string())
print(f"\nmachines with MORE indices than declared: {int(ok['too_many'].sum()):,} "
      f"of {len(ok):,} ({100*ok['too_many'].mean():.2f}%)")
print(f"machines whose max index >= declared count: {int(ok['beyond'].sum()):,} "
      f"({100*ok['beyond'].mean():.2f}%)")
print("\nby declared count:")
print(ok.groupby("decl").agg(machines=("ip", "size"), over=("too_many", "sum"),
                             mean_obs=("n_obs", "mean")).to_string())
