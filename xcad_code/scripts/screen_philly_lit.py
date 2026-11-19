#!/usr/bin/env python3
"""Philly deep-dive, part 3: who cites/uses the trace and what do they do with it.

Semantic Scholar answered 429 and OpenAlex answered "anonymous search rate
limited" from our shared egress IP, so discovery runs on arXiv's Atom API
(generous) and OpenAlex is retried with backoff in a *small* number of calls.

Questions this must answer, in order of how much they cost us:
  Q1  Is there any paper whose *task* is predicting job -> machine/node from a
      production trace (i.e. our r1/placement-as-label framing)?
  Q2  Do papers using Philly report GPU model names, or only (count, memory)?
  Q3  Citation footprint of the trace paper (OpenAlex, if it answers).
"""
import json
import re
import subprocess
import time
import urllib.parse
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}


def curl(url, to=90):
    p = subprocess.run(["curl", "-sS", "-L", "--max-time", str(to),
                        "-A", "xcad-screen", url],
                       capture_output=True, text=True)
    return p.stdout or ""


def arxiv(q, n=15):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": q, "start": 0, "max_results": n, "sortBy": "relevance"})
    raw = curl(url)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return None
    out = []
    for e in root.findall("a:entry", NS):
        t = " ".join((e.findtext("a:title", "", NS) or "").split())
        s = " ".join((e.findtext("a:summary", "", NS) or "").split())
        i = e.findtext("a:id", "", NS) or ""
        d = (e.findtext("a:published", "", NS) or "")[:10]
        m = re.search(r"(\d{4}\.\d{5})", i)
        out.append((m.group(1) if m else i, d, t, s))
    return out


QUERIES = [
    ("placement is the label", 'abs:"job placement"'),
    ("placement+GPU", 'abs:"job placement" AND abs:"GPU"'),
    ("node assignment from trace", 'abs:"node assignment" AND abs:"trace"'),
    ("which machine", 'abs:"which server" OR abs:"which machine"'),
    ("Philly in full text", 'abs:"Philly"'),
    ("machine_configs fingerprint", 'abs:"machine_configs" OR abs:"cluster_job_log"'),
    ("heterogeneous GPU cluster + trace", 'abs:"heterogeneous GPU cluster" AND abs:"trace"'),
    ("graph learned on cluster trace", 'abs:"cluster trace" AND abs:"graph neural"'),
    ("temporal graph + scheduling", 'abs:"temporal graph" AND abs:"scheduling"'),
]

print("=" * 96)
print("Q1/Q2  arXiv survey of task framing")
print("=" * 96)
pool = {}
for label, q in QUERIES:
    rows = arxiv(q)
    if rows is None:
        print(f"\n### {label}  (parse failed)")
        continue
    print(f"\n### {label}   q={q}   n={len(rows)}")
    for aid, d, t, s in rows:
        pool.setdefault(aid, (d, t, s, label))
        ph = "PHILLY" if "philly" in (t + s).lower() else ""
        print(f"    {aid} {d} {ph:6s} [{label}] {t[:80]}")
    time.sleep(3)

print(f"\npool size (unique ids) = {len(pool)}")
print("\n--- papers whose abstract says placement is *learned/predicted* ---")
pat = re.compile(r"(predict|learn|infer|classif|neural|graph).{0,90}(placement|assignment)|(placement|assignment).{0,90}(predict|learn|infer|classif)", re.I)
for aid, (d, t, s, lab) in sorted(pool.items()):
    m = pat.search(s)
    if m:
        print(f"  {aid} {d} {t[:74]}")
        print(f"      ...{' '.join(s[max(0,m.start()-60):m.end()+90].split())}...")

print("\n--- papers naming Philly, and the vocabulary they use ---")
TERMS = {
    "placement/ip": r"placement|placed|which (machine|server|node)|node assignment",
    "model name": r"K80|M40|P40|P100|V100|A100|T4|GPU (model|type)",
    "count/memory": r"number of GPUs|GPU (count|memory)|memory per GPU|12 ?GB|24 ?GB",
    "arrival/sim": r"arrival|simulat|open-loop|replay",
    "preempt/elastic": r"preempt|elastic|re-?queue|checkpoint",
    "fairness/JCT": r"fairness|JCT|completion time|queueing|waiting time",
    "utilization": r"utiliz|idle|fragment",
    "failure": r"failur|reliab|anomaly",
}
for aid, (d, t, s, lab) in sorted(pool.items(), key=lambda kv: kv[0]):
    if "philly" not in (t + s).lower():
        continue
    print(f"\n  {aid} {d} {t[:80]}")
    for k, p in TERMS.items():
        ms = re.findall(p, s, re.I)
        if ms:
            m = re.search(p, s, re.I)
            print(f"      [{k}] n={len(ms)}  ...{' '.join(s[max(0,m.start()-70):m.end()+110].split())}...")

print("\n--- papers that report GPU *model names* for a public trace (any dataset) ---")
for aid, (d, t, s, lab) in sorted(pool.items(), key=lambda kv: kv[0]):
    m = re.search(r"(K80|M40|P40|P100|V100|A100|T4|L40S)", s)
    if m:
        print(f"  {aid} {d} {t[:70]}")
        mm = re.search(r"[^.]*(K80|M40|P40|P100|V100|A100|T4|L40S)[^.]*\.", s)
        print(f"      {' '.join((mm.group(0) if mm else '').split())[:220]}")

print("\n" + "=" * 96)
print("Q3  OpenAlex citation footprint of the trace paper (retried, few calls)")
print("=" * 96)
oa = "https://api.openalex.org"
sel = "&select=id,publication_year,cited_by_count,display_name"
for attempt in range(4):
    raw = curl(f"{oa}/works?search=" + urllib.parse.quote(
        "Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads") +
        "&per-page=5" + sel, to=40)
    try:
        d = json.loads(raw)
    except Exception:
        d = {}
    if "results" in d:
        for w in d["results"]:
            print(f"    {w['id'].rsplit('/',1)[-1]:>12} {w.get('publication_year')} "
                  f"cites={w.get('cited_by_count'):>4} {w['display_name'][:72]}")
        wid = None
        for w in d["results"]:
            if "Multi-Tenant GPU" in w["display_name"]:
                wid = w["id"].rsplit("/", 1)[-1]
                break
        if wid:
            time.sleep(12)
            raw = curl(f"{oa}/works?filter=cites:{wid}&per-page=200" + sel, to=60)
            try:
                c = json.loads(raw)
            except Exception:
                print("    (citing list fetch failed)")
                c = {}
            rows = c.get("results", [])
            print(f"  citing works returned: {len(rows)} of {c.get('meta', {}).get('count')}")
            cnt = {}
            for r in rows:
                t = r["display_name"] or ""
                for k, p in {"schedul": r"schedul", "placement": r"placement|plac",
                             "simulat": r"simulat", "heterogen": r"heterogen",
                             "predict/learn": r"predict|learn|neural",
                             "fair": r"fair", "energy": r"energ|power",
                             "LLM": r"LLM|large language", "failur": r"failur|reliab",
                             "graph": r"graph"}.items():
                    if re.search(p, t, re.I):
                        cnt[k] = cnt.get(k, 0) + 1
            print("    title census:", cnt)
            for r in rows[:30]:
                print(f"      {r.get('publication_year')} c{r.get('cited_by_count'):<4} {(r.get('display_name') or '')[:86]}")
        break
    print(f"    attempt {attempt+1}: {raw[:150]}")
    time.sleep(35)

print("\nDONE")
