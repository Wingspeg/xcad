#!/usr/bin/env python3
"""PART B/C of the Philly deep-dive (GitHub part was done by hand from the
official repo tarball, because api.github.com is rate-limited from our egress).

B: how big is the citing community of the Philly trace paper, and what tasks?
C: does anyone treat a trace's placement decision as the thing to predict?
D: does anyone claim GPU *model names* for Philly (which the schema forbids)?
"""
import json
import re
import subprocess
import time
from collections import Counter

SS = "https://api.semanticscholar.org/graph/v1"


def get(url, to=60):
    p = subprocess.run(["curl", "-sS", "-L", "--max-time", str(to),
                        "-H", "User-Agent: xcad-screen", url],
                       capture_output=True, text=True)
    return p.stdout or ""


def ss(url):
    time.sleep(2)          # anonymous Semantic Scholar is rate-limited per IP
    raw = get(url)
    try:
        return json.loads(raw)
    except Exception:
        print("    (json fail)", raw[:160].replace("\n", " "))
        return {}


print("=" * 96)
print("PART B  the trace paper and its citations")
print("=" * 96)
cands = [
    "Large Deep Learning Clusters at Internet Scale",
    "Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads",
]
pid = None
for q in cands:
    d = ss(f"{SS}/paper/search?query={q.replace(' ', '%20')}"
           "&fields=title,year,venue,citationCount,externalIds&limit=5")
    print(f"\n### q={q}  total={d.get('total')}")
    for p in d.get("data", []):
        print(f"    {p.get('year')} | cites {p.get('citationCount'):>4} | "
              f"{(p.get('title') or '')[:74]} | {p.get('paperId')}")
        if pid is None and re.search(r"Multi-Tenant GPU Clusters|Deep Learning Clusters",
                                     p.get("title") or "", re.I):
            pid = p["paperId"]
print("\nchosen paperId:", pid)
if pid:
    rows = []
    for off in (0, 100, 200, 300, 400, 500):
        d = ss(f"{SS}/paper/{pid}/citations?fields=title,year,venue,abstract"
               f"&limit=100&offset={off}")
        got = d.get("data", [])
        rows += got
        if len(got) < 100:
            break
    print(f"citing papers fetched: {len(rows)}")
    kw = {
        "scheduling/systems": r"schedul|scheduler|placement|gang",
        "simulator/emulation": r"simulat|emulat|replay|trace-driven",
        "characterization": r"characteriz|measurement|empirical|analys|understand|study",
        "prediction/learning": r"predict|learning|learned|reinforcement|neural",
        "utilization/power/energy": r"utiliz|power|energy|carbon",
        "failure/reliability": r"failur|reliab|fault|anomaly",
        "fairness/multi-tenant": r"fair|multi.tenant|quota",
        "LLM/inference": r"LLM|large language|inference",
    }
    cnt = Counter()
    for r in rows:
        c = r.get("citingPaper") or {}
        t = " ".join([(c.get("title") or ""), (c.get("abstract") or "")])
        for k, pat in kw.items():
            if re.search(pat, t, re.I):
                cnt[k] += 1
    print("\n  task census over citing papers (title+abstract):")
    for k, v in cnt.most_common():
        print(f"    {k:<26} {v:>3} / {len(rows)}")

    # who actually consumes WHICH fields, judged from abstract text
    FIELD = {
        "job arrival/request volume": r"job (arrival|submission)|number of jobs|workload character",
        "GPU count requested": r"GPU (count|request|number)|number of GPUs",
        "preemption/re-queue": r"preempt|re-?queue|checkpoint",
        "machine-level placement (ip)": r"placed on|machine (id|assignment)|which machine|node assignment|placement decision",
        "GPU model/heterogeneity": r"GPU (model|type)|heterogene",
        "utilization series": r"utilization",
    }
    fc = Counter()
    ex = {}
    for r in rows:
        c = r.get("citingPaper") or {}
        t = " ".join([(c.get("title") or ""), (c.get("abstract") or "")])
        for k, pat in FIELD.items():
            m = re.search(pat, t, re.I)
            if m:
                fc[k] += 1
                ex.setdefault(k, [])
                if len(ex[k]) < 2:
                    s = max(0, m.start() - 70)
                    ex[k].append(((c.get("title") or "")[:50] + " || " +
                                  " ".join(t[s:m.end() + 110].split())))
    print("\n  field-use census (only where the abstract itself says it):")
    for k, v in fc.most_common():
        print(f"    {k:<30} {v:>3} / {len(rows)}")
        for e in ex[k]:
            print(f"        {e[:230]}")

print("\n" + "=" * 96)
print("PART C/D  targeted searches: placement-as-label, and Philly GPU models")
print("=" * 96)
for label, qq in [
    ("placement as prediction target", "job placement prediction GPU cluster trace"),
    ("learning the scheduler from a trace", "learned scheduler production trace GPU cluster"),
    ("Philly + model names", "Philly GPU cluster trace V100 K80"),
    ("Philly by name", "Philly GPU cluster"),
]:
    d = ss(f"{SS}/paper/search?query={qq.replace(' ', '%20')}"
           "&fields=title,year,abstract,citationCount&limit=10")
    print(f"\n### {label}   q={qq}   total={d.get('total')}")
    for p in d.get("data", []):
        ab = " ".join((p.get("abstract") or "").split())
        print(f"    {p.get('year')} | c{p.get('citationCount') or 0:<4} | {(p.get('title') or '')[:76]}")
        if re.search(r"philly|placement", ab, re.I):
            m = re.search(r"[^.]*?(Philly|placement)[^.]*\.", ab, re.I)
            if m:
                print(f"        abs: {m.group(0)[:200]}")

print("\nDONE")
