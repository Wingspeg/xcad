#!/usr/bin/env python3
"""How does the literature actually USE the Microsoft Philly trace?

Motivation: our own screen excluded Philly because the GPU *model name* is not
published (only (gpu count, per-card memory) tiers) and because the recurrence
channel does not separate from the class prior.  Before we decide whether to
build anything on Philly, find out empirically which columns other papers use,
and whether anyone has used job -> machine placement as a prediction target.

Method: query the arXiv and GitHub APIs (anonymous), fetch the full text of the
most relevant hits via ar5iv, and grep for the vocabulary that identifies a use
pattern:  attempts / vc / preemption / re-queue / gpu type / machine id.
"""
import json
import re
import subprocess
import time
import urllib.parse
import xml.etree.ElementTree as ET

# Network egress: curl inherits https_proxy/http_proxy from the environment.


def curl(url, to=90):
    p = subprocess.run(["curl", "-sS", "-L", "--max-time", str(to), url],
                       capture_output=True, text=True)
    return p.stdout or ""


def arxiv(q, max_results=12):
    # NOTE: q contains spaces and double quotes -> must be urlencoded.
    # The same bug cost us a re-run on the Zenodo screen earlier today.
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": q, "start": 0,
        "max_results": max_results, "sortBy": "relevance"})
    raw = curl(url)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        print("  (arxiv parse failed)", raw[:120])
        return []
    out = []
    for e in root.findall("a:entry", ns):
        t = " ".join((e.findtext("a:title", "", ns) or "").split())
        s = " ".join((e.findtext("a:summary", "", ns) or "").split())
        i = e.findtext("a:id", "", ns)
        d = (e.findtext("a:published", "", ns) or "")[:10]
        out.append((d, t, i, s))
    return out


QUERIES = [
    ('all:"Philly" AND all:"GPU cluster"', "philly by name"),
    ('abs:"GPU cluster trace" AND abs:"job"', "gpu cluster trace"),
    ('all:"training jobs" AND all:"preemption" AND all:"GPU cluster"', "preemption"),
    ('ti:"GPU" AND abs:"re-queue" ', "re-queue"),
    ('all:"machine_configs" OR all:"cluster_job_log"', "raw Philly column names"),
    ('all:"Philly" AND all:"NVIDIA"', "does anyone name the cards"),
    ('abs:"GPU job scheduling" AND abs:"trace-driven"', "scheduling + trace-driven"),
]

print("=" * 96)
print("PART A  arXiv: which papers mention the Philly trace / these fields")
print("=" * 96)
seen = {}
for q, label in QUERIES:
    print(f"\n### {label}   q={q}")
    for d, t, i, s in arxiv(q)[:10]:
        hit = re.search(r"(\d{4}\.\d{5})", i)
        aid = hit.group(1) if hit else i
        if aid in seen:
            print(f"    [dup] {d} {t[:80]}")
            continue
        seen[aid] = (d, t, s)
        mark = "PHILLY" if "philly" in (t + s).lower() else ""
        print(f"    {aid} {d} {mark:7s} {t[:86]}")

print("\n" + "=" * 96)
print("PART B  fetch full text for the Philly-named hits and grep the use vocabulary")
print("=" * 96)
TERMS = {
    "job log fields": r"attempts|job log|cluster_job_log|vc\b|fairness|re-?queu",
    "placement target": r"assigned (to|on) (a|the) (machine|node)|which (machine|node)|placement|schedule on",
    "model name": r"K80|M40|P40|P100|V100|GPU (model|type|card type)",
    "capacity tiers": r"12 ?GB|24 ?GB|number of GPUs|memory per GPU",
    "arrival/sim only": r"arrival|trace-driven simulat|workload generat|open-loop",
    "prediction task": r"predict|learn(ed)? schedul|classif",
}
cands = [a for a, (d, t, s) in seen.items() if "philly" in (t + s).lower()]
# papers that *use* the trace often don't put "Philly" in the abstract; fill the
# remaining slots with GPU-cluster-trace papers so the vocabulary grep has teeth.
fill = [a for a, (d, t, s) in seen.items()
        if "philly" not in (t + s).lower()
        and "trace" in (t + s).lower() and "gpu" in (t + s).lower()]
cands = (cands + fill)[:6]
print("candidates:", cands)
for aid in cands:
    html = curl(f"https://ar5iv.org/abs/{aid}", to=120)
    txt = re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", html))
    print(f"\n--- {aid}  {seen[aid][1][:80]}  chars={len(txt):,}")
    if len(txt) < 4000:
        print("    (page too small / redirect; skipping)")
        continue
    for k, pat in TERMS.items():
        ms = re.findall(pat, txt, re.I)
        if ms:
            m = re.search(pat, txt, re.I)
            s = max(0, m.start() - 110)
            print(f"    [{k}] n={len(ms):<4} ...{txt[s:m.end()+170]}...")
    time.sleep(2)

print("\n" + "=" * 96)
print("PART C  GitHub: who ships a Philly parser, and what columns do they emit")
print("=" * 96)
for q in ("philly-traces", "philly gpu cluster", "msr-fiddle",
          "gpu cluster trace analysis", "cluster-gpu-trace"):
    raw = curl("https://api.github.com/search/repositories?q=" +
               q.replace(" ", "+") + "&per_page=6")
    try:
        d = json.loads(raw)
    except Exception:
        print("  (rate-limited or bad json)")
        continue
    if "items" not in d:
        print(f"  q={q}: {str(d)[:100]}")
        continue
    print(f"\n### q={q}  total={d.get('total_count')}")
    for it in d["items"][:6]:
        print(f"    {it['full_name']} | star {it['stargazers_count']} | "
              f"{(it.get('description') or '')[:70]}")

print("\nDONE")
