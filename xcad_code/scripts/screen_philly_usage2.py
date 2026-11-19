#!/usr/bin/env python3
"""Pass 2 on the Philly trace: what the *official* repo analyses consume, and who
cites the trace paper.

Pass 1 (screen_philly_usage.py) showed the arXiv-visible use pattern is
arrivals / requests / preemption / fairness-JCT, i.e. simulator inputs.  The
remaining question decides whether Philly can be our second dataset:

  Q1  Does any published analysis of this trace read the *machine* side at all
      (Model / Memory / Number_of_GPU), or only the job side?
      -> read msr-fiddle/philly-traces: file tree, README, and the column names
         that appear in its Analysis notebooks.
  Q2  How large is the citing community and what tasks do they run?
      -> Semantic Scholar citations of the ATC'19 trace paper.
"""
import json
import re
import subprocess
import urllib.parse

API = "https://api.github.com"
HDR = ["-H", "Accept: application/vnd.github+json"]


def curl(url, to=90, extra=()):
    p = subprocess.run(["curl", "-sS", "-L", "--max-time", str(to)] + list(extra) +
                       ["-H", "User-Agent: xcad-screen", url],
                       capture_output=True, text=True)
    return p.stdout or ""


def gh(url):
    raw = curl(url, extra=HDR)
    try:
        return json.loads(raw)
    except Exception:
        return {"__err__": raw[:200]}


print("=" * 96)
print("PART A  msr-fiddle/philly-traces : file tree, README, notebook columns")
print("=" * 96)
meta = gh(f"{API}/repos/msr-fiddle/philly-traces")
print("repo:", meta.get("full_name"), "| star", meta.get("stargazers_count"),
      "| desc:", (meta.get("description") or "")[:120],
      "| license:", (meta.get("license") or {}).get("spdx_id"),
      "| pushed:", meta.get("pushed_at"))
sha = meta.get("default_branch", "master")
tree = gh(f"{API}/repos/msr-fiddle/philly-traces/git/trees/{sha}?recursive=1")
paths = [t["path"] for t in tree.get("tree", [])] if "tree" in tree else []
if not paths:
    print("  tree fetch failed:", str(tree)[:200])
print(f"files={len(paths)}")
for p in paths:
    print("   ", p)

COLS = [r"\bNumber_of_GPU", r"\bModel\b", r"\bMemory\b", r"\bAttempts\b",
        r"\bbudget_vc\b", r"\bfairness\b", r"\bmachine\b", r"\bMachine\b",
        r"\Bgpu\d+", r"\bStatus\b", r"\bTime\b", r"\bSpace\b", r"\bParameter\b",
        r"cluster_job_log", r"machine_configs", r"cluster_machine_log",
        r"\bxlarge", r"\byet\t?another\b"]
hits = {}
nb = [p for p in paths if p.endswith((".ipynb", ".py", ".md", ".txt"))]
print(f"\nsource/analysis files to grep: {len(nb)}")
# raw.githubusercontent.com is not subject to the api.github.com anonymous
# 60-req/hr budget (our shared egress IP already got a 403 once today).
RAW = f"https://raw.githubusercontent.com/msr-fiddle/philly-traces/{sha}/"
for p in nb[:40]:
    content = curl(RAW + urllib.parse.quote(p), to=60)
    if content.startswith(("404: ", "403: ", "<!DOCTYPE")):
        print(f"    {p}  (fetch failed: {content[:60]!r})")
        continue
    for c in COLS:
        n = len(re.findall(c, content))
        if n:
            hits.setdefault(c, [0, []])
            hits[c][0] += n
            if len(hits[c][1]) < 2:
                m = re.search(c, content)
                s = max(0, m.start() - 90)
                hits[c][1].append(p + ": " + " ".join(content[s:m.end() + 90].split()))
    print(f"    {p}  {len(content):,} chars")

print("\n  vocabulary found in the official repo:")
if not hits:
    print("    (none of COLS matched)")
for c, (n, ex) in sorted(hits.items(), key=lambda kv: -kv[1][0]):
    print(f"    {c:<22} n={n}")
    for e in ex:
        print(f"        {e[:200]}")

print("\n" + "=" * 96)
print("PART B  Semantic Scholar: citations of the Philly trace paper")
print("=" * 96)
SS = "https://api.semanticscholar.org/graph/v1"


def ss(url):
    raw = curl(url, to=60)
    try:
        return json.loads(raw)
    except Exception:
        return {"__err__": raw[:200]}


q = ss(f"{SS}/paper/search?query=" +
       "Large%20Deep%20Learning%20Clusters%20at%20Internet%20Scale&fields=title,year,venue,citationCount,externalIds&limit=5")
for p in q.get("data", [])[:3]:
    print("   ", p.get("title"), p.get("year"), p.get("citationCount"), p["paperId"])
pid = None
for p in q.get("data", [])[:3]:
    if "Internet Scale" in (p.get("title") or "") or "Deep Learning Clusters" in (p.get("title") or ""):
        pid = p["paperId"]
        break
print("chosen paperId:", pid)
if pid:
    cites = ss(f"{SS}/paper/{pid}/citations?fields=title,year,venue&limit=200")
    rows = cites.get("data", [])
    print(f"  citing papers returned: {len(rows)} (total may be larger)")
    kw = {
        "scheduling": r"schedul",
        "placement": r"placement|place",
        "simulator": r"simulat",
        "workload characterization": r"characteriz|measurement|empirical|analysis of",
        "prediction/learning": r"predict|learn|model selection|reinforcement",
        "preemption/elastic": r"preempt|elastic|spot",
        "fairness": r"fair",
        "heterogeneity": r"heterogen",
        "energy": r"energy|power",
        "failure/reliability": r"failur|reliab|fault",
    }
    from collections import Counter
    cnt = Counter()
    for r in rows:
        t = (r.get("citingPaper") or {}).get("title") or ""
        for k, pat in kw.items():
            if re.search(pat, t, re.I):
                cnt[k] += 1
    for k, v in cnt.most_common():
        print(f"    {k:<28} {v:>3} / {len(rows)}")
    print("  sample titles:")
    for r in rows[:25]:
        c = r.get("citingPaper") or {}
        print(f"    {c.get('year')} | {(c.get('title') or '')[:92]}")

print("\n" + "=" * 96)
print("PART C  Is anyone using a trace's placement decision as the LABEL?")
print("=" * 96)
for qq in ("GPU%20cluster%20job%20placement%20prediction",
          "learned%20scheduler%20trace%20placement%20accuracy",
          "node%20assignment%20prediction%20cluster%20trace%20machine%20learning"):
    d = ss(f"{SS}/paper/search?query={qq}&fields=title,year,venue,abstract&limit=8")
    print(f"\n### q={qq.replace('%20',' ')}  total={d.get('total')}")
    for p in d.get("data", []):
        ab = " ".join((p.get("abstract") or "").split())[:170]
        print(f"    {p.get('year')} | {(p.get('title') or '')[:80]}")
        if re.search(r"placement", ab, re.I):
            print(f"        abs: {ab}")

print("\nDONE")
