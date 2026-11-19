#!/usr/bin/env python3
"""HPC-testbed class: first-hand reachability probe (screening doc S11).

Question this answers: is there any *anonymously downloadable* public trace,
outside the cloud vendors already excluded in S9.9, that publishes
   (a) job -> node identity placement, and
   (b) node -> GPU hardware label (model name, or count/memory tier)?

Method rule (learned the hard way this round): claim nothing about an external
data source that is not backed by bytes this script actually fetched.  A 403
from api.github.com is NOT evidence of absence, so repo existence is settled by
the HTML 404 page instead.  Nothing is written into the repo except stdout.

Run with egress through whichever proxy your machine uses, if any:
    export https_proxy=http://<your-proxy-host>:<port>
    python3 xcad_code/scripts/screen_hpc_reachability.py
"""
import html as ihtml
import json
import os
import re
import subprocess
import sys
import urllib.parse

OUT = os.environ.get("HPC_PROBE_TMP", "/tmp/hpcscreen")
os.makedirs(OUT, exist_ok=True)

# key -> (url, kind)  kind: page | json | api_search | zenodo
TARGETS = [
    # --- Grid5000 (Inria, FR) ---
    ("g5k_home", "https://www.grid5000.fr/w/Grid5000:Home", "page"),
    ("g5k_tracing", "https://www.grid5000.fr/w/Tracing", "page"),
    ("g5k_api_sid", "https://api.grid5000.fr/sid/sites", "page"),
    ("g5k_traces_repo", "https://github.com/grid5000/grid5000-traces", "page"),
    # --- Chameleon (NSF, US) ---
    ("cham_index", "https://chameleoncloud.readthedocs.io/en/latest/index.html", "page"),
    ("cham_power", "https://chameleoncloud.readthedocs.io/en/latest/technical/"
                   "power_monitoring/index.html", "page"),
    ("cham_data_page", "https://www.chameleoncloud.org/data/", "page"),
    # --- HPCTraceKit (the public HPC scheduling-log collection) ---
    ("htk_github", "https://github.com/hpc/traces", "page"),
    ("htk_pypi_json", "https://pypi.org/pypi/hpctracekit/json", "page"),
    # --- TACC / CloudLab ---
    ("tacc_mirror", "http://ipsych.korea.ac.kr/papers/trace/tacc/", "page"),
    ("cloudlab_root", "https://www.cloudlab.us/", "page"),
    ("cloudlab_goldfish", "https://github.com/c2smith/goldfish", "page"),
]

PATTERNS = {
    "gpu": r"\bGPU\b|nvidia|Tesla|V100|A100|H100|GH200",
    "slurm": r"slurm|\bsbatch\b|scontrol|accounting",
    "placement": r"node ?id|host ?name|nodename|compute node|node list|hostname",
    "download": r"\.tar\.bz2|\.tar\.gz|\.csv|\.parquet|zenodo|figshare|dataverse|doi:10\.",
    "authwall": r"login required|please log in|sign in to view|javascript is disabled",
}


def curl(url, dst):
    p = subprocess.run(
        ["curl", "-sS", "-L", "--max-time", "90", "-o", dst,
         "-w", "%{http_code} %{size_download} %{url_effective}", url],
        capture_output=True, text=True)
    return (p.stdout or "").strip(), (p.stderr or "").strip()[:120]


def to_text(path):
    try:
        raw = open(path, "rb").read()
    except OSError:
        return ""
    t = raw.decode("utf-8", errors="replace")
    if "<!doctype html" in t[:300].lower() or "<html" in t[:2000].lower():
        t = re.sub(r"(?s)<(script|style).*?</\1>", " ", t)
        t = ihtml.unescape(re.sub(r"(?s)<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t)


def report(key, url):
    dst = os.path.join(OUT, key + ".blob")
    meta, err = curl(url, dst)
    body = to_text(dst)
    hits = {k: len(re.findall(v, body, re.I)) for k, v in PATTERNS.items()}
    verdict = "-"
    if hits["authwall"]:
        verdict = "AUTH-WALL (bytes fetched, but content is a login/anti-bot page)"
    elif re.search(r"Page not found", body) and "github.com" in url:
        verdict = "GITHUB-404 (repo path does not exist)"
    elif meta.startswith("4"):
        verdict = "HTTP-" + meta[:3]
    print(f"--- {key:<17} {meta:<60} chars={len(body):<7} {verdict}")
    print(f"    {url}")
    print(f"    hits={hits}" + (f"  err={err}" if err else ""))
    m = re.search(PATTERNS["gpu"], body, re.I)
    if m:
        s = max(0, m.start() - 130)
        print(f"    [gpu] ...{body[s:m.start() + 220]}...")
    return body


print("=" * 100)
print("PART 1  entry points, one fetch each")
print("=" * 100)
bodies = {k: report(k, u) for k, u, _kind in TARGETS}

# Chameleon: which doc links could possibly be a data portal?
print("\n  cham_index links mentioning monitor/data/metric/trace:")
raw = open(os.path.join(OUT, "cham_index.blob"), encoding="utf-8", errors="replace").read()
for m in re.finditer(r'href="([^"]*)"[^>]*>([^<]{2,60})', raw):
    u, x = m.group(1), m.group(2)
    if re.search(r"monitor|data|metric|trace", u + x, re.I):
        print(f"     {x.strip()} -> {u}")
raw = open(os.path.join(OUT, "cham_power.blob"), encoding="utf-8", errors="replace").read()
print("  cham_power external data links (zenodo|figshare|dataverse|globus|.csv|.tar):",
      len(re.findall(r"zenodo|figshare|dataverse|globus|\.csv|\.tar", raw, re.I)))
raw = open(os.path.join(OUT, "cloudlab_root.blob"), encoding="utf-8", errors="replace").read()
print("  cloudlab_root links mentioning trace/measure/dataset:",
      [m.group(1) for m in re.finditer(r'href="([^"]*)"', raw)
       if re.search(r"trace|measure|dataset", m.group(1), re.I)][:10])

print("\n" + "=" * 100)
print("PART 2  discovery APIs (anonymous): Zenodo records + GitHub repo search")
print("=" * 100)
ZQ = ['"gpu cluster trace" slurm',
      '"GPU cluster workload trace"',
      "hpctracekit",
      "slurm accounting open data GPUs"]
for i, q in enumerate(ZQ):
    url = "https://zenodo.org/api/records?" + urllib.parse.urlencode(
        {"q": q, "size": 10})
    dst = os.path.join(OUT, f"zen_{i}.json")
    meta, err = curl(url, dst)
    try:
        d = json.load(open(dst))
        h = d.get("hits", {})
        print(f"--- zenodo q={q!r:<38} {meta} total={h.get('total')}")
        print(f"    {url}")
        for x in (h.get("hits") or [])[:5]:
            fl = [f.get("key") for f in (x.get("files") or [])][:3]
            print("      ", (x.get("title") or "")[:92], "|", fl)
    except Exception as e:
        print(f"--- zenodo q={q!r} {meta} FAILED {e} {err}")

GQ = ["hpctracekit", "hpc job traces slurm gpu", "cluster trace gpu placement dataset"]
for q in GQ:
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": q, "per_page": 8})
    p = subprocess.run(["curl", "-sS", "-L", "--max-time", "60",
                        "-H", "Accept: application/vnd.github+json", url],
                       capture_output=True, text=True)
    try:
        d = json.loads(p.stdout)
    except Exception:
        d = {}
    if "total_count" not in d:
        print(f"--- github q={q!r} rate-limited/error -> {str(d)[:110]}")
        print("    (not evidence of absence; existence settled in PART 1 via HTML)")
        continue
    print(f"--- github q={q!r:<38} total={d['total_count']}")
    for it in d.get("items", [])[:5]:
        print("      ", it["full_name"], "|", (it.get("description") or "")[:70])

print("\nDONE  (0 GPU minutes; every claim above is backed by bytes in " + OUT + ")")
