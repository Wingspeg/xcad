#!/usr/bin/env python3
"""Screen Alibaba cluster-trace-v2026-maas: fetch the official docs and HEAD the
OSS archives.  Read-only; writes two logs under outputs/reports/.  Run on the server."""
import subprocess, os
PROXY = os.environ.get("XCAD_PROXY", "")          # optional egress proxy, e.g. http://host:port
ENV = {"HOME": os.path.expanduser("~"), "PATH": "/usr/bin:/bin"}
if PROXY:
    ENV.update({"https_proxy": PROXY, "http_proxy": PROXY})
RAW = "https://raw.githubusercontent.com/alibaba/clusterdata/master/"
BUCKET = "https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-v2026-maas/data/"
OUT = os.path.expanduser("~/xcad/xcad_code/outputs/reports")


def curl(args):
    return subprocess.run(["curl", "-sS", "-m", "90"] + args, capture_output=True, text=True, env=ENV).stdout


paths = ["cluster-trace-v2026-maas/README.md",
         "cluster-trace-v2026-maas/docs/schema.md",
         "cluster-trace-v2026-maas/docs/data_download.md",
         "README.md"]
with open(f"{OUT}/screen_maas_readme.log", "w") as fh:
    for p in paths:
        t = curl([RAW + p])
        fh.write("=" * 30 + f" {p}  bytes={len(t)}\n{t}\n")
        print(f"{p:52s} bytes={len(t)}")
    # gpu_type / head-related lines only, for a quick grep in the terminal
    sch = curl([RAW + "cluster-trace-v2026-maas/docs/schema.md"])
    print("\n--- schema lines mentioning type/model/id ---")
    for l in sch.split("\n"):
        if any(k in l for k in ("gpu_type", "model_id", "base_id", "app_id", "cluster_id", "params_b", "day")):
            print("   ", l.strip()[:120])

with open(f"{OUT}/screen_maas_oss_head.log", "w") as fh:
    for f in ["instance_daily.zip", "instance_week_5min.zip", "app_metrics.csv.zip"]:
        r = curl(["-sSI", BUCKET + f])
        keep = [l.strip() for l in r.split("\n")
                if l.lower().startswith(("http/", "content-length", "accept-ranges", "content-type", "x-oss-object"))]
        fh.write("=== " + f + "\n" + "\n".join(keep) + "\n\n")
        print(f"{f:24s} " + " ; ".join(keep[1:4]))
print("DONE")
