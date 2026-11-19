#!/usr/bin/env python3
"""Screen Microsoft Philly (msr-fiddle/philly-traces) against the R1-1 criteria.

Criteria (fixed before reading numbers, same four as screen_v2026.py):
  (i)   leak-free cohort >= 500 test edges per cut (entity recurs over snapshots)
  (ii)  the two competing channels must be SEPARABLE:
          MEM   = the entity's own historical mode of the outcome
          STAT  = best static predictor that is not the entity id
        and the outcome must not be implied by the job's own request
  (iii) a GPU *class* label per placement, with enough classes to be non-trivial
  (iv)  trainable on one 3090 (scale order of v2020)

Run on the server:  python3 screen_philly.py <dir-with-cluster_job_log>
"""
import collections
import json
import os
import statistics
import sys

TIER = {}          # machineId -> "2x12GB" / "8x24GB"
NGPU = {}          # machineId -> number of gpus on that server


def load_machines(path):
    with open(path) as f:
        next(f)
        for line in f:
            p = line.split(",")
            if len(p) < 3:
                continue
            mid = p[0].strip()
            ng = p[1].strip()
            mem = p[2].strip().lower().replace(" ", "")
            TIER[mid] = "%s_%s" % (ng, mem)
            NGPU[mid] = int(ng)


def main():
    d = sys.argv[1]
    load_machines(os.path.join(d, "cluster_machine_list"))
    jobs = json.load(open(os.path.join(d, "cluster_job_log")))
    print("jobs=%d machines_tiered=%d" % (len(jobs), len(TIER)))
    print("(iii) tiers: %s" % dict(collections.Counter(TIER.values())))

    rows = []       # one record per job that actually got placed
    bad_ts = 0
    for j in jobs:
        atts = j.get("attempts") or []
        placed = [a for a in atts if isinstance(a.get("start_time"), str)
                  and a["start_time"][:4].isdigit()]
        if not atts and not placed:
            bad_ts += 1
            continue
        if not placed:
            bad_ts += 1
            continue
        last = placed[-1]
        det = last.get("detail") or []
        if not det:
            continue
        per_mach = [len(x.get("gpus") or []) for x in det]
        ng = max(per_mach)
        tiers = sorted({TIER.get(x.get("ip"), "?") for x in det})
        if "?" in tiers:
            continue
        st = last["start_time"]
        rows.append({
            "user": j.get("user"), "vc": j.get("vc"), "job": j.get("jobid"),
            "day": st[:10], "ts": st, "ng": ng, "nmach": len(det),
            "forced8": ng > 2,
            "tier": tiers[0] if len(tiers) == 1 else "MIXED",
            "n_att": len(placed),
            "att_tiers": [sorted({TIER.get(x.get("ip"), "?") for x in (a.get("detail") or [])})[0]
                          for a in placed
                          if (a.get("detail") and all(x.get("ip") in TIER for x in a["detail"]))],
        })
    print("(iv) placed jobs=%d  span=%s .. %s   (dropped: no usable timestamp=%d)"
          % (len(rows), min(r["ts"] for r in rows)[:10], max(r["ts"] for r in rows)[:10], bad_ts))
    days = sorted({r["day"] for r in rows})
    print("     distinct days=%d  users=%d  vcs=%d" %
          (len(days), len({r["user"] for r in rows}), len({r["vc"] for r in rows})))
    per_user = collections.Counter(r["user"] for r in rows)
    c = collections.Counter(per_user.values())
    print("     jobs/user: median=%d p90=%d max=%d ; users with >=10 jobs=%d ; 1 job=%d" % (
        statistics.median(sorted(per_user.values())),
        sorted(per_user.values())[int(.9 * len(per_user))], max(per_user.values()),
        sum(v for k, v in c.items() if k >= 10), c[1]))
    print("     multi-tier-capable users (ever on both tiers): %d / %d" % (
        len([u for u, _ in per_user.items()
             if len({r["tier"] for r in rows if r["user"] == u}) > 1]), len(per_user)))

    # ---- confound check: is the tier implied by the job's own GPU request? ----
    tab = collections.Counter((r["forced8"], r["tier"]) for r in rows)
    print("\n== request-size vs outcome tier (per-server GPU count > 2 forces the 8-GPU tier) ==")
    for f8 in (False, True):
        tot = sum(w for (x, t), w in tab.items() if x == f8)
        if not tot:
            continue
        dist = {t: w for (x, t), w in tab.items() if x == f8}
        print("   per-server GPUs %s : total=%6d  %s" % (
            "<=2" if not f8 else ">2", tot,
            "  ".join("%s=%.3f" % (t, w / tot) for t, w in sorted(dist.items()))))
    n_f8 = sum(w for (x, t), w in tab.items() if x)
    tot = sum(tab.values())
    print("   requests that DETERMINE the tier: %d / %d = %.1f%%" % (n_f8, tot, 100.0 * n_f8 / tot))

    # ---- real re-placement across attempts ----
    mig = [r for r in rows if len(set(r["att_tiers"])) > 1]
    print("\n== true migration: same job, different tier across retry attempts: %d / %d"
          % (len(mig), len(rows)))

    # ---- (i)+(ii) leak-free cohort, entity = user (and vc x user) ----
    for ent_name, key in (("user", lambda r: r["user"]),
                          ("vc:user", lambda r: (r["vc"], r["user"]))):
        by_day = collections.defaultdict(list)
        for r in rows:
            if r["tier"] != "MIXED":
                by_day[r["day"]].append(r)
        all_days = sorted(by_day)
        print("\n== entity=%s : leak-free cohort (cut every 10 days + last) ==" % ent_name)
        print("%-11s %8s %8s %10s %8s %8s %10s" %
              ("cut_day", "test_e", "cohort", "rec_share", "MEM", "PRIOR", "onlyMEM"))
        tot_c = 0
        worst = None
        for i, day in enumerate(all_days):
            if i % 10 and i != len(all_days) - 1:
                continue
            if i:
                hist = collections.defaultdict(collections.Counter)
                hcount = collections.Counter()
                for dd in all_days[:i]:
                    for r in by_day[dd]:
                        hist[key(r)][r["tier"]] += 1
                        hcount[r["tier"]] += 1
                test = [r for r in by_day[day] if r["tier"] != "MIXED"]
                rec = [r for r in test if sum(hist[key(r)].values()) >= 2]
                prior = hcount.most_common(1)[0][0] if hcount else None
                mem_hit = sum(1 for r in rec
                              if hist[key(r)].most_common(1)[0][0] == r["tier"])
                pri_hit = sum(1 for r in rec if r["tier"] == prior)
                only = sum(1 for r in rec
                           if hist[key(r)].most_common(1)[0][0] == r["tier"] and r["tier"] != prior)
                if rec:
                    print("%-11s %8d %8d %9.3f %8.4f %8.4f %10d" % (
                        day, len(test), len(rec), len(rec) / max(1, len(test)),
                        mem_hit / len(rec), pri_hit / len(rec), only))
                    tot_c += len(rec)
                    if worst is None or len(rec) < worst:
                        worst = len(rec)
        print("   (cumulative recurrent edges over all days=%d, min per measured cut=%s)"
              % (tot_c, worst))
        # full-window version: every day after the first 3
        hist = collections.defaultdict(collections.Counter)
        cohort = mem_hit = pri_hit = only = n = 0
        prior_all = collections.Counter(
            r["tier"] for day in by_day for r in by_day[day]
            if r["tier"] != "MIXED" and day <= sorted(by_day)[9]).most_common(1)[0][0]
        for day in sorted(by_day):
            for r in by_day[day]:
                if r["tier"] == "MIXED":
                    continue
                h = hist[key(r)]
                if sum(h.values()) >= 2:
                    cohort += 1
                    m = h.most_common(1)[0][0]
                    n += 1
                    mem_hit += (m == r["tier"])
                    pri_hit += (prior_all == r["tier"])
                    only += (m == r["tier"] and prior_all != r["tier"])
                h[r["tier"]] += 1
        print("   ALL-DAY sweep: cohort=%d  MEM=%.4f  PRIOR=%.4f  onlyMEM=%d" % (
            cohort, mem_hit / max(1, n), pri_hit / max(1, n), only))


if __name__ == "__main__":
    main()
