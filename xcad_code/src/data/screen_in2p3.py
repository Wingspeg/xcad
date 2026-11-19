#!/usr/bin/env python3
"""CC-IN2P3 2024 Workload Dataset 筛查 —— 四条判据原样搬运（与 screen_philly.py 同一框架）。

输入：~/xcad/datasets/in2p3/gpu.tsv
  列: month jobid state user account partition eligible start end allocgpus hostnames
  （hostnames 为 Slurm 分配到的机器名；gpu.tsv 只含 partition ~ /gpu/ 或 allocgpus>0 的作业）

判据（看数之前写的，与 §1 一致）：
  (iii) GPU 类别数 —— 卡型 / 节点档位的类数，需要 >= 3 才有"异构"可谈
  (i)   无泄漏 cohort —— 存在切点使"该实体历史 >=2 次"的测试边 >= 500 条
  (ii)  通道分离 —— 同一批复发边上 MEM 明显高于 PRIOR（历史类别先验），
        且 onlyMEM（MEM 对而 PRIOR 错）有量级 —— 只有分得开，归因才唯一
  (iv)  规模/成本 —— 单卡能训

用法：python3 screen_in2p3.py gpu.tsv
"""
import collections
import sys

# 论文 §2.2 Table 1 的 GPU 节点分组（三组都是 4xV100，差别在 CPU/内存）
GROUP = {}
for h in ["02", "10", "11", "12", "13", "14"]:
    GROUP["g%s" % h] = "gA_20c_192G"
for h in ["04", "05", "06", "07", "09", "15", "16", "17", "18", "19"]:
    GROUP["g%s" % h] = "gB_24c_192G"
for h in ["01", "08"]:
    GROUP["g%s" % h] = "gC_24c_1546G"
# 文件里的机器名是 g001 这种三位数，做一次归一
GROUP = {"g%03d" % int(k[1:]): v for k, v in GROUP.items()}


def day(ts):
    try:
        t = int(ts)
    except (TypeError, ValueError):
        return None
    if t <= 0:
        return None
    return time_gmtime(t)


def time_gmtime(t):
    import datetime
    return datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")


def main(path):
    rows = []
    bad = collections.Counter()
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 11:
                bad["short"] += 1
                continue
            month, jobid, state, user, acct, part, elig, start, end, gpus, hosts = p[:11]
            hs = [x for x in hosts.split(",") if x]
            uniq = sorted(set(hs))
            if not uniq:
                bad["no_host"] += 1
                continue
            if len(uniq) > 1:
                bad["multi_host"] += 1
            d = day(start) or day(elig)
            if d is None:
                bad["no_time"] += 1
                continue
            rows.append({"day": d, "user": user, "acct": acct, "part": part,
                         "host": uniq[0], "group": GROUP.get(uniq[0], "?"),
                         "ngpu": int(gpus or 0), "state": state,
                         "nmach": len(uniq)})
    print("== 载入 ==")
    print("   gpu-ish jobs=%d  丢弃=%s" % (len(rows), dict(bad)))
    print("   天数=%d  范围 %s -> %s" % (len({r["day"] for r in rows}),
                                        min(r["day"] for r in rows),
                                        max(r["day"] for r in rows)))
    hosts = collections.Counter(r["host"] for r in rows)
    print("   主机数=%d  top=%s" % (len(hosts), hosts.most_common(6)))
    grp = collections.Counter(r["group"] for r in rows)
    print("   节点档位分布=%s  (卡型: 论文 Table 1 三组全是 4xV100)" % dict(grp))
    parts = collections.Counter(r["part"] for r in rows)
    print("   分区分布=%s" % dict(parts))
    ng = collections.Counter(r["ngpu"] for r in rows)
    print("   请求卡数分布=%s" % dict(sorted(ng.items())))
    for ent, name in (("user", "user"), ("acct", "account")):
        c = collections.Counter(r[ent] for r in rows)
        v = sorted(c.values())
        print("   %s 数=%d  每实体作业数 中位=%d p90=%d 最大=%d" % (
            name, len(c), v[len(v) // 2], v[int(len(v) * .9)], v[-1]))

    # 迁移率：同一 user 落在 >1 台主机
    uh = collections.defaultdict(set)
    for r in rows:
        uh[r["user"]].add(r["host"])
    multi = sum(1 for v in uh.values() if len(v) > 1)
    print("   跨主机用户=%d/%d=%.3f" % (multi, len(uh), multi / max(1, len(uh))))

    by_day = collections.defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)

    print("== 按天分布 ==")
    cnt = [len(by_day[d]) for d in sorted(by_day)]
    cnt.sort()
    print("   每天 GPU 作业数 中位=%d p90=%d 最大=%d 最小=%d" % (
        cnt[len(cnt) // 2], cnt[int(len(cnt) * .9)], cnt[-1], cnt[0]))

    print("== (i)+(ii) 无泄漏按天扫描（实体=user，结果=主机）==")
    print("   %-11s %6s %7s %8s %8s %8s" % ("cut_day", "n", "cohort", "MEM", "PRIOR", "onlyMEM"))
    days = sorted(by_day)
    worst = 10 ** 9
    tot = 0
    for i, d in enumerate(days):
        if i < 7 or i % 5:
            continue
        hist = collections.defaultdict(collections.Counter)
        htot = collections.Counter()
        for dd in days[:i]:
            for r in by_day[dd]:
                hist[r["user"]][r["host"]] += 1
                htot[r["host"]] += 1
        if not htot:
            continue
        prior = htot.most_common(1)[0][0]
        n = coh = mem = pri = only = 0
        for r in by_day[d]:
            h = hist[r["user"]]
            if sum(h.values()) < 2:
                continue
            coh += 1
            m = h.most_common(1)[0][0]
            n += 1
            mem += (m == r["host"])
            pri += (prior == r["host"])
            only += (m == r["host"] and prior != r["host"])
        if n:
            print("   %-11s %6d %7d %8.4f %8.4f %8d" % (
                d, n, coh, mem / n, pri / n, only))
            worst = min(worst, coh)
            tot += coh
    print("   (cohort 合计=%d, 最小切点=%s)" % (tot, worst))

    print("== 同一件事，结果换成节点档位(3 类) ==")
    for i, d in enumerate(days):
        if i < 10 or i % 10:
            continue
        hist = collections.defaultdict(collections.Counter)
        htot = collections.Counter()
        for dd in days[:i]:
            for r in by_day[dd]:
                if r["group"] == "?":
                    continue
                hist[r["user"]][r["group"]] += 1
                htot[r["group"]] += 1
        if not htot:
            continue
        prior = htot.most_common(1)[0][0]
        n = coh = mem = pri = only = 0
        for r in by_day[d]:
            if r["group"] == "?" or sum(hist[r["user"]].values()) < 2:
                continue
            coh += 1
            m = hist[r["user"]].most_common(1)[0][0]
            n += 1
            mem += (m == r["group"])
            pri += (prior == r["group"])
            only += (m == r["group"] and prior != r["group"])
        print("   %-11s n=%-6d cohort=%-6d MEM=%.4f PRIOR=%.4f onlyMEM=%d" % (
            d, n, coh, mem / max(1, n), pri / max(1, n), only))

    print("== 实体换成 account，结果=主机，按周合并（缓解日粒度不稳）==")
    week = collections.defaultdict(list)
    for r in rows:
        import datetime
        y, m, dd = (int(x) for x in r["day"].split("-"))
        iso = datetime.date(y, m, dd).isocalendar()
        week["%d-W%02d" % (iso[0], iso[1])].append(r)
    wk = sorted(week)
    for i, w in enumerate(wk):
        if i < 2:
            continue
        hist = collections.defaultdict(collections.Counter)
        htot = collections.Counter()
        for ww in wk[:i]:
            for r in week[ww]:
                hist[r["acct"]][r["host"]] += 1
                htot[r["host"]] += 1
        prior = htot.most_common(1)[0][0]
        n = coh = mem = pri = only = 0
        for r in week[w]:
            h = hist[r["acct"]]
            if sum(h.values()) < 2:
                continue
            coh += 1
            m = h.most_common(1)[0][0]
            n += 1
            mem += (m == r["host"])
            pri += (prior == r["host"])
            only += (m == r["host"] and prior != r["host"])
        print("   %-9s n=%-6d cohort=%-6d MEM=%.4f PRIOR=%.4f onlyMEM=%d" % (
            w, n, coh, mem / max(1, n), pri / max(1, n), only))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gpu.tsv")
