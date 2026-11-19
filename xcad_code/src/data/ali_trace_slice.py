#!/usr/bin/env python3
"""Byte-range slicer for Alibaba cluster-trace-gpu-v2026 archives.

The published archives are single- or multi-member ZIPs served with
``Accept-Ranges: bytes``.  ``asi_opensource_pod_hourly.zip`` is 351.8 GB with one
STORED (method 0) parquet per ``day=*/hour=*`` partition, so a partition can be
materialised locally with two ranged GETs instead of downloading the archive.

Usage
-----
    python3 ali_trace_slice.py index  asi_opensource_pod_hourly.zip
    python3 ali_trace_slice.py list   asi_opensource_pod_hourly.zip --day 0
    python3 ali_trace_slice.py get    asi_opensource_pod_hourly.zip \
                                      --member 'asi_opensource_pod_hourly/day=0/hour=00/part-000.parquet' \
                                      --out ~/xcad/datasets/v2026
"""
import argparse
import json
import os
import struct
import sys
import urllib.request

BASE = "https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-gpu-v2026/data/"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
CACHE_DIR = os.path.join(REPO_ROOT, "datasets", "v2026", "_index")


def url_of(name):
    return name if name.startswith("http") else BASE + name


def http_get(url, start=None, end=None, timeout=300):
    headers = {}
    if start is not None:
        headers["Range"] = "bytes=%d-%d" % (start, end)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def object_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def parse_cd_entry(cd, p):
    """Return (name, method, comp_size, uncomp_size, local_hdr_offset) for entry at p."""
    (sig, c_vmb, c_vneed, c_flags, c_method, c_mtime, c_mdate, c_crc, c_csize, c_osize,
     c_nlen, c_elen, c_clen, c_dstart, c_iattr, c_eattr, c_lhdr) = \
        struct.unpack_from("<IHHHHHHIIIHHHHHII", cd, p)
    assert sig == 0x02014B50, hex(sig)
    name = cd[p + 46: p + 46 + c_nlen].decode("utf-8", "replace")
    extra = cd[p + 46 + c_nlen: p + 46 + c_nlen + c_elen]
    csize, osize, lhdr = c_csize, c_osize, c_lhdr
    q = 0
    while q + 4 <= len(extra):
        hid, hsz = struct.unpack_from("<HH", extra, q)
        body = extra[q + 4: q + 4 + hsz]
        if hid == 0x0001:  # Zip64 extended information: present in declared order
            r = 0
            if osize == 0xFFFFFFFF and r + 8 <= len(body):
                osize = struct.unpack_from("<Q", body, r)[0]; r += 8
            if csize == 0xFFFFFFFF and r + 8 <= len(body):
                csize = struct.unpack_from("<Q", body, r)[0]; r += 8
            if lhdr == 0xFFFFFFFF and r + 8 <= len(body):
                lhdr = struct.unpack_from("<Q", body, r)[0]; r += 8
        q += 4 + hsz
    return name, c_method, csize, osize, lhdr, 46 + c_nlen + c_elen + c_clen


def build_index(name, tail_bytes=1 << 20, max_cd=256 << 20):
    url = url_of(name)
    size = object_size(url)
    tail = http_get(url, max(0, size - tail_bytes), size - 1)
    e = tail.rfind(b"PK\x05\x06")
    if e < 0:
        raise SystemExit("EOCD not found in trailing %d bytes" % tail_bytes)
    (_, disk, disk_cd, ent_disk, ent_total, cd_size, cd_off, _) = struct.unpack_from("<IHHHHIIH", tail, e)
    if ent_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
        loc = tail.rfind(b"PK\x06\x07")
        if loc < 0:
            raise SystemExit("Zip64 EOCD locator not found")
        (_, _, z_off, _) = struct.unpack_from("<IIQI", tail, loc)
        z = http_get(url, z_off, z_off + 55)
        (_, _, _, _, _, _, _, ent_total, cd_size, cd_off) = struct.unpack_from("<IQHHIIQQQQ", z, 0)
    if cd_size > max_cd:
        raise SystemExit("central directory %d B exceeds --max-cd" % cd_size)
    cd = http_get(url, cd_off, cd_off + cd_size - 1)
    members = {}
    p = 0
    while p + 46 <= len(cd):
        mname, method, csize, osize, lhdr, step = parse_cd_entry(cd, p)
        members[mname] = {"lhdr": lhdr, "method": method, "csize": csize, "osize": osize}
        p += step
    return {"url": url, "object_size": size, "cd_offset": cd_off, "members": members}


def cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, os.path.basename(name) + ".index.json")


def load_index(name, refresh=False):
    cp = cache_path(name)
    if not refresh and os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)
    idx = build_index(name)
    with open(cp, "w") as f:
        json.dump(idx, f)
    return idx


def member_data_start(url, lhdr, timeout=120):
    """Local headers carry their own (possibly different) name/extra lengths."""
    h = http_get(url, lhdr, lhdr + 29, timeout=timeout)
    assert len(h) == 30, "short local header read"
    (sig, _v, _fl, _m, _t, _d, _c, _cs, _os, nlen, elen) = struct.unpack_from("<IHHHHHIIIHH", h, 0)
    assert sig == 0x04034B50, hex(sig)
    return lhdr + 30 + nlen + elen


def cmd_index(args):
    idx = load_index(args.name, refresh=args.refresh)
    ms = idx["members"]
    methods = {}
    for m in ms.values():
        methods[m["method"]] = methods.get(m["method"], 0) + 1
    print("object   : %s" % idx["url"])
    print("size     : %d B (%.2f GB)" % (idx["object_size"], idx["object_size"] / 1024 ** 3))
    print("members  : %d   compress-method histogram: %s" % (len(ms), methods))
    tot = sum(m["csize"] for m in ms.values())
    print("sum csize: %d B (%.2f GB)" % (tot, tot / 1024 ** 3))
    print("index    : %s" % cache_path(args.name))


def cmd_list(args):
    idx = load_index(args.name)
    rows = []
    for n, m in idx["members"].items():
        if args.pattern and args.pattern not in n:
            continue
        rows.append((n, m["csize"]))
    rows.sort()
    for n, c in rows[: args.limit]:
        print("%-72s %13d" % (n, c))
    print("-- %d members, %.2f GB total" % (len(rows), sum(c for _, c in rows) / 1024 ** 3))


def cmd_get(args):
    idx = load_index(args.name)
    url = idx["url"]
    want = [n for n in sorted(idx["members"]) if (not args.pattern or args.pattern in n)]
    if args.days:
        keep = set()
        for d in args.days.split(","):
            keep.add("/day=%s/" % int(d))
        want = [n for n in want if any(k in n for k in keep)]
    if args.every_hour_in_day:
        # keep only hour=00 of each day present in the selection
        want = [n for n in want if "/hour=00/" in n or "/hour=" not in n]
    if args.max_members and len(want) > args.max_members:
        step = int(round(len(want) / float(args.max_members)))
        want = want[::step][: args.max_members]
    os.makedirs(args.out, exist_ok=True)
    est = sum(idx["members"][n]["csize"] for n in want)
    print("fetching %d members (%.2f GB) -> %s" % (len(want), est / 1024 ** 3, args.out))
    for i, n in enumerate(want, 1):
        m = idx["members"][n]
        tag = n.replace("/", "_").replace("=", "")
        dst = os.path.join(args.out, tag)
        if os.path.exists(dst) and os.path.getsize(dst) == m["csize"]:
            print("[%d/%d] cached %s" % (i, len(want), tag))
            continue
        d0 = member_data_start(url, m["lhdr"])
        data = http_get(url, d0, d0 + m["csize"] - 1)
        # every published v2026 partition is STORED, so the ranged bytes are the
        # parquet file verbatim; anything else would need the deflate stream.
        assert m["method"] == 0, "member %s is compressed (method %d)" % (n, m["method"])
        with open(dst, "wb") as f:
            f.write(data)
        print("[%d/%d] %-64s %10d B" % (i, len(want), tag, len(data)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "list", "get"])
    ap.add_argument("name")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--pattern", default="")
    ap.add_argument("--days", default="", help="comma separated day numbers")
    ap.add_argument("--every-hour-in-day", action="store_true")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--max-members", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(CACHE_DIR), "raw"))
    args = ap.parse_args()
    {"index": cmd_index, "list": cmd_list, "get": cmd_get}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
