#!/usr/bin/env python3
"""Inspect Alibaba cluster-trace-v2026-maas instance_daily.zip WITHOUT downloading it.

ZIP64-aware: range-read the tail, parse the ZIP64 EOCD, range-read the central
directory, and report each member's method + local-header offset so we can slice
one parquet shard by byte range instead of pulling 10.5 GB.

Correct ZIP64 EOCD layout (after the 4-byte signature):
    <Q  size of zip64 EOCD record
     HH version made by / needed
     II disk number / disk with CD
     QQ entries this disk / entries total
     QQ CD size / CD offset
"""
import os
import struct
import subprocess

PROXY = os.environ.get("XCAD_PROXY", "")          # optional egress proxy, e.g. http://host:port
PROXY_ARGS = ["-x", PROXY] if PROXY else []
URL = ("https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/"
       "cluster-trace-v2026-maas/data/instance_daily.zip")
SIZE = 10_528_173_427
OUT = os.path.expanduser("~/xcad/xcad_code/outputs/reports/probe_maas_zip.log")


def rng(start, end):
    r = subprocess.run(
        ["curl", "-sS", "-m", "900", *PROXY_ARGS,
         "-H", f"Range: bytes={start}-{end}", URL, "--output", "-"],
        capture_output=True)
    want = end - start + 1
    if r.returncode != 0 or len(r.stdout) != want:
        raise SystemExit(f"range {start}-{end} failed rc={r.returncode} "
                         f"got={len(r.stdout)} want={want}")
    return r.stdout


log = open(OUT, "w")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


tail = rng(SIZE - 1_500_000, SIZE - 1)
i = tail.rfind(b"PK\x06\x06")
assert i >= 0, "no ZIP64 EOCD in the last 1.5 MB"
rec_size, v_made, v_need, disk, cd_disk, ent_d, ent_t, cd_size, cd_off = \
    struct.unpack_from("<QHHIIQQQQ", tail, i + 4)
p(f"ZIP64 EOCD: entries_this_disk={ent_d} entries_total={ent_t} "
  f"cd_size={cd_size:,} cd_offset={cd_off:,}")
assert cd_off < SIZE and cd_off + cd_size <= SIZE

cd = rng(cd_off, cd_off + cd_size - 1)
members, o = [], 0
while o + 46 <= len(cd) and cd[o:o + 4] == b"PK\x01\x02":
    method = struct.unpack_from("<H", cd, o + 10)[0]
    csize, usize = struct.unpack_from("<II", cd, o + 20)
    nlen, elen, clen = struct.unpack_from("<HHH", cd, o + 28)
    lho = struct.unpack_from("<I", cd, o + 42)[0]
    name = cd[o + 46:o + 46 + nlen].decode()
    body = cd[o + 46 + nlen: o + 46 + nlen + elen]
    # ZIP64 extended info (0x0001) carries original-size, then compressed-size,
    # then local-header-offset -- but only for the slots that read 0xFFFFFFFF
    # in the 32-bit fields. Walking it positionally is what mangled part_60.
    got = []
    e = 0
    while e + 4 <= len(body):
        sid, sz = struct.unpack_from("<HH", body, e)
        if sid == 0x0001:
            got = list(struct.unpack_from("<" + "Q" * (sz // 8), body, e + 4))
            break
        e += 4 + sz
    k = 0
    for name_ in ("usize", "csize", "lho"):
        cur = {"usize": usize, "csize": csize, "lho": lho}[name_]
        if cur == 0xFFFFFFFF and k < len(got):
            v = got[k]
            if name_ == "usize":
                usize = v
            elif name_ == "csize":
                csize = v
            else:
                lho = v
            k += 1
    members.append((name, method, csize, usize, lho))
    o += 46 + nlen + elen + clen

methods = sorted({m[1] for m in members})
p(f"members={len(members)}  methods={methods} (0=STORED 8=DEFLATE)")
p(f"sum raw={sum(m[3] for m in members):,}  "
  f"sum compressed={sum(m[2] for m in members):,}  "
  f"ratio={sum(m[2] for m in members) / max(1, sum(m[3] for m in members)):.3f}")
for m in members[:4] + members[-1:]:
    p("   %-34s method=%d  compressed=%13s  raw=%13s  local_hdr@%d"
      % (m[0], m[1], f"{m[2]:,}", f"{m[3]:,}", m[4]))

# Verify the "local header is 30 + nlen + elen bytes" assumption on one member and
# confirm the PAR1 footer sits at lho + 30 + nlen + elen + raw - 8.
name, method, csize, usize, lho = members[0]
lh = rng(lho, lho + 61)
assert lh[:4] == b"PK\x03\x04", lh[:4]
n2, e2 = struct.unpack_from("<HH", lh, 26)
ds = lho + 30 + n2 + e2
p(f"\nmember[0]={name} method={method} raw={usize:,} nlen={n2} elen={e2} data@{ds:,}")
head = rng(ds, ds + 3)
foot = rng(ds + usize - 8, ds + usize - 1)
flen = struct.unpack("<I", foot[:4])[0]
p(f"   head4={head!r}  footer4={foot[4:]!r}  PAR1={foot[4:] == b'PAR1'}  "
  f"footer_len={flen:,}")
p(f"   => slice window for a full member: bytes={ds}-{ds + usize - 1} "
  f"({usize / 1e6:.1f} MB)")
p("\nSLICING RECIPE (used by download_maas_shards.py): "
  "st = lho + 30 + nlen + elen ; en = st + raw - 1. "
  "Anchoring on lho alone truncates the last 59 bytes and kills the footer.")
log.close()
