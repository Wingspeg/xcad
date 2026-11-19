#!/usr/bin/env python3
"""Byte-range-slice members out of Alibaba cluster-trace-v2026-maas archives
without downloading them (10.5 GB for instance_daily.zip; each shard is 33-194 MB).

Every member of instance_daily.zip is STORED (probe_maas_zip.py: 61 members,
methods=[0], sum raw == sum compressed == archive size), so an HTTP Range request
over the member's data window returns a byte-exact, independently valid parquet
file.  This is the same technique xcad_code/scripts/ali_trace_slice.py used for v2026.

Window:  st = local_header_offset + 30 + nlen + elen
         en = st + raw_size - 1
Anchoring on local_header_offset alone drops the 59-byte local header
(30 + nlen 29 + elen 0) and truncates the footer -> "Parquet magic bytes not found".

Usage:  python fetch_maas_shards.py <zip_url> <archive_size> <out_dir> <idx> [<idx> ...]
"""
import os
import struct
import subprocess
import sys

PROXY = os.environ.get("XCAD_PROXY", "")          # optional egress proxy, e.g. http://host:port
PROXY_ARGS = ["-x", PROXY] if PROXY else []


def rng(url, start, end):
    r = subprocess.run(
        ["curl", "-sS", "-m", "1800", *PROXY_ARGS,
         "-H", f"Range: bytes={start}-{end}", url, "--output", "-"],
        capture_output=True)
    want = end - start + 1
    if r.returncode != 0 or len(r.stdout) != want:
        raise SystemExit(f"range {start}-{end} failed rc={r.returncode} "
                         f"got={len(r.stdout)} want={want}")
    return r.stdout


def central_dir(url, size):
    tail = rng(url, size - 1_500_000, size - 1)
    i = tail.rfind(b"PK\x06\x06")                       # ZIP64 EOCD
    if i >= 0:
        _, _, _, _, _, _, ent_t, cd_size, cd_off = struct.unpack_from("<QHHIIQQQQ", tail, i + 4)
    else:
        i = tail.rfind(b"PK\x05\x06")                   # plain EOCD
        _, _, _, _, _, ent_t, cd_size, cd_off = struct.unpack_from("<HHHHHHII", tail, i + 4)
    return rng(url, cd_off, cd_off + cd_size - 1), ent_t


def members(cd):
    out, o = [], 0
    while o + 46 <= len(cd) and cd[o:o + 4] == b"PK\x01\x02":
        method = struct.unpack_from("<H", cd, o + 10)[0]
        csize, usize = struct.unpack_from("<II", cd, o + 20)
        nlen, elen, clen = struct.unpack_from("<HHH", cd, o + 28)
        lho = struct.unpack_from("<I", cd, o + 42)[0]
        name = cd[o + 46:o + 46 + nlen].decode()
        body = cd[o + 46 + nlen: o + 46 + nlen + elen]
        got, e = [], 0
        while e + 4 <= len(body):
            sid, sz = struct.unpack_from("<HH", body, e)
            if sid == 0x0001:
                got = list(struct.unpack_from("<" + "Q" * (sz // 8), body, e + 4))
                break
            e += 4 + sz
        k = 0
        for slot in ("usize", "csize", "lho"):
            cur = {"usize": usize, "csize": csize, "lho": lho}[slot]
            if cur == 0xFFFFFFFF and k < len(got):
                v = got[k]
                if slot == "usize":
                    usize = v
                elif slot == "csize":
                    csize = v
                else:
                    lho = v
                k += 1
        out.append(dict(name=name, method=method, csize=csize, usize=usize, lho=lho))
        o += 46 + nlen + elen + clen
    return out


def main(url, size, out_dir, idxs):
    os.makedirs(out_dir, exist_ok=True)
    ms = members(central_dir(url, int(size))[0])
    print(f"{len(ms)} members in {os.path.basename(url)}")
    for i in idxs:
        m = ms[i]
        assert m["method"] == 0, f"{m['name']} is not STORED; must decompress instead"
        lh = rng(url, m["lho"], m["lho"] + 61)
        assert lh[:4] == b"PK\x03\x04"
        n2, e2 = struct.unpack_from("<HH", lh, 26)
        st = m["lho"] + 30 + n2 + e2
        en = st + m["usize"] - 1
        blob = rng(url, st, en)
        assert len(blob) == m["usize"]
        assert blob[:4] == b"PAR1" and blob[-4:] == b"PAR1", "footer check failed"
        dst = os.path.join(out_dir, os.path.basename(m["name"]))
        with open(dst, "wb") as f:
            f.write(blob)
        print(f"  {m['name']} -> {dst}  {len(blob):,} B  PAR1 ok")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], [int(a) for a in sys.argv[4:]])
