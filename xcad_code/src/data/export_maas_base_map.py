#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_maas_base_map.py -- maas 图实体编号 -> base_id 段索引（给 --base-emb 用）。

为什么要这张表
--------------
S20 的字段判别力测量（同一把尺子、同一批 121,371 行）给出：在"历史里从没出现过的新
head"这一层（占测试边 84.8%），一个**不学习**的 base_id 历史直方图过滤式 H@1 = 0.2980，
而 tail_prior = 0.1651、模型 base = 0.1233。=> 信号在，只是模型没用上：它把每个
model_id 当孤立名字学 embedding，新名字没学过就退回边际分布。

修法（本轮只改这一件事）：head 表征 = dynamic_emb[id] + base_emb[base_of[id]]。
本脚本负责把 base_of[id] 造出来，只做数据映射，不做任何统计/打分。

纪律
----
* 只用 base_id（请求侧属性），不碰 gpu_type / cluster_id / site_id；
* model_id -> base_id 跨天**不是**严格函数（2.41% 的 model 历史上换过父，见
  outputs/reports/probe_maas_base_conflict.log），但 (model, day) 网格上多值只有 0.78%
  => 取历史众数 base 折叠成一个父节点，被折叠的数量写进 report，不假装它不存在；
* blank base_id 与 30 个 tail 实体一起进 pad 行（索引 n_bases），并在 mask 里置 0
  => pad 行拿到梯度=0、tail 段表征逐位不变，不会给"能打分的答案集合"加公共偏置；
* 编号以磁盘上的 entity2id.txt 为准（不重建 id 空间），并独立核对
  head 段连续、tail 段是末尾连续段、两段不相交 —— S11 撞号那类错误的自检。

Usage (server):
    ~/xcad/xcad_code/.venv/bin/python xcad_code/src/data/export_maas_base_map.py
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

PREFIXES = ("b_", "s_", "x_", "o_")


def read_graph_entities(data_dir):
    """entity2id.txt: name<TAB>id。按 id 落位返回 names[id]（不假设文件行序=id 序）。"""
    rec = []
    with open(os.path.join(data_dir, "entity2id.txt")) as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) != 2 or p[0] in ("head", "relation"):
                continue
            rec.append((int(p[1]), p[0]))
    ids = [r[0] for r in rec]
    assert sorted(ids) == list(range(len(ids))), "entity2id 不是 0..N-1 连续编号"
    names = [None] * len(ids)
    for eid, nm in rec:
        assert names[eid] is None, f"id {eid} 重复"
        names[eid] = nm
    assert len(set(names)) == len(names), "entity2id 有重名"
    return names


def read_tail_ids(data_dir):
    """边文件里出现过的 tail id（rel==0），用来定 tail 段。"""
    tails, heads = set(), set()
    for fn in ("train.txt", "valid.txt", "test.txt"):
        with open(os.path.join(data_dir, fn)) as f:
            for ln in f:
                p = ln.split()
                if len(p) < 4 or int(p[1]) != 0:
                    continue
                heads.add(int(p[0]))
                tails.add(int(p[2]))
    return heads, tails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="~/xcad/xcad_code/xcad_model/data/maas")
    ap.add_argument("--shards", default="~/xcad/datasets/maas/instance_daily/part_*.parquet")
    ap.add_argument("--out-npy", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    data_dir = os.path.expanduser(args.data_dir)
    out_npy = args.out_npy or os.path.join(data_dir, "base_of.npy")
    out_json = args.out_json or os.path.join(data_dir, "base_map_report.json")

    names = read_graph_entities(data_dir)
    n_ents = len(names)
    heads_in_edges, tails_in_edges = read_tail_ids(data_dir)
    tail_lo = max(tails_in_edges) - len(tails_in_edges) + 1
    assert min(tails_in_edges) == tail_lo, "tail 段不是末尾连续段"
    assert not (heads_in_edges & tails_in_edges), "head/tail 段相交（撞号）"
    print(f"[entities] n_ents={n_ents}  head 段=[0,{tail_lo})  tail 段=[{tail_lo},{n_ents})")

    paths = [p for p in sorted(glob.glob(os.path.expanduser(args.shards)))
             if "slim" not in os.path.basename(p)]
    assert paths, f"no shards under {args.shards}"
    raw = pd.concat([pd.read_parquet(p, columns=["model_id", "base_id"]) for p in paths],
                    ignore_index=True)
    print(f"[raw] rows={len(raw)} shards={len(paths)}")

    m = raw[["model_id", "base_id"]].dropna()
    m = m[m.model_id.astype("string").str.strip().ne("")
          & m.base_id.astype("string").str.strip().ne("")]
    nb = m.groupby("model_id").base_id.nunique()
    n_conflict = int((nb > 1).sum())
    # probe_maas_base_conflict.py 的读数：model_id->base_id 在【同一张表里跨天】不是严格函数
    # （1,935/80,257 = 2.41% 的 model 历史上换过父，最大 79 个 base），但在【(model,day) 网格】
    # 上近似单值（多值 cell 只有 0.78%），全新 head 那层多值占 1.41%。实体 embedding 每个 id
    # 只有一个向量，装不下"随快照变的父"，所以取历史众数 base、平票按字典序（可复现）。
    # 代价写死：2.41% 的 head 丢掉父节点的时间变化；本轮不动输出层，先量这一件事值多少。
    cnt = m.groupby(["model_id", "base_id"]).size().rename("n").reset_index()
    modal = (cnt.sort_values(["model_id", "n", "base_id"], ascending=[True, False, True])
                .drop_duplicates("model_id").set_index("model_id").base_id.to_dict())
    tab = modal
    print(f"[pairs] model_id 有 base 的个数={len(tab)}  distinct base={len(set(tab.values()))}"
          f"  跨期多 base 被众数折叠={n_conflict} ({n_conflict/len(nb):.4%})")

    bases_all = sorted(set(tab.values()))
    n_head = int(tail_lo)

    # 词表只取【图里真有 head 挂上】的 base：raw 里可能有只属于被丢弃行（空 model_id）的 base，
    # 留成 base_emb 的空行会白占参数、也会让下面的覆盖自检变成假通过。
    keys = [tab.get(nm) for nm in names[:n_head]]
    used_bases = sorted({k for k in keys if k is not None})
    b2i = {x: i for i, x in enumerate(used_bases)}
    n_bases = len(used_bases)
    pad = n_bases                                    # pad 行索引

    base_of = np.full(n_ents, pad, dtype=np.int64)
    hit = miss = stripped = 0
    for eid, key in enumerate(keys):
        if key is None:
            nm = names[eid]
            for p in PREFIXES:                       # 试一次"前缀是导出时加的"
                alt = tab.get(nm[len(p):]) if nm.startswith(p) else None
                if alt is not None:
                    key, stripped = alt, stripped + 1
                    break
        if key is None:
            miss += 1
        else:
            base_of[eid] = b2i[key]
            hit += 1
    print(f"[map] head={n_head}  映射到真实 base={hit} ({hit/n_head:.4f})  "
          f"落 pad={miss} ({miss/n_head:.4f})  去前缀才命中={stripped}")
    print(f"[vocab] raw distinct base={len(bases_all)}  图内 head 用到的 base={n_bases}"
          f"  被丢弃的 base={len(bases_all) - n_bases}")
    assert hit / n_head > 0.95, "head 的 base 覆盖率过低，映射口径可能有误"
    assert int(base_of[tail_lo:].max()) == pad and (base_of[tail_lo:] == pad).all(), \
        "tail 段未全部落 pad"
    # 自检：每个真实 base 至少被一个 head 用到（防止索引错位造出空父）
    used = set(base_of[:tail_lo].tolist()) - {pad}
    assert len(used) == n_bases, f"有 {n_bases - len(used)} 个 base 没有 head 挂上"

    np.save(out_npy, base_of)
    rep = {"n_ents": n_ents, "head_lo": 0, "head_hi": int(tail_lo), "n_bases": n_bases,
           "n_bases_in_raw": len(bases_all),
           "pad_index": pad, "n_heads_mapped": hit, "n_heads_padded": miss,
           "n_heads_hit_after_strip_prefix": stripped,
           "head_base_coverage": round(hit / n_head, 6),
           "n_models_multi_base_collapsed": n_conflict,
           "multi_base_share": round(n_conflict / max(len(nb), 1), 6),
           "model_id_to_base": "modal base over history (count desc, then lexicographic)",
           "sources": {"entity2id": os.path.join(data_dir, "entity2id.txt"),
                       "shards": [os.path.basename(p) for p in paths]}}
    with open(out_json, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("[write]", out_npy, out_json)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
