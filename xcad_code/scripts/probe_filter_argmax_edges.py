#!/usr/bin/env python
"""只读探针：核对"过滤前段内 argmax"与"过滤后 model_top1"不一致的那几条边，
在两种口径下各自是否答对。

背景：手稿 §5.5 写了一句"过滤动作只在 3/11,836 条边上改变了 argmax，而这些边在两种
口径下都答错"。前半句有 hybrid.log 的 G-SCORE 读数支持，后半句当时是**推断**而非实测
（两条 argmax 的总体命中率逐位相同，逻辑上允许"一条由对变错、一条由错变对"相互抵消）。
本脚本把那几条边的 (head, gold, 过滤前 argmax, 过滤后 argmax) 打出来，把推断换成实测。

编号口径与 hybrid_profile_graph.py 一致，全部用"段内偏移"：0 = CPU 列，1..5 = 五个 GPU 类。
gold = tail - gpu_base ∈ 1..5；pre = raw_seg.argmax(1) ∈ 0..5；post = model_top1 - gpu_base。

用法（服务器）：
  PY=~/xcad/xcad_code/.venv/bin/python
  $PY probe_filter_argmax_edges.py \
      --scores-dir ~/xcad/outputs/reports/b7_mlp_hybrid_2026_11_19/scores
"""
import argparse
import os

import numpy as np
import torch

CLS = {0: "CPU", 1: "T4", 2: "MISC", 3: "P100", 4: "V100", 5: "V100M32"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scores-dir",
        default=os.path.expanduser(
            "~/xcad/outputs/reports/b7_mlp_hybrid_2026_11_19/scores"
        ),
    )
    ap.add_argument("--seeds", default="42,123,2024")
    args = ap.parse_args()

    for s in [int(x) for x in args.seeds.split(",")]:
        p = os.path.join(os.path.expanduser(args.scores_dir), f"preds_s{s}.pkl")
        d = torch.load(p, map_location="cpu")
        base = int(d["gpu_base"])
        print(f"== seed {s}  keys={sorted(d.keys())}  gpu_base={base}")
        pre = d["raw_seg"].numpy().argmax(1)          # 过滤前段内 argmax（0..5）
        post = d["model_top1"].numpy() - base         # 过滤后存档 top-1（1..5）
        gold = d["tails"].numpy() - base              # r1 真值（1..5）
        heads = d["heads"].numpy()
        assert gold.min() >= 1 and gold.max() <= 5, (gold.min(), gold.max())

        ok_pre = pre == gold
        ok_post = post == gold
        diff = np.flatnonzero(pre != post)
        print(f"   n={len(gold)}  不一致边数={len(diff)}  "
              f"acc(过滤前)={ok_pre.mean():.6f}  acc(过滤后)={ok_post.mean():.6f}")
        for i in diff:
            print(f"   edge {i:>6}  head={heads[i]:>7}  gold={CLS[gold[i]]:<7} "
                  f"pre={CLS[pre[i]]:<7}({'对' if ok_pre[i] else '错'})  "
                  f"post={CLS[post[i]]:<7}({'对' if ok_post[i] else '错'})")
        if len(diff):
            both_wrong = int(((~ok_pre) & (~ok_post))[diff].sum())
            flip = int((ok_pre[diff] != ok_post[diff]).sum())
            both_right = int((ok_pre[diff] & ok_post[diff]).sum())
            print(f"   不一致边中：两口径都错 = {both_wrong}/{len(diff)}，"
                  f"一对一错（相互抵消）= {flip}，都对 = {both_right}")


if __name__ == "__main__":
    main()
