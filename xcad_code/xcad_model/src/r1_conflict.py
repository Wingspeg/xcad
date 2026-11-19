# -*- coding: utf-8 -*-
"""
r1_conflict.py —— R1-1 反驳用的离线分层: 冲突单元格 + 构造性 cohort 上的逐边归因.

要堵的话
    "The baseline's collapse ... may reflect entity turnover and graph sparsity as
     much as the profile-versus-recurrence dichotomy."
他的预测: 塌缩的臂只是因为没有历史可用; 一旦有历史(复发 cohort), generic TKG 就会回升。
我们的预测: 即使在他条件最强的一格里, generic TKG 仍然等于类别先验, 而 xCAD 的增益来自属性通道。

所以这里把 test 的 r1 边切成互斥的格子, 每格同时给四条读数:
    模型 accuracy / compat 头 accuracy / 纯记忆查表 accuracy (=该格的历史众数, 即"历史可用"的定义性上界)
    以及模型在"历史说 A、属性说 B"的冲突边里**跟着谁**。
最后一条是关键: 它把"用了历史"从一个抽象说法变成计数。

输入: main.py --dump-preds 落盘的 .pt(每条边一行), 加上 xcad_v2 的 train/valid/test.txt(算历史众数)。
用法:
    python r1_conflict.py --data xcad_v2 \
        joint=.../joint_w20.pt compat_only=.../compat_w20.pt het_only=.../het_w20.pt base=.../base_w20.pt
"""

import argparse
import collections
import os
import sys

import numpy as np
import torch

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
GPU_NAMES = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]
R1_REL = 1


def history_mode(data_dir, gpu_base):
    """head -> (历史 r1 tail 众数类, 该 head 在 history 里的 r1 边数, 不同快照数)。"""
    cnt = collections.defaultdict(collections.Counter)
    snaps = collections.defaultdict(set)
    for split in ("train.txt", "valid.txt"):
        rows = np.loadtxt(os.path.join(data_dir, split), dtype=np.int64)
        rows = rows[rows[:, 1] == R1_REL]
        for h, t, s in zip(rows[:, 0], rows[:, 2], rows[:, 3]):
            k = int(t) - gpu_base
            if 0 <= k < 6:
                cnt[int(h)][k] += 1
            snaps[int(h)].add(int(s))
    mode, n_edge = {}, {}
    for h, c in cnt.items():
        mode[h] = c.most_common(1)[0][0]
        n_edge[h] = sum(c.values())
    span = {h: len(s) for h, s in snaps.items()}
    return mode, n_edge, span


def load_dump(path):
    d = torch.load(path, map_location="cpu")
    heads = d["heads"].numpy()
    tails = d["tails"].numpy()
    gb = int(d["gpu_base"])
    gold = tails - gb
    out = {"heads": heads, "gold": gold, "n": len(heads)}
    # 模型给出的 GPU 类: 优先用段内 top1(无约束), 段外记 -1
    if "model_top1" in d:
        cls = d["model_top1"].numpy() - gb
        out["model"] = np.where((cls >= 0) & (cls < 6), cls, -1)
        out["seg_rank"] = d["seg_rank"].numpy()
    if "compat_pred" in d:
        out["compat"] = d["compat_pred"].numpy()
    return out


def acc(pred, gold, mask=None):
    if mask is None:
        mask = np.ones(len(gold), dtype=bool)
    m = mask & (gold >= 0) & (gold < 6)
    if m.sum() == 0:
        return float("nan"), 0
    return float((pred[m] == gold[m]).mean()), int(m.sum())


def report(name, a, mode_map, n_edge, span_map):
    heads, gold = a["heads"], a["gold"]
    hmode = np.array([mode_map.get(int(h), -1) for h in heads])
    hn = np.array([n_edge.get(int(h), 0) for h in heads])
    hspan = np.array([span_map.get(int(h), 0) for h in heads])
    rec = hn >= 1
    model, compat = a.get("model"), a.get("compat")

    print(f"\n===================== arm: {name}  (n={a['n']}) =====================")
    hdr = f"{'格子':>22s} {'边数':>7s} {'模型acc':>9s} {'属性acc':>9s} {'记忆acc':>8s} " \
          f"{'先验acc':>8s} {'跟历史':>7s} {'跟属性':>7s} {'两边都不':>8s}"
    print(hdr)

    def row(label, mask):
        n = int(mask.sum())
        if n == 0:
            print(f"{label:>22s} {0:7d} {'-':>9s} {'-':>9s} {'-':>8s} {'-':>8s} "
                  f"{'-':>7s} {'-':>7s} {'-':>8s}")
            return
        m_acc, _ = acc(model, gold, mask) if model is not None else (float("nan"), 0)
        c_acc, _ = acc(compat, gold, mask) if compat is not None else (float("nan"), 0)
        mem_acc, _ = acc(hmode, gold, mask)
        prior = np.full_like(gold, 1)          # 恒答 T4(全体 r1 众数类)
        p_acc, _ = acc(prior, gold, mask)
        if model is not None and compat is not None:
            both = mask & (hmode >= 0) & (hmode != compat)
            nb = int(both.sum())
            if nb:
                f_hist = float((model[both] == hmode[both]).mean()) * 100
                f_prof = float((model[both] == compat[both]).mean()) * 100
                f_nei = float(((model[both] != hmode[both]) & (model[both] != compat[both])).mean()) * 100
                print(f"{label:>22s} {n:7d} {m_acc:9.4f} {c_acc:9.4f} {mem_acc:8.4f} "
                      f"{p_acc:8.4f} {f_hist:6.2f}% {f_prof:6.2f}% {f_nei:7.2f}%")
                return
        print(f"{label:>22s} {n:7d} {m_acc:9.4f} {c_acc:9.4f} {mem_acc:8.4f} {p_acc:8.4f} "
              f"{'-':>7s} {'-':>7s} {'-':>8s}")

    all_mask = np.ones(a["n"], dtype=bool)
    row("全部 test", all_mask)
    row("复发 h>=1 (FULL&r1)", rec)
    row("全新 (不可复现)", ~rec)
    for k in (2, 3):
        row(f"cohort n_hist>={k}", hn >= k)
    row("span>=2 (真跨时间)", hspan >= 2)
    if model is not None and compat is not None:
        conflict = rec & (hmode >= 0) & (hmode != compat)
        agree = rec & (hmode >= 0) & (hmode == compat)
        row("复发+历史/属性冲突", conflict)
        row("复发+历史/属性一致", agree)
        # 冲突格里单独看"历史对/属性错"这一子格: 他条件最强的一格
        strongest = conflict & (hmode == gold) & (compat != gold)
        print(f"  [最强格] 历史对而属性错 n={int(strongest.sum())}: "
              f"模型答对 {float((model[strongest] == gold[strongest]).mean())*100 if strongest.sum() else float('nan'):.2f}%  "
              f"跟历史 {float((model[strongest] == hmode[strongest]).mean())*100 if strongest.sum() else float('nan'):.2f}%  "
              f"跟属性 {float((model[strongest] == compat[strongest]).mean())*100 if strongest.sum() else float('nan'):.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", help="形如 name=path.pt")
    ap.add_argument("--data", default="xcad_v2")
    args = ap.parse_args()

    d = os.path.join(DATA_ROOT, args.data)
    probe = torch.load(args.dumps[0].split("=", 1)[-1], map_location="cpu")
    gb = int(probe["gpu_base"])
    mode_map, n_edge, span_map = history_mode(d, gb)
    print(f"[data] {d}  gpu_base={gb}  history head 带 r1 边: {len(mode_map)}")
    for spec in args.dumps:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name, path = os.path.basename(spec), spec
        report(name, load_dump(path), mode_map, n_edge, span_map)
    return 0


if __name__ == "__main__":
    sys.exit(main())
