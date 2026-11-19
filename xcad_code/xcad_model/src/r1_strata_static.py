#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r1_strata_static.py — R1-1 分层实验的**静态侧**(CPU, 不用 GPU, 不用 checkpoint)。

与 r1_strata.py 用同一套无标签分层器, 把 diagnose_sklearn 的三个非时序预测器
(众数先验 / 算法查表 / 13 维资源画像 RandomForest)按 head 是否复发拆开:

    复现缺失对神经模型是"embedding 没被训练过", 对静态模型根本不是障碍 ——
    node_feat_algo.npy 给每个算法实体都备好了特征, 新 head 照样能预测。
    所以这张表回答的是: **静态通道的性能是否与实体复现无关**。
    若 RF 在 novel 层 ≈ recurrent 层, 而神经模型的增益也全在 novel 层,
    那"关系演化"就没有可归因的残差了。

协议与主实验对齐: 拟合用 train+valid 的 r1 边(= RE-GCN 的 history), 评测 test.txt,
分层器只看 history 侧出现过的 head id, 不使用 test 真值。

用法:
    cd xcad_model/src && ../../.venv/bin/python r1_strata_static.py            # xcad(旧编号)
    ... r1_strata_static.py --data xcad_v2                                     # 新编号
"""
import argparse
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..")))
from src.r1_strata import (GPU_CLASS_NAMES, N_GPU_CLASSES, head_sets,  # noqa: E402
                           prior_tail)


def load_r1(path):
    """test/valid/train.txt -> r1_suits 边 [M,3] = (head, tail, tau)。"""
    raw = np.loadtxt(path, dtype=np.int64)
    if raw.ndim == 1:                      # 极端情况: 只有一行
        raw = raw.reshape(1, -1)
    r1 = raw[raw[:, 1] == 1]
    return np.stack([r1[:, 0], r1[:, 2], r1[:, 3]], axis=1)


def load_snapshots(path):
    """整份 .txt 按 τ 切成快照列表(与 split_by_time 同义, 只用来做分层器)。"""
    raw = np.loadtxt(path, dtype=np.int64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return [raw[raw[:, 3] == t] for t in sorted(set(raw[:, 3].tolist()))]


def metrics(y_true, y_pred, gpu_base):
    acc = float((y_true == y_pred).mean())
    hits, occ = [], []
    per = {}
    for k in range(1, N_GPU_CLASSES + 1):
        c = gpu_base + k
        m = (y_true == c)
        n = int(m.sum())
        if n == 0:
            hits.append(0.0)
            per[GPU_CLASS_NAMES[k]] = (0.0, 0)
        else:
            a = float((y_true[m] == y_pred[m]).mean())
            hits.append(a)
            occ.append(a)
            per[GPU_CLASS_NAMES[k]] = (a, n)
    return {"n": int(len(y_true)), "acc": acc,
            "macro5": sum(hits) / N_GPU_CLASSES,
            "macro_occ": (sum(occ) / len(occ)) if occ else float("nan"),
            "n_occ": len(occ), "per_class": per}


def show(name, m):
    cls = "  ".join(f"{k}={v[0]*100:5.1f}/{v[1]}" for k, v in m["per_class"].items())
    print(f"    {name:22s} n={m['n']:6d}  acc={m['acc']*100:6.2f}%  "
          f"macro5={m['macro5']*100:6.2f}%  macro_occ={m['macro_occ']*100:6.2f}%"
          f"(c={m['n_occ']})   {cls}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="xcad", help="xcad(旧局部编号) / xcad_v2(全局四段编号)")
    ap.add_argument("--hist-len", type=int, default=20, help="WIN 分层器的历史快照数(=test_history_len)")
    ap.add_argument("--n-feats", type=int, default=13,
                    help="只用 node_feat_algo.npy 的前 N 维。12 = 去掉 instance_count —— "
                         "该列是全 trace 聚合(含预测时刻之后才创建的实例), 属于已知越界(§4.1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = os.path.join(_HERE, "..", "data", args.data)
    feat_full = np.load(os.path.join(data_dir, "node_feat_algo.npy")).astype(np.float64)
    assert feat_full.shape[1] >= args.n_feats, f"特征只有 {feat_full.shape[1]} 维"
    feat = feat_full[:, :args.n_feats]
    N_ALGO = feat.shape[0]
    print(f"\n[features] {N_ALGO} 个算法实体 × 取前 {args.n_feats}/{feat_full.shape[1]} 维"
          f"{'  (已去掉 instance_count)' if args.n_feats < feat_full.shape[1] else ''}")

    tr = load_r1(os.path.join(data_dir, "train.txt"))
    va = load_r1(os.path.join(data_dir, "valid.txt"))
    te = load_r1(os.path.join(data_dir, "test.txt"))
    hist = np.concatenate([tr, va], axis=0)          # RE-GCN 测试期的 history
    gpu_base = 124885 if hist[:, 1].max() >= 124885 else 0

    # 只保留 head 落在算法段、tail 落在 GPU 段 5 类之内的边(与 diagnose_sklearn 同口径)
    def keep(a):
        m = (a[:, 0] < N_ALGO) & np.isin(a[:, 1], [gpu_base + k for k in range(1, 6)])
        return a[m]

    hist, te = keep(hist), keep(te)
    print(f"\n{'='*78}\n  {args.data}: history r1 边={len(hist)}  test r1 边={len(te)}  "
          f"gpu_base={gpu_base}\n{'='*78}")

    # ── 分层器(无标签) ──
    # 快照保留完整 4 列: head_sets 用 col0/col1, prior_tail 用 col2(tail)
    hist_snaps = (load_snapshots(os.path.join(data_dir, "train.txt")) +
                  load_snapshots(os.path.join(data_dir, "valid.txt")))
    full_r1, full_any = head_sets(hist_snaps)
    win_r1, win_any = head_sets(hist_snaps[-args.hist_len:])
    p = prior_tail(hist_snaps)

    sc = StandardScaler().fit(feat[hist[:, 0]])
    Xtr = sc.transform(feat[hist[:, 0]])
    Xte = sc.transform(feat[te[:, 0]])
    ytr, yte, hte = hist[:, 1], te[:, 1], te[:, 0]

    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                random_state=args.seed, n_jobs=-1).fit(Xtr, ytr)

    # 算法查表: 该 head 在 history 里最常配的 GPU 类; 没出现过则退回先验
    cnt = {}
    for a, t in zip(hist[:, 0].tolist(), ytr.tolist()):
        d = cnt.setdefault(a, {})
        d[t] = d.get(t, 0) + 1
    lookup_map = {a: max(d.items(), key=lambda kv: kv[1])[0] for a, d in cnt.items()}
    preds = {
        "众数先验(history)": np.full(len(yte), p),
        "算法查表(众数 GPU)": np.array([lookup_map.get(a, p) for a in hte.tolist()]),
        "RandomForest(13维)": rf.predict(Xte),
    }

    strata = [("全部 test 边", np.ones(len(hte), bool))]
    for label, s in (("FULL&r1", full_r1), ("FULL&any", full_any),
                     ("WIN&r1", win_r1), ("WIN&any", win_any)):
        is_rec = np.isin(hte, list(s))
        strata += [(f"{label} 复发", is_rec), (f"{label} 全新", ~is_rec)]
        print(f"  [分层器 {label}] 复发 {is_rec.sum()}/{len(hte)} = "
              f"{is_rec.sum()/len(hte)*100:.2f}%")

    for name, mask in strata:
        if mask.sum() == 0:
            print(f"\n  ── {name}: n=0")
            continue
        print(f"\n  ── {name}  (n={mask.sum()})")
        for pname, y_pred in preds.items():
            show(pname, metrics(yte[mask], y_pred[mask], gpu_base))


if __name__ == "__main__":
    main()
