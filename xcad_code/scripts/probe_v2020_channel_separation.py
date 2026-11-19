#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_v2020_channel_separation.py — 把 maas 那张"两通道分离"表在 v2020 上按同一口径量一遍。

为什么要它
----------
`docs/reviews/2026-11-16_second_dataset_screening.md` §17.4 与
`docs/reviews/2026-11-17_maas_main_table.md` §6 都写了一行
"v2020 两条通道 top-1 同点率 = 0.999747"。这个数字**不是**记忆通道与属性通道的同点率：
0.999747 在 R1-1 文档里逐处定义的都是
「compat 属性头直读 argmax 与模型 GPU 段内 top-1 的同点率（11,833/11,836）」。
记忆 vs 属性在 v2020 上的同点率**从来没有被测过**，
本节 0 GPU 把它补上，顺便把跨数据集对照表里那个空格填成实测而不是引用。

三件事，一件一件来
------------------
1) **回归**：静态口径下的三条闭式通道必须逐位复现 §2.4 那张表
   （恒答众数 0.4251/0.3871/0.4287、算法查表 0.4780/0.9990/0.4287、RF 0.8898/0.8055/0.8978）。
   复现不了就停 —— 说明尺子换了，下面的读数不可用。
2) **新读数**：同一批行上，两两通道的 top-1 同点率 + onlyX/onlyY（只有 X 答对的边数）
   + 复合上界 oracle（任一通道答对）。这是判据 (ii)「两通道是否分离」在 v2020 上的**直接**测量。
3) **时序口径**：把记忆通道换成"每个 test 快照只用严格更早的 train+valid 边"
   （§26 在 maas 上做的那次修口径），并给出 §25 那课要求的**可用率差**：
   拟合期行里"该 head 有严格更早的 r1 边"的比例 vs 测试期同样比例。
   这个差决定 (D) 里那条 head 先验通道在 v2020 上是恒开、恒关还是按行分开。

口径纪律：rel=1 (r1_suits)、tail ∈ {124886..124890} 共 5 个卡型 id、
head 必须落在算法段（<102,610）—— 与 `r1_strata_static.py` / §2.4 完全一致；
分层器只看 train+valid 的 head id，不用 test 真值；
候选只有 5 类，所以"top-1"= argmax，不需要在 124,891 个实体上排名。
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..", "xcad_model")))
from src.r1_strata import GPU_CLASS_NAMES, head_sets  # noqa: E402

DATA = os.path.join(_HERE, "..", "xcad_model", "data", "xcad_v2")
R1 = 1
GPU_IDS = [124886, 124887, 124888, 124889, 124890]   # gpu_base=124885 + 1..5
CLS = {g: i for i, g in enumerate(GPU_IDS)}
NAME = {GPU_CLASS_NAMES[k]: k for k in range(1, 6)}
N_ALGO = 102610

# §2.4 / §6.1 已发表的读数（回归对照，不是本脚本算出来的）
REF = {"prior": (0.4251, 0.3871, 0.4287),
       "mem": (0.4780, 0.9990, 0.4287),
       "rf": (0.8898, 0.8055, 0.8978),
       "model": {"all": 0.884843, "rec": 0.697947, "novel": 0.902525}}


def load_r1(split):
    a = np.loadtxt(os.path.join(DATA, f"{split}.txt"), dtype=np.int64)
    r1 = a[(a[:, 1] == R1) & (a[:, 0] < N_ALGO) & np.isin(a[:, 2], GPU_IDS)]
    return r1[:, 0], r1[:, 2], r1[:, 3]          # head, tail, tau


def load_snaps(split):
    a = np.loadtxt(os.path.join(DATA, f"{split}.txt"), dtype=np.int64)
    return [a[a[:, 3] == t] for t in sorted(set(a[:, 3].tolist()))]


def mode_of(heads, tails, tau, lo=None):
    """返回 (每 head 的众数 class, 全局众数 class)。lo=None 表示不设时间下限。"""
    keep = np.ones(len(tau), bool) if lo is None else (tau < lo)
    cnt = {}
    glob = np.zeros(len(GPU_IDS), np.int64)
    for h, t in zip(heads[keep].tolist(), tails[keep].tolist()):
        d = cnt.setdefault(h, np.zeros(len(GPU_IDS), np.int64))
        d[CLS[t]] += 1
        glob[CLS[t]] += 1
    prior = int(glob.argmax())
    # 并列按"全局类频次"高者胜（与查表口径一致的确定性破平，不引入随机）
    lut = {}
    for h, d in cnt.items():
        lut[h] = int(np.lexsort((-glob, -d))[0])
    return lut, prior, glob


def acc(pred, gold):
    return float((pred == gold).mean())


def stratum_line(tag, key, idx, ch, model_ref):
    """打印一个分层里所有通道的精度 + 两两同点率 + onlyX/onlyY + oracle。"""
    g = ch["gold"][idx]
    names = ["prior", "mem", "rf", "mem_ts"]
    out = {"tag": tag, "n": int(len(idx)), "model_ref": model_ref}
    print(f"\n  ── {tag}  (n={len(idx)})")
    for n in names:
        p = ch[n][idx]
        out[n] = round(acc(p, g), 4)
        print(f"    {n:6s} acc={out[n]:.4f}")
    print(f"    model  acc={model_ref:.4f}   (已发表读数, joint_xcad_v2 日志)")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa, pb = ch[a][idx], ch[b][idx]
            agree = float((pa == pb).mean())
            ga, gb = g == pa, g == pb
            out[f"agree_{a}__{b}"] = round(agree, 4)
            out[f"only_{a}__vs_{b}"] = int((ga & ~gb).sum())
            out[f"only_{b}__vs_{a}"] = int((~ga & gb).sum())
            print(f"    agree[{a} vs {b}] = {agree:.6f}   "
                  f"only_{a}={int((ga & ~gb).sum()):5d}  only_{b}={int((~ga & gb).sum()):5d}")
    orac = float(((g == ch["mem"][idx]) | (g == ch["rf"][idx])).mean())
    out["oracle_mem_rf"] = round(orac, 4)
    print(f"    ORACLE(mem ∪ rf) = {orac:.4f}   (模型 {model_ref:.4f} ⇒ 空位 {orac - model_ref:+.4f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-feats", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "outputs", "reports"))
    args = ap.parse_args()

    htr, ttr, tautr = load_r1("train")
    hva, tva, tauva = load_r1("valid")
    hte, tte, tauta = load_r1("test")
    H = np.concatenate([htr, hva]); T = np.concatenate([ttr, tva]); U = np.concatenate([tautr, tauva])
    print(f"[edges] history r1={len(H)}  test r1={len(hte)}  "
          f"train ts {tautr.min()}..{tautr.max()}  valid {tauva.min()}..{tauva.max()}  test {tauta.min()}..{tauta.max()}")

    # ── 分层器（无标签，与 §2.4 同一套：train+valid 全量快照里出现过 r1 head）──
    full_r1, _ = head_sets(load_snaps("train") + load_snaps("valid"))
    is_rec = np.isin(hte, list(full_r1))
    idx = {"all": np.arange(len(hte)), "rec": np.where(is_rec)[0], "novel": np.where(~is_rec)[0]}
    print(f"[strata] 复发 {is_rec.sum()}/{len(hte)} = {is_rec.mean()*100:.2f}%")

    # ── 通道 1/2：静态口径 ──
    lut, prior, glob = mode_of(H, T, U)
    gold_idx = np.array([CLS[t] for t in tte.tolist()], dtype=np.int64)
    ch = {"gold": gold_idx,
          "prior": np.full(len(tte), prior),
          "mem": np.array([lut.get(h, prior) for h in hte.tolist()])}
    print(f"[static] 全局类计数 {dict(zip([GPU_CLASS_NAMES[k] for k in range(1,6)], glob.tolist()))}")

    # ── 通道 4：时序口径（严格更早），每个快照一张表；test 与 fit 共用同一批表 ──
    luts_by_ts, priors_by_ts = {}, {}
    for ts in sorted(set(U.tolist()) | set(tauta.tolist())):
        luts_by_ts[ts], priors_by_ts[ts], _ = mode_of(H, T, U, lo=ts)

    mem_ts = np.zeros(len(tte), np.int64)
    avail_test = np.zeros(len(tte), bool)
    for ts in np.unique(tauta):
        m = tauta == ts
        mem_ts[m] = [luts_by_ts[ts].get(h, priors_by_ts[ts]) for h in hte[m].tolist()]
        avail_test[m] = np.isin(hte[m], list(luts_by_ts[ts]))
    # 拟合期可用率（§25 那课要的泄漏探针）：train+valid 行里有多少行有严格更早的同 head r1 边
    avail_fit = np.zeros(len(H), bool)
    for ts in np.unique(U):
        m = U == ts
        avail_fit[m] = np.isin(H[m], list(luts_by_ts[ts]))
    print(f"[avail] 有严格更早历史的行比例  fit={avail_fit.mean():.4f}  test={avail_test.mean():.4f}"
          f"   差={avail_fit.mean()-avail_test.mean():+.4f}")

    # ── 捷径可用性（§25 的"79.8% 的 argmax 就是自己这行 gold"那一格，v2020 版）──
    # 静态表对每一条 train+valid 的 r1 行都非空（表就是由这些行造的，含它自己）⇒ 拟合期可用率恒 1.0000；
    # 要看的是"argmax 正好等于自己这行 gold"的比例，它才是 α 能学到的捷径有多甜。
    static_fit_ok = np.array([lut.get(h, prior) == CLS[t]
                              for h, t in zip(H.tolist(), T.tolist())], bool)
    ts_ok = np.array([luts_by_ts[u].get(h, -1) == CLS[t]
                      for h, t, u in zip(H.tolist(), T.tolist(), U.tolist())], bool)
    print(f"[shortcut] 拟合期：静态表 argmax==本行 gold 的比例 = {static_fit_ok.mean():.4f}"
          f"   时序口径（同率，含不可用行记错）= {ts_ok.mean():.4f}"
          f"   仅在可用行上 = {ts_ok[avail_fit].mean():.4f}")

    # ── 通道 3：属性通道 = RF（与 r1_strata_static 同超参/同特征/同 seed）──
    feat = np.load(os.path.join(DATA, "node_feat_algo.npy")).astype(np.float64)[:, :args.n_feats]
    sc = StandardScaler().fit(feat[H])
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                               random_state=args.seed, n_jobs=-1).fit(sc.transform(feat[H]), T)
    ch["rf"] = np.array([CLS[t] for t in rf.predict(sc.transform(feat[hte]))])
    ch["mem_ts"] = mem_ts
    for k in ("prior", "mem", "mem_ts"):
        ch[k] = ch[k].astype(np.int64)

    # ── 回归对照 ──
    # 容差 0.0015 的理由：§2.4 那行的 RF 读数（88.98/80.55/89.78）是 11-12 那次跑的，
    # 今天重跑 `r1_strata_static.py --data xcad_v2` 得到 88.97/80.45/89.77
    # （sklearn 版本换了之后森林的并列打破规则变了，复发层差 10 条边）。
    # 本脚本与**当前**脚本逐位一致，所以对照对象改用当前实现，容差只用于挡住真正的口径漂移。
    print("\n" + "=" * 78 + "\n  回归对照（必须复现 §2.4 那张表，否则下面读数不可用）\n" + "=" * 78)
    ok = True
    for key, name in (("prior", "恒答众数"), ("mem", "算法查表"), ("rf", "RandomForest")):
        got = [round(acc(ch[key][idx[s]], ch["gold"][idx[s]]), 4) for s in ("all", "rec", "novel")]
        want = list(REF[key])
        hit = all(abs(a - b) < 1.5e-3 for a, b in zip(got, want))
        ok &= hit
        print(f"  {name:12s} got={got} ref(§2.4)={want}  {'OK' if hit else 'MISMATCH'}")
    if not ok:
        print("\n[STOP] 尺子与已发表表不一致，本脚本输出不可用于结论。")
        return 2

    rows = []
    for s, tag in (("all", "全部 test 边"), ("rec", "FULL&r1 复发"), ("novel", "FULL&r1 全新")):
        rows.append(stratum_line(tag, s, idx[s], ch, REF["model"][s]))

    # 分层里的 mem/mem_ts 可用率
    print("\n  [可用率] 全部/复发/全新 三档上 test 行有严格更早历史的比例 = "
          + " / ".join(f"{avail_test[idx[s]].mean():.4f}" for s in ("all", "rec", "novel")))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "probe_v2020_channel_separation.json")
    with open(path, "w") as f:
        json.dump({"n_test_r1": int(len(hte)), "n_recurrent": int(is_rec.sum()),
                   "avail_fit": round(float(avail_fit.mean()), 6),
                   "avail_test": round(float(avail_test.mean()), 6),
                   "strata": rows, "regression_ok": bool(ok)}, f, ensure_ascii=False, indent=1)
    print(f"\n[written] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
