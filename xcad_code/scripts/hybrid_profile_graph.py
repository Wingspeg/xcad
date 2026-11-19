#!/usr/bin/env python3
"""
B7 / R2-4 的 hybrid：同一把尺子下把【图通道分数】与【静态画像通道分数】用可学习的方式混合。

判线/口径全部预注册在 docs/reviews/2026-11-19_b7_mlp_hybrid.md（§1 口径契约 / §3 形状 / §4 门禁 / §5 判线），
本脚本只是那份文档的翻译，条件一字未改。四道门禁不过 ⇒ 直接 exit(2)，不出表。

输入（都在服务器上）：
  --scores-dir   seg_s{seed}.test.pt / seg_s{seed}.valid.pt   (--dump-seg-scores 落的过滤前 GPU 段分数)
                 preds_s42.pkl                                  (给 G-SCORE 的第二问：与 b1_preds.pkl 对得上)
  --b1-pkl       night_2026_11_19/b1_preds.pkl                  (已发表那一次 test 的逐边落盘)
  数据/特征      ../data/xcad_v2/{train,valid,test}.txt + node_feat_algo.npy

用法：
  python hybrid_profile_graph.py --seeds 42 123 2024 --json-out readings.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

GPU_TYPES = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]      # 段内偏移 0..5
CLS = GPU_TYPES[1:]                                                # r1 真值只可能是这 5 类
FEAT_NAMES = ["avg_plan_cpu", "std_plan_cpu", "min_plan_cpu", "max_plan_cpu",
              "avg_plan_mem", "std_plan_mem", "min_plan_mem", "max_plan_mem",
              "avg_plan_gpu", "std_plan_gpu", "min_plan_gpu", "max_plan_gpu",
              "instance_count"]

# 已发表读数（G-SCORE 用；来源 docs/reviews/2026-11-17_idspace_main_table.md 的 P0 门禁）
PUB_R1_H1 = 0.884843
PUB_R1_MACRO = 0.5085
NOISE = 0.02          # v2020 run-to-run 噪声 0.0172 ⇒ 主张线取 0.02


# ───────────────────────────────── 数据读取 ─────────────────────────────────
def read_r1(txt_path):
    """返回 (heads, tails, taus)：该文件里 rel==1 的正向 r1_suits 边，按文件读序。"""
    raw = np.loadtxt(txt_path, dtype=np.int64)
    raw = raw.reshape(-1, raw.shape[-1])
    m = raw[:, 1] == 1
    r = raw[m]
    return r[:, 0], r[:, 2], r[:, 3]


def load_graph_channel(scores_dir, seed, split):
    """--dump-seg-scores 落的文件：raw_seg=[n,6] 过滤前分数, heads/gold 逐边身份。"""
    p = os.path.join(scores_dir, f"seg_s{seed}.{split}.pt")
    d = torch.load(p, map_location="cpu")
    return dict(path=p, seg=d["raw_seg"].numpy().astype(np.float64),
                heads=d["heads"].numpy(), gold=d["gold"].numpy(),
                gpu_base=int(d["gpu_base"]), n=int(d["n_rows"]))


# ───────────────────────────────── 指标 ─────────────────────────────────
def metrics(y, pred, gpu_base):
    """macro5 / 总 acc / 逐类 recall+precision / 预测边际。类下标 = tail - gpu_base - 1 ∈ 0..4。"""
    K = 5
    acc = float((pred == y).mean())
    rec, prec = [], []
    for k in range(K):
        m = (y == k)
        p = (pred == k)
        rec.append(float((pred[m] == k).mean()) if m.any() else 0.0)
        prec.append(float((y[p] == k).mean()) if p.any() else 0.0)
    marg = [int((pred == k).sum()) for k in range(K)]
    n = [int((y == k).sum()) for k in range(K)]
    return dict(acc=acc, macro=float(np.mean(rec)), recall=rec, precision=prec,
                marginal=marg, n=n)


def log_softmax(a):
    a = a - a.max(1, keepdims=True)
    return a - np.log(np.exp(a).sum(1, keepdims=True))


# ───────────────────────────────── 主流程 ─────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data/xcad_v2")
    ap.add_argument("--feat-path", default="../data/xcad_v2/node_feat_algo.npy")
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--b1-pkl", default=None, help="用于 G-SCORE 第二问的已发表逐边落盘")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    ap.add_argument("--class-weights", nargs="+", default=["none", "balanced"])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_dir))
    feat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.feat_path))
    feat = np.load(feat_path).astype(np.float64)
    print(f"[feat] {feat_path}  shape={feat.shape}")

    tr_h, tr_t, tr_tau = read_r1(os.path.join(data_dir, "train.txt"))
    va_h, va_t, va_tau = read_r1(os.path.join(data_dir, "valid.txt"))
    te_h, te_t, te_tau = read_r1(os.path.join(data_dir, "test.txt"))
    print(f"[r1] train={len(tr_h)}  valid={len(va_h)}  test={len(te_h)}")
    # 画像通道要求每个 r1 head 都有 13 维画像、每个 tail 都落在 5 个卡型列内，
    # 否则两个通道的行集会不一致 —— 直接算出来，不假设。
    for _nm, _h, _t in (("train", tr_h, tr_t), ("valid", va_h, va_t), ("test", te_h, te_t)):
        assert int(_h.max()) < feat.shape[0], f"{_nm}: head id 超出画像矩阵 {feat.shape}"
        _off = _t - 124885 - 1          # 124885 = gpu_base(xcad_v2)；与 .pt 里的 gpu_base 由后面一条断言对齐
        assert _off.min() >= 0 and _off.max() <= 4, (
            f"{_nm}: 有 r1 tail 不在 5 个卡型段内 (min={_off.min()}, max={_off.max()})")

    gate_fail = []

    # ════ 通道导出与图通道读数自检（与 seed 无关的部分先做一遍）════
    print("\n" + "=" * 84)
    print("  门禁 G-SCORE / G-ALIGN")
    print("=" * 84)
    graph = {}
    for s in args.seeds:
        for split in ("test", "valid"):
            g = load_graph_channel(args.scores_dir, s, split)
            graph[(s, split)] = g
            # G-ALIGN：与 txt 读序逐位相同
            ref = (te_h, te_t) if split == "test" else (va_h, va_t)
            align = (len(g["heads"]) == len(ref[0])
                     and bool((g["heads"] == ref[0]).all())
                     and bool((g["gold"] == ref[1]).all()))
            # 图通道读数：6 列 argmax（应与已发表逐位相同）/ 去掉 CPU 的 5 列 argmax（混合器真正看到的）
            # 注意两套编号：6 列 argmax 给的是【段内偏移 0..5】(0=CPU)，
            # 而 y 是【类下标 0..4】= 偏移-1。写错这一格会让门禁假失败(本轮踩过一次)。
            y = g["gold"] - g["gpu_base"] - 1              # 0..4
            assert y.min() >= 0 and y.max() <= 4, f"gold 不在 5 类段内: {g['path']}"
            a6 = g["seg"].argmax(1) - 1                    # 换算到类下标；-1 = 选了 CPU 列
            a5 = g["seg"][:, 1:].argmax(1)                 # 0..4
            m6 = metrics(y, a6, g["gpu_base"])
            m5 = metrics(y, a5, g["gpu_base"])
            acc6, mac6 = m6["acc"], m6["macro"]
            cpu_win = int((a6 == -1).sum())
            same = int((a6 != a5).sum())
            line = (f"  s{s} {split:<5} n={g['n']:>6}  6列argmax acc={acc6:.6f} macro={mac6:.6f}  "
                    f"5列argmax acc={m5['acc']:.6f} macro={m5['macro']:.6f}  "
                    f"CPU列胜出={cpu_win}  6/5列不一致={same}  G-ALIGN={'PASS' if align else 'FAIL'}")
            print(line)
            if split == "test" and s == args.seeds[0]:
                if abs(acc6 - PUB_R1_H1) > 5e-7 or abs(round(mac6, 4) - PUB_R1_MACRO) > 5e-7:
                    gate_fail.append(f"G-SCORE: 过滤前分数 argmax 未复现已发表读数 "
                                     f"({acc6:.6f}/{mac6:.6f} vs {PUB_R1_H1}/{PUB_R1_MACRO})")
            if not align:
                gate_fail.append(f"G-ALIGN: s{s}/{split} 导出行序与 {split}.txt 的 r1 读序不同")
            g["y"] = y

    # G-SCORE 第二问：与 b1_preds.pkl 的逐边读数相同（证明这次跑的就是已发表那一次模型）
    if args.b1_pkl:
        b1 = torch.load(args.b1_pkl, map_location="cpu")
        cur = torch.load(os.path.join(args.scores_dir, "preds_s42.pkl"), map_location="cpu")
        ok = all(
            k in cur and cur[k].shape == b1[k].shape and bool((cur[k] == b1[k]).all())
            for k in ("heads", "tails", "filt_rank", "model_top1", "seg_rank"))
        print(f"  [G-SCORE/b1] preds_s42.pkl vs {os.path.basename(args.b1_pkl)}: "
              f"heads/tails/filt_rank/model_top1/seg_rank 逐边相同 = {'PASS' if ok else 'FAIL'}")
        if not ok:
            gate_fail.append("G-SCORE: 本轮 preds_s42.pkl 与已发表 b1_preds.pkl 不是同一批逐边读数")
        if "raw_seg" in cur:
            a = cur["raw_seg"].numpy().argmax(1)
            t1 = cur["model_top1"].numpy() - int(cur["gpu_base"])
            print(f"  [G-SCORE] 过滤前 6 列 argmax 与存档 model_top1 不一致边数 = "
                  f"{int((a != t1).sum())}/{len(a)}   （过滤动作只影响这些边）")

    # G-NOLEAK：mixer 拟合集(valid) 与测试集边身份交集必须 = 0
    ev = set(zip(va_h.tolist(), [1] * len(va_h), va_t.tolist(), va_tau.tolist()))
    et = set(zip(te_h.tolist(), [1] * len(te_h), te_t.tolist(), te_tau.tolist()))
    inter = len(ev & et)
    print(f"  [G-NOLEAK] valid∩test 的 r1 边身份 (h,r,t,tau) 交集 = {inter} (必须 0)   "
          f"head 交集 = {len(set(va_h) & set(te_h))}/{len(set(te_h))} 个 test head（信息性, 非门禁）")
    if inter:
        gate_fail.append(f"G-NOLEAK: mixer 拟合集与测试集有 {inter} 条同身份边")

    if gate_fail:
        print("\n" + "!" * 84)
        for f in gate_fail:
            print("  !! " + f)
        print("门禁未过 ⇒ 按 §4 整块作废，不出表。")
        sys.exit(2)
    print("\n[门禁] G-SCORE / G-ALIGN / G-NOLEAK 全过。\n")

    # ════ 混合实验 ════
    all_rows = []
    for cw in args.class_weights:
        for s in args.seeds:
            gt, gv = graph[(s, "test")], graph[(s, "valid")]
            # ── 画像通道：RF 只用 train 拟合（§1：mixer 在 valid 上看到的是 out-of-sample 概率）──
            Xtr = feat[tr_h]
            Xva, Xte = feat[va_h], feat[te_h]
            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xva_s, Xte_s = (scaler.transform(X) for X in (Xtr, Xva, Xte))
            ytr = tr_t - gt["gpu_base"] - 1
            yva = va_t - gt["gpu_base"] - 1
            clf_par = None if cw == "none" else "balanced"
            rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=s,
                                        n_jobs=-1, class_weight=clf_par)
            rf.fit(Xtr_s, ytr)
            assert sorted(rf.classes_.tolist()) == list(range(5)), f"RF 类不齐: {rf.classes_}"
            prob_va, prob_te = (rf.predict_proba(X) for X in (Xva_s, Xte_s))
            logP_va, logP_te = (np.log(np.clip(p, 1e-9, 1.0)) for p in (prob_va, prob_te))
            # ── 图通道：过滤前 5 列（去 CPU）按行 log-softmax ──
            logG_va = log_softmax(gv["seg"][:, 1:])
            logG_te = log_softmax(gt["seg"][:, 1:])
            y_te, y_va = gt["y"], yva

            variants = {
                "hybrid": (np.hstack([logP_va, logG_va]), np.hstack([logP_te, logG_te])),
                "H_P(画像单通道)": (logP_va, logP_te),
                "H_G(图单通道)": (logG_va, logG_te),
            }
            res = {}
            coefs = {}
            for name, (A_va, A_te) in variants.items():
                mix = LogisticRegression(C=1.0, max_iter=2000, random_state=s)
                mix.fit(A_va, y_va)
                pred = mix.predict(A_te)
                res[name] = metrics(y_te, pred, gt["gpu_base"])
                coefs[name] = mix.coef_
            # 参照：单通道各自的"无混合器"直接 argmax（画像 = RF top-1，图 = 段内 top-1）
            res["RF_top1(画像, train-only)"] = metrics(
                y_te, prob_te.argmax(1), gt["gpu_base"])
            res["G_top1(图, 段内过滤前)"] = metrics(
                y_te, logG_te.argmax(1), gt["gpu_base"])
            # 参照：已发表口径的 RF(train+valid) —— 只为说明 hybrid 的画像通道弱在哪
            rf2 = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=s,
                                         n_jobs=-1, class_weight=clf_par)
            Xtrv_s = scaler.transform(np.vstack([Xtr, Xva]))
            rf2.fit(Xtrv_s, np.concatenate([ytr, yva]))
            res["RF_top1(画像, train+valid)"] = metrics(
                y_te, rf2.predict_proba(Xte_s).argmax(1), gt["gpu_base"])

            # ── 机制线③：两通道 top-1 一致率 + 四格 ──
            pP, pG, pH = res["RF_top1(画像, train-only)"], res["G_top1(图, 段内过滤前)"], res["hybrid"]
            predP, predG = prob_te.argmax(1), logG_te.argmax(1)
            agree = float((predP == predG).mean())
            cP, cG = (predP == y_te), (predG == y_te)
            quad = dict(both=int((cP & cG).sum()), onlyP=int((cP & ~cG).sum()),
                        onlyG=int((~cP & cG).sum()), neither=int((~cP & ~cG).sum()))
            # ── 机制线①：混合器两个特征块的权重范数比 ──
            W = np.abs(coefs["hybrid"])
            wn_p, wn_g = float(W[:, :5].sum()), float(W[:, 5:].sum())
            all_rows.append(dict(cw=cw, seed=s, n_test=int(len(y_te)),
                                 hyb_feat_scale=dict(logP_absmax=float(np.abs(logP_te).max()),
                                                     logG_absmax=float(np.abs(logG_te).max())),
                                 res=res, agree_P_G=agree, quad=quad,
                                 wnorm=dict(profile=wn_p, graph=wn_g,
                                            ratio=(wn_p / wn_g if wn_g else float("inf")))))

            # 表
            print("=" * 84)
            print(f"  hybrid　class_weight={cw}  seed={s}  test n={len(y_te)}  "
                  f"(画像通道 RF=train-only 拟合, 混合器拟合于 valid 12,600)")
            print("=" * 84)
            hdr = (f"  {'变体':<26}{'总acc':>9}{'macro5':>9}"
                   + "".join(f"{c:>8}" for c in CLS) + f"{'|':>2}")
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for name in ("hybrid", "H_P(画像单通道)", "H_G(图单通道)",
                         "RF_top1(画像, train-only)", "G_top1(图, 段内过滤前)",
                         "RF_top1(画像, train+valid)"):
                r = res[name]
                print(f"  {name:<26}{r['acc']*100:>9.2f}{r['macro']*100:>9.2f}"
                      + "".join(f"{x*100:>8.2f}" for x in r["recall"]))
            print(f"  {'预测类边际(hybrid)':<26}"
                  + "  ".join(f"{c}={n}" for c, n in zip(CLS, res['hybrid']['marginal'])))
            print(f"  {'真值边际':<26}"
                  + "  ".join(f"{c}={n}" for c, n in zip(CLS, res['hybrid']['n'])))
            print(f"  [机制①] |w| 块范数 画像={wn_p:.4f} 图={wn_g:.4f} 比={wn_p/max(wn_g,1e-12):.3f}"
                  f"   特征量级 |logP|max={np.abs(logP_te).max():.2f} |logG|max={np.abs(logG_te).max():.2f}")
            print(f"  [机制②] hybrid 相对 H_G 的逐类 recall 变化: "
                  + "  ".join(
                      f"{c}{(res['hybrid']['recall'][k] - res['H_G(图单通道)']['recall'][k]) * 100:+.2f}"
                      for k, c in enumerate(CLS)))
            print(f"  [机制③] 两通道 top-1 一致率={agree:.4f}   四格 都对={quad['both']} "
                  f"只画像对={quad['onlyP']} 只图对={quad['onlyG']} 都错={quad['neither']}"
                  f"   (共 {len(y_te)})")
            print(f"  [判线] H − max(H_P,H_G) = "
                  f"{(res['hybrid']['macro']-max(res['H_P(画像单通道)']['macro'], res['H_G(图单通道)']['macro']))*100:+.2f} 点"
                  f"   (噪声线 {NOISE*100:.2f})")
            print()

    # ── 跨种子汇总 + J1/J2/J3 ──
    print("=" * 84)
    print("  跨种子汇总（macro5, %）")
    print("=" * 84)
    summary = {}
    for cw in args.class_weights:
        rows = [r for r in all_rows if r["cw"] == cw]
        print(f"\n  [口径 class_weight={cw}]  n_seed={len(rows)}")
        for name in ("hybrid", "H_P(画像单通道)", "H_G(图单通道)",
                     "RF_top1(画像, train-only)", "G_top1(图, 段内过滤前)",
                     "RF_top1(画像, train+valid)"):
            v = [r["res"][name]["macro"] * 100 for r in rows]
            a = [r["res"][name]["acc"] * 100 for r in rows]
            print(f"    {name:<26} macro={np.mean(v):6.2f} ± {np.std(v, ddof=1) if len(v)>1 else 0:5.2f}  "
                  f"(逐种子 {' '.join(f'{x:.2f}' for x in v)})   acc={np.mean(a):6.2f}")
        d = [(r["res"]["hybrid"]["macro"]
              - max(r["res"]["H_P(画像单通道)"]["macro"], r["res"]["H_G(图单通道)"]["macro"])) * 100
             for r in rows]
        mean_d = float(np.mean(d))
        pos = sum(1 for x in d if x > 0)
        verdict = ("J1 互补性成立" if (mean_d >= NOISE * 100 and pos >= 2)
                   else "J3 混合有害" if mean_d <= -NOISE * 100
                   else "J2 互补性不成立")
        print(f"    Δ = H − max(H_P,H_G): 逐种子 {' '.join(f'{x:+.2f}' for x in d)}  "
              f"均值 {mean_d:+.2f}  正号 {pos}/{len(d)}  ⇒ 【{verdict}】")
        summary[cw] = dict(delta=d, mean_delta=mean_d, n_positive=pos, verdict=verdict)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(dict(rows=all_rows, summary=summary,
                           gates=dict(score="PASS", align="PASS", noleak="PASS")), f,
                      ensure_ascii=False, indent=1, default=float)
        print(f"\n[json] {args.json_out}")


if __name__ == "__main__":
    main()
