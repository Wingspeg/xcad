#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2-6（少数类那一半）：把 ③（仅算法兼容性头）与 ④（联合）的逐类 Hits@1 按种子配对口径摊开。

为什么要这个件：正文 §5.7 原来写"完整模型相对 ③ 把两个多数类推高而把 P100 归零，
但这几处差值都不大于本文实测的运行间波动"。这句话要成立，就必须能证明
"不大于波动"到底是对哪个尺子说的——全体 r1 H@1 的运行间波动是 0.0172（1.72 点），
而 MISC 的均值差是 +2.2 点，**拿它当尺子这句话就站不住**。真正的尺子是同一配置、
同一臂自身的逐种子跨度。本脚本因此只读日志、逐种子抽逐类读数，输出配对差与同臂跨度。

门禁（读不到就报错，不猜）：
  G1  每份日志都必须能在最后一次 `Load Model name` 之后抽到 5 个类 + macro（防止误取 valid 块）；
  G2  n 必须等于已知样本数 5032/5246/552/811/195，否则说明不是 xcad_v2 那张图；
  G3  macro 必须等于 5 个类读数的算术平均（容差 5e-4），否则抽取错位。

用法：python3 probe_r26_minority_paired.py <out.json>
"""
import json
import os
import re
import statistics
import sys

N_EXPECT = {"T4": 5032, "MISC": 5246, "P100": 552, "V100": 811, "V100M32": 195}
CLASSES = list(N_EXPECT)

ARMS = {
    "3_compat_only": {
        42: "baselines_idspace/ablate_v2/a2_compat.log",
        123: "night_2026_11_18/s4_compat_s123.log",
        2024: "night_2026_11_18/s4_compat_s2024.log",
    },
    "4_joint": {
        42: "baselines_idspace/ablate_v2/a4_joint.log",
        123: "baselines_idspace/v2_joint_s123.log",
        2024: "baselines_idspace/v2_joint_s2024.log",
    },
}
ROOT = os.path.expanduser("~/xcad/outputs/reports")


def final_block(txt: str) -> str:
    i = txt.rfind("Load Model name")
    if i < 0:
        raise SystemExit("G1 FAIL: 没有 `Load Model name`，无法定位 final test 块")
    return txt[i:]


def parse(path: str) -> dict:
    txt = open(path, encoding="utf-8", errors="replace").read()
    blk = final_block(txt)
    hits, ns = {}, {}
    for m in re.finditer(r"(T4|MISC|P100|V100|V100M32):\s+Hits@1=([\d.]+)\s+\(n=(\d+)\)", blk):
        hits[m.group(1)] = float(m.group(2))
        ns[m.group(1)] = int(m.group(3))
    missing = [c for c in CLASSES if c not in hits]
    if missing:
        raise SystemExit(f"G1 FAIL: {path} 缺类 {missing}")
    for c in CLASSES:
        if ns[c] != N_EXPECT[c]:
            raise SystemExit(f"G2 FAIL: {path} 的 {c} n={ns[c]} != {N_EXPECT[c]}")
    mm = re.search(r"Per-GPU Hits@1 Macro \(unweighted\) = ([\d.]+)", blk)
    if not mm:
        raise SystemExit(f"G1 FAIL: {path} 没有 macro 行")
    macro = float(mm.group(1))
    calc = sum(hits[c] for c in CLASSES) / len(CLASSES)
    if abs(calc - macro) > 5e-4:
        raise SystemExit(f"G3 FAIL: {path} macro={macro} 与逐类均值 {calc:.4f} 不符")
    return {"per_class": hits, "macro": macro,
            "r1_h1": float(re.search(r"r1_suits\s+MRR=[\d.]+\s+H@1=([\d.]+)", blk).group(1))
                     if re.search(r"r1_suits\s+MRR=[\d.]+\s+H@1=([\d.]+)", blk) else None}


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "r26_minority_paired.json"
    data = {}
    for arm, per_seed in ARMS.items():
        data[arm] = {}
        for seed, rel in per_seed.items():
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p):
                raise SystemExit(f"缺日志：{p}")
            data[arm][str(seed)] = parse(p)
            print(f"[read] {arm} seed={seed} macro={data[arm][str(seed)]['macro']:.4f} "
                  + " ".join(f"{c}={data[arm][str(seed)]['per_class'][c]:.4f}" for c in CLASSES))

    rows = []
    for c in CLASSES + ["macro"]:
        def get(arm, seed):
            d = data[arm][str(seed)]
            return d["macro"] if c == "macro" else d["per_class"][c]
        a3 = [get("3_compat_only", s) for s in (42, 123, 2024)]
        a4 = [get("4_joint", s) for s in (42, 123, 2024)]
        diffs = [round((y - x) * 100, 2) for x, y in zip(a3, a4)]
        span3 = round((max(a3) - min(a3)) * 100, 2)
        span4 = round((max(a4) - min(a4)) * 100, 2)
        rows.append({
            "class": c,
            "arm3_per_seed": [round(v * 100, 2) for v in a3],
            "arm4_per_seed": [round(v * 100, 2) for v in a4],
            "arm3_mean": round(sum(a3) / 3 * 100, 2),
            "arm4_mean": round(sum(a4) / 3 * 100, 2),
            "paired_diff_per_seed": diffs,
            "mean_diff": round(sum(a4) / 3 * 100 - sum(a3) / 3 * 100, 2),
            "arm3_own_seed_span": span3,
            "arm4_own_seed_span": span4,
            "max_abs_paired_diff": round(max(abs(d) for d in diffs), 2),
        })
        r = rows[-1]
        r["resolvable_vs_own_span"] = bool(
            abs(r["mean_diff"]) > max(r["arm3_own_seed_span"], r["arm4_own_seed_span"])
        )

    summary = {
        "provenance": {k: {str(s): v for s, v in d.items()} for k, d in ARMS.items()},
        "gates": "G1 final-block-only / G2 class n / G3 macro==mean(per-class)",
        "rows": rows,
        "noise_reference": {
            "run_to_run_r1_H1_spread_points": 1.72,
            "note": "正文里'不大于运行间波动'若以全体 r1 H@1 的 1.72 点为尺子，MISC 的 +2.23 点越线；"
                    "以同臂自身逐种子跨度为尺子（MISC ③ = 6.88 点）才成立 ⇒ 正文引用处必须改用后者。",
        },
    }
    json.dump(summary, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n| 类 | ③ 逐种子(42/123/2024) | ④ 逐种子 | ③均值 | ④均值 | 均值差 | 配对差(逐种子) | ③自身跨度 | ④自身跨度 | 超过自身跨度? |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| {} | {} | {} | {} | {} | {:+.2f} | {} | {:.2f} | {:.2f} | {} |".format(
            r["class"], r["arm3_per_seed"], r["arm4_per_seed"], r["arm3_mean"], r["arm4_mean"],
            r["mean_diff"], r["paired_diff_per_seed"], r["arm3_own_seed_span"], r["arm4_own_seed_span"],
            "是" if r["resolvable_vs_own_span"] else "否"))
    print(f"\n[写出] {out_path}")


if __name__ == "__main__":
    main()
