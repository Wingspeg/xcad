#!/usr/bin/env python3
"""summarize_cw_2026_11_18.py — 类别加权臂读数汇总 + 按预注册判线判定。

判线在开卡前已写死（见 cw_2026_11_18.sh 头部与本文件 JUDGE 注释），此处只做机械套用：
  C   控制组 = class-weight none 三种子 macro5 均值 = (0.5085+0.5069+0.5069)/3 = 0.507433
  L1  不回退线：某档三种子 macro5 均值 >= C - 0.02
  L2  稀有类起效线：P100 或 V100M32 的 Hits@1 在该档 >=2 个种子上 >= 0.05
  L3  有效线（可写进论文的改进）：某档三种子 macro5 均值 >= C + 0.02
  L4  天花板参照：RF 同测试边集 macro 0.582 / P100 0.1214 / V100M32 0.2308
  STOP 停止线：两档 L2 全不过 => "类别加权救不了 P100/V100M32" 钉死，本轴不再开臂
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cw_log_parse import class_weight_lines, parse  # noqa: E402

ROOT = os.path.expanduser("~/xcad/outputs/reports/")
OUT = os.path.join(ROOT, "class_weight_2026_11_18")
SEEDS = [42, 123, 2024]
MODES = ["sqrt", "inv"]
CLS5 = ["T4", "MISC", "P100", "V100", "V100M32"]
NOISE = 0.02
RARE_LINE = 0.05
RF = dict(macro5=0.582, P100=0.1214, V100M32=0.2308)

CTRL = [("night_2026_11_18/p0_regression.log", 42),
        ("baselines_idspace/v2_joint_s123.log", 123),
        ("baselines_idspace/v2_joint_s2024.log", 2024)]

exp = json.load(open(os.path.join(OUT, "expected_weights.json")))
C = float(np.mean([exp["control"][str(s)]["macro5"] for s in SEEDS]))

print("==== 类别加权臂汇总（final-test 块，尺子 = 主表 11,836 条）====")
print(f"控制组 class-weight=none 三种子 macro5 = "
      f"{[exp['control'][str(s)]['macro5'] for s in SEEDS]}  均值 C = {C:.6f}\n")

rows, probs = [], []
ctrl_rows = []
for rel, s in CTRL:
    d = parse(os.path.join(ROOT, rel))
    if not d["ok"]:
        probs.append(f"控制组 seed{s}: {d['why']}")
        continue
    ctrl_rows.append(dict(mode="none", seed=s, **{k: d[k] for k in
                     ("macro5", "r1_h1", "r1_mrr", "inseg_h1", "agree", "best_epoch")},
                     **{c: d["per"][c] for c in CLS5}))

for mode in MODES:
    for s in SEEDS:
        rel = f"class_weight_2026_11_18/{mode}_s{s}.log"
        path = os.path.join(ROOT, rel)
        d = parse(path)
        if not d["ok"]:
            probs.append(f"{mode} seed{s}: {d['why']} ({rel})")
            continue
        # 权重行必须等于预注册值
        cwl = "\n".join(class_weight_lines(path))
        want_w = np.array(exp["expected"][mode]["w"])
        m_w = re.search(r"w\(all 6\)\s*=\s*\[([0-9. ]+)\]", cwl)
        got = [float(x) for x in m_w.group(1).split()] if m_w else []
        if "mode=" + mode not in cwl:
            probs.append(f"{mode} seed{s}: 日志里没有 [class-weight] mode={mode} 行（flag 没生效？）")
        if len(got) != 6 or np.max(np.abs(np.array(got) - want_w)) > 1.5e-4:
            probs.append(f"{mode} seed{s}: [class-weight] w={got} 与预注册 "
                         f"{want_w.round(4).tolist()} 不符")
        rows.append(dict(mode=mode, seed=s, **{k: d[k] for k in
                    ("macro5", "r1_h1", "r1_mrr", "inseg_h1", "agree", "best_epoch")},
                    **{c: d["per"][c] for c in CLS5}))

hdr = f"{'mode':6s} {'seed':5s} {'macro5':>7s} " + " ".join(f"{c:>8s}" for c in CLS5) + \
      f" {'r1H@1':>7s} {'r1MRR':>7s} {'段内H@1':>8s} {'agree':>7s} {'best':>5s}"
for grp, name in ((ctrl_rows, "控制组"), (rows, "加权臂")):
    print(f"── {name} " + "─" * max(0, len(hdr) - len(name) - 4))
    print(hdr)
    for r in grp:
        print(f"{r['mode']:6s} {r['seed']:<5d} {r['macro5']:7.4f} " +
              " ".join(f"{r[c]:8.4f}" for c in CLS5) +
              f" {r['r1_h1']:7.4f} {r['r1_mrr']:7.4f} {r['inseg_h1']:8.4f} "
              f"{r['agree']:7.4f} {str(r['best_epoch']):>5s}")

print("\n── 每档三种子均值 / 跨度 ──────────────────────────────────────────")
print(f"{'mode':6s} {'macro5 均值':>11s} {'跨度':>7s} " +
      " ".join(f"{c+'均值':>10s}" for c in CLS5) + f" {'r1H@1':>8s}")
verdicts = {}
for mode in MODES:
    g = [r for r in rows if r["mode"] == mode]
    if len(g) != 3:
        probs.append(f"{mode}: 只有 {len(g)}/3 个种子有读数，判线不完整")
    if not g:
        continue
    mac = np.array([r["macro5"] for r in g], dtype=float)
    m = dict(macro_mean=float(mac.mean()), macro_span=float(mac.max() - mac.min()))
    for c in CLS5:
        m[c] = float(np.mean([r[c] for r in g]))
        m[c + "_ge"] = int(sum(1 for r in g if r[c] >= RARE_LINE))
    m["r1_h1"] = float(np.mean([r["r1_h1"] for r in g]))
    verdicts[mode] = m
    print(f"{mode:6s} {m['macro_mean']:11.4f} {m['macro_span']:7.4f} " +
          " ".join(f"{m[c]:10.4f}" for c in CLS5) + f" {m['r1_h1']:8.4f}")

print("\n── 预注册判线套用 ────────────────────────────────────────────────")
n_pass = 0
for mode, m in verdicts.items():
    L1 = m["macro_mean"] >= C - NOISE
    L2 = max(m["P100_ge"], m["V100M32_ge"]) >= 2
    L3 = m["macro_mean"] >= C + NOISE
    L4m = m["macro_mean"] >= RF["macro5"] - 0.03
    print(f"  [{mode}] L1 不回退(>= {C - NOISE:.4f}) : {'PASS' if L1 else 'FAIL'} "
          f"读数 {m['macro_mean']:.4f}（Δ={m['macro_mean'] - C:+.4f}）")
    print(f"           L2 稀有类起效(>=2 seed 达 {RARE_LINE}) : {'PASS' if L2 else 'FAIL'} "
          f"P100 {m['P100_ge']}/3 均值 {m['P100']:.4f} | V100M32 {m['V100M32_ge']}/3 均值 {m['V100M32']:.4f}")
    print(f"           L3 有效线(>= {C + NOISE:.4f})        : {'PASS' if L3 else 'FAIL'}")
    print(f"           L4 逼近 RF 上界 {RF['macro5']}        : {'是' if L4m else '否'}；"
          f"稀有类是否达画像可分度({RF['P100']}/{RF['V100M32']}) : "
          f"{'是' if m['P100'] >= RF['P100'] or m['V100M32'] >= RF['V100M32'] else '否'}")
    n_pass += int(L2)
STOP = len(verdicts) == len(MODES) and n_pass == 0
print(f"\n  停止线（两档 L2 全不过 => 类加权救不了稀有类，本轴不再开臂）: {'触发' if STOP else '未触发'}")

if probs:
    print("\n!! 问题清单")
    for p in probs:
        print("   " + p)

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "readings.json"), "w") as f:
    json.dump(dict(control_mean=C, rows=ctrl_rows + rows, verdicts=verdicts,
                   lines=dict(noise=NOISE, rare=RARE_LINE, rf=RF),
                   stop_line=STOP, problems=probs), f, indent=2, ensure_ascii=False)
np.savetxt(os.path.join(OUT, "readings.csv"),
           [[r["mode"], r["seed"], r["macro5"]] + [r[c] for c in CLS5] +
            [r["r1_h1"], r["r1_mrr"], r["inseg_h1"], r["agree"], r["best_epoch"]]
            for r in ctrl_rows + rows],
           fmt="%s", delimiter=",",
           header="mode,seed,macro5," + ",".join(CLS5) + ",r1_h1,r1_mrr,inseg_h1,agree,best_epoch",
           comments="")
print(f"\n已存 -> {OUT}/readings.{{json,csv}}")
