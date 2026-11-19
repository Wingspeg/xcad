#!/usr/bin/env python3
"""render_ours_row_2026_11_18.py — 把类别加权轴的读数渲染成可直接粘进手稿的 markdown 片段。

用途：§13.4 要求"换表一次改完"，涉及表 2 ours 行、表 3 ④ 列、表 4 ④ 行、以及 §5.2 正文里
  写进句子的四五个数。这些数字必须同源、同口径、同种子集合，不能手抄。本脚本一次性算出并打印，
  改稿时逐段复制，改完再跑一遍核对（幂等）。

口径纪律（与 cw_log_parse / summarize_cw5 相同）：
  * 只读 final test 块，逐类 n 必须 = 5032/5246/552/811/195，否则拒读；
  * 表 2 的"总 MRR / 总 H@1"= MRR/Hits **(filter_ent)**，不是 raw_ent（G-CAL 逐 seed 复现主表 §3）；
  * mean ± sd 里的 sd 是**样本标准差**（ddof=1），×100 显示。

默认允许部分种子在场（只用于排版预演，输出会显式标出缺哪几臂）；
正式改稿必须加 --require-full，此时 10 臂不齐直接退出。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cw_log_parse import parse  # noqa: E402

ROOT = os.path.expanduser("~/xcad/outputs/reports/")
CW = os.path.join(ROOT, "class_weight_2026_11_18")
EXT = os.path.join(ROOT, "class_weight_ext_2026_11_18")

CLS5 = ["T4", "MISC", "P100", "V100", "V100M32"]
SEEDS_OLD = [42, 123, 2024]
SEEDS_NEW = [7, 2025]
ALL_SEEDS = SEEDS_OLD + SEEDS_NEW

SRC = {
    ("none", 42): os.path.join(ROOT, "night_2026_11_18/p0_regression.log"),
    ("none", 123): os.path.join(ROOT, "baselines_idspace/v2_joint_s123.log"),
    ("none", 2024): os.path.join(ROOT, "baselines_idspace/v2_joint_s2024.log"),
    ("inv", 42): os.path.join(CW, "inv_s42.log"),
    ("inv", 123): os.path.join(CW, "inv_s123.log"),
    ("inv", 2024): os.path.join(CW, "inv_s2024.log"),
}
for _s in SEEDS_NEW:
    SRC[("none", _s)] = os.path.join(EXT, f"none_s{_s}.log")
    SRC[("inv", _s)] = os.path.join(EXT, f"inv_s{_s}.log")

MAIN_TABLE = {42: (0.065962, 0.059156), 123: (0.066374, 0.059459), 2024: (0.065321, 0.058699)}
COLS = ["mrr_filter_ent", "h1_filter_ent", "r1_mrr", "r1_h1", "macro5"]

ap = argparse.ArgumentParser()
ap.add_argument("--require-full", action="store_true", help="10 臂不齐则退出（正式改稿用）")
args = ap.parse_args()

# ── G-CAL（永远跑）─────────────────────────────────────────────────────────
for s, (em, eh) in MAIN_TABLE.items():
    d = parse(SRC[("none", s)])
    if not (d.get("ok") and d["mrr_filter_ent"] == em and d["h1_filter_ent"] == eh):
        print(f"FATAL: G-CAL FAIL seed {s} -> {d.get('mrr_filter_ent')}/{d.get('h1_filter_ent')} "
              f"期望 {em}/{eh}")
        sys.exit(3)
print("G-CAL 主表口径复现（none 老种子 filter_ent）: PASS\n")

rows, missing = {}, []
for (mode, s), p in SRC.items():
    d = parse(p)
    if not d.get("ok"):
        missing.append(f"{mode}_s{s}")
        continue
    rows[(mode, s)] = {**d, **d["per"]}

if missing:
    print(f"!! 在场 {len(rows)}/10 臂；缺: {' '.join(missing)}")
    if args.require_full:
        print("FATAL: --require-full 且臂不齐 —— 不输出可粘贴片段。")
        sys.exit(4)
    print("   下面是【排版预演】，数字会随补种变化，不要粘进手稿。\n")

seeds = [s for s in ALL_SEEDS if ("inv", s) in rows and ("none", s) in rows]


def col(mode, key, sds=None):
    v = [100 * rows[(mode, s)][key] for s in (sds or seeds)]
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def cell(mode, key):
    mu, sd = col(mode, key)
    return f"{mu:.2f} ± {sd:.2f}"


print(f"种子集合 = {seeds}（mean ± 样本标准差，×100）\n")

print("── 表 2 ours 行 ──")
print("| **xCAD（ours）** | **" + "** | **".join(cell("inv", c) for c in COLS) + "** |")
print("| （对照：none 档）  | " + " | ".join(cell("none", c) for c in COLS) + " |")

print("\n── 表 4 ④ 行（总 MRR / r1 MRR / r1 H@1 / r1 macro / top-1 落在 GPU 候选段）──")
inseg = [100 * rows[("inv", s)]["in_seg_ratio"] for s in seeds]
print("| ④ ②＋③（xCAD 完整） | **" + "** | **".join(
    cell("inv", c) for c in ["mrr_filter_ent", "r1_mrr", "r1_h1"]) + f"** | {cell('inv','macro5')} "
    f"| {np.mean(inseg):.0f} ± {np.std(inseg, ddof=1):.0f} |")

print("\n── 表 3 ④ 列（逐类 Hits@1 ×100，均值；配对差 = inv − none）──")
for c in CLS5:
    mi, _ = col("inv", c)
    mn, _ = col("none", c)
    print(f"  {c:<9} {mi:6.1f}   (none {mn:6.1f},  Δ {mi-mn:+6.1f})")
mi, _ = col("inv", "macro5")
mn, _ = col("none", "macro5")
print(f"  {'macro':<9} {mi:6.1f}   (none {mn:6.1f},  Δ {mi-mn:+6.1f})")

print("\n── §5.2 正文要用的句子数字 ──")
dmac = mi - mn
dh1 = col("inv", "r1_h1")[0] - col("none", "r1_h1")[0]
dmrr = col("inv", "r1_mrr")[0] - col("none", "r1_mrr")[0]
dtm = col("inv", "mrr_filter_ent")[0] - col("none", "mrr_filter_ent")[0]
dth = col("inv", "h1_filter_ent")[0] - col("none", "h1_filter_ent")[0]
print(f"  macro {mi:.2f}（配对 Δ {dmac:+.2f}）  r1 H@1 Δ {dh1:+.2f}  r1 MRR Δ {dmrr:+.2f}"
      f"  总 MRR Δ {dtm:+.2f}  总 H@1 Δ {dth:+.2f}")
smu, ssd = col("inv", "macro5")
print(f"  inv macro {len(seeds)} 种子均值±sd = {smu:.2f} ± {ssd:.2f}   逐种子: "
      + " ".join(f"{100*rows[('inv',s)]['macro5']:.4f}" for s in seeds))
print("  与上界比较（口径必须成对出现）：加权 ours 属【类别平衡目标】一侧 ⇒ 参照 0.683，"
      "不是 0.582；未加权 ours 参照 0.582。")
print("  稀有类召回（inv）：" + "  ".join(
    f"{c} {col('inv', c)[0]:.2f}" for c in ["P100", "V100M32"])
      + f"   vs 平衡口径 RF P100 36.48 / V100M32 82.22")
