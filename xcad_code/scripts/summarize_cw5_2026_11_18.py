#!/usr/bin/env python3
"""summarize_cw5_2026_11_18.py — 5 种子类别加权汇总 + 按 §13.3 预注册判线判定"主表是否换 inv 档"。

判线在开卡前写死（docs/reviews/2026-11-18_class_weight.md §13.3），本脚本只做机械套用：

  W1 换表线（三条同时满足才把表 2 的 ours 行换成 inv）
    W1a  Δ5 = mean(macro5|inv,5) − mean(macro5|none,5) ≥ 0.02   （噪声阈值，v2020 实测 0.0172）
    W1b  inv 五种子 macro5 的样本标准差 ≤ 0.02
    W1c  同种子配对 inv > none 在 ≥4/5 个种子上成立
  W2  W1a 过但 W1b/W1c 不过 ⇒ 不换主表，inv 进消融小节按逐种子原始值列，正文只写方向不写量级
  W3  W1a 不过 ⇒ 判负，主表保持 none，§39 的 "+4.84 点" 降级为"3 种子上的方向"

口径纪律（与 §31/§37/§38/§39 一致）：
  * 只读 final test 那一块，逐类 n 必须是 5032/5246/552/811/195，否则拒读（cw_log_parse.parse 里断言）。
  * 表 2 的"总 MRR / 总 H@1"取 **filter_ent** 口径（不是 raw_ent）；本脚本对这两列逐 seed 复现
    2026-11-17_idspace_main_table.md §3 的三行原始值，复现不过直接停。
  * inv 臂的权重向量必须与 G0 独立算出的 expected_weights.json 逐位一致（4 位小数）。
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cw_log_parse import class_weight_lines, parse  # noqa: E402

ROOT = os.path.expanduser("~/xcad/outputs/reports/")
CW = os.path.join(ROOT, "class_weight_2026_11_18")
EXT = os.path.join(ROOT, "class_weight_ext_2026_11_18")

SEEDS_OLD = [42, 123, 2024]
SEEDS_NEW = [7, 2025]
ALL_SEEDS = SEEDS_OLD + SEEDS_NEW
CLS5 = ["T4", "MISC", "P100", "V100", "V100M32"]

NOISE = 0.02          # v2020 run-to-run 实测 0.0172 ⇒ 判线取 0.02
W1C_MIN = 4           # 配对至少 4/5 同号

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

# 主表 §3 逐 seed 原始值（filter_ent 口径），G-CAL 用
MAIN_TABLE = {42: (0.065962, 0.059156), 123: (0.066374, 0.059459), 2024: (0.065321, 0.058699)}

problems = []

print("==== 类别加权 5 种子汇总（final test；表 2 口径 = filter_ent）====\n")

# ── G-CAL：解析器口径必须逐 seed 复现已发表的主表 ours 行 ──────────────────
print("── G-CAL 主表口径复现（none 老种子 filter_ent MRR / H@1）──")
g_cal = []
for s, (em, eh) in MAIN_TABLE.items():
    d = parse(SRC[("none", s)])
    ok = d.get("ok") and d["mrr_filter_ent"] == em and d["h1_filter_ent"] == eh
    g_cal.append(ok)
    print(f"  seed {s:<5} 读数 {d.get('mrr_filter_ent')} / {d.get('h1_filter_ent')}"
          f"   期望 {em} / {eh}   {'PASS' if ok else 'FAIL'}")
if not all(g_cal):
    print("\nFATAL: G-CAL 没过 —— filter_ent 口径对不上主表，后面的数全部不可用。")
    sys.exit(3)
print("  G-CAL: ALL PASS\n")

exp = json.load(open(os.path.join(CW, "expected_weights.json")))
w_inv = np.asarray(exp["expected"]["inv"]["w"], dtype=float)

# ── 读 10 臂 ───────────────────────────────────────────────────────────────
rows = []
for mode in ("none", "inv"):
    for s in ALL_SEEDS:
        path = SRC[(mode, s)]
        d = parse(path)
        if not d.get("ok"):
            problems.append(f"{mode}_s{s}: 拒读 —— {d.get('why')}")
            print(f"  !! {mode}_s{s} 拒读: {d.get('why')}")
            continue
        cw_lines = class_weight_lines(path)
        if mode == "inv":
            m = [ln for ln in cw_lines if "capped&renorm w(all 6)" in ln]
            if not m:
                problems.append(f"inv_s{s}: 日志里没有 [class-weight] 权重行")
            else:
                got = np.asarray([float(x) for x in
                                  re.search(r"w\(all 6\)=\[([0-9. ]+)\]", m[-1]).group(1).split()])
                if not np.allclose(got, w_inv, atol=5e-5):
                    problems.append(f"inv_s{s}: 权重向量与 G0 期望不符 {got} vs {w_inv}")
        rows.append(dict(mode=mode, seed=s, path=path, **{k: d[k] for k in (
            "macro5", "mrr_filter_ent", "h1_filter_ent", "r1_mrr", "r1_h1", "inseg_h1",
            "agree", "best_epoch")}, **{c: d["per"][c] for c in CLS5}))

if len(rows) != 10:
    print(f"\nFATAL: 只读到 {len(rows)}/10 臂，判线不套用。问题: {problems}")
    sys.exit(4)

METRICS = ["macro5", "mrr_filter_ent", "h1_filter_ent", "r1_mrr", "r1_h1"] + CLS5 + ["agree"]


def stat(mode, metric, seeds=None):
    v = [r[metric] for r in rows if r["mode"] == mode and (seeds is None or r["seed"] in seeds)]
    return float(np.mean(v)), float(np.std(v, ddof=1)), v


print("── 逐种子原始值（×100）──")
hdr = ["macro5", "mrr_filter_ent", "h1_filter_ent", "r1_mrr", "r1_h1"] + CLS5
print("%-6s %-6s" % ("mode", "seed") + "".join("%12s" % h for h in hdr))
for r in sorted(rows, key=lambda x: (x["mode"], ALL_SEEDS.index(x["seed"]))):
    print("%-6s %-6d" % (r["mode"], r["seed"]) + "".join("%12.4f" % (100 * r[h]) for h in hdr))

print("\n── 五种子均值 ± 样本标准差（×100）──")
tbl = {}
for m in METRICS:
    for mode in ("none", "inv"):
        mu, sd, vals = stat(mode, m)
        tbl[(mode, m)] = dict(mean=mu, sd=sd, vals=vals)
        print(f"  {mode:<5} {m:<16} {100*mu:8.2f} ± {100*sd:5.2f}   "
              + " ".join(f"{100*v:.2f}" for v in vals))

# ── 配对 ───────────────────────────────────────────────────────────────────
pair = []
for s in ALL_SEEDS:
    a = [r for r in rows if r["mode"] == "inv" and r["seed"] == s][0]
    b = [r for r in rows if r["mode"] == "none" and r["seed"] == s][0]
    pair.append(dict(seed=s, inv=a["macro5"], none=b["macro5"], diff=a["macro5"] - b["macro5"],
                     h1_inv=a["r1_h1"], h1_none=b["r1_h1"]))
print("\n── 同种子配对（macro5 = inv − none；r1 H@1 同列以显代价）──")
for p in pair:
    print(f"  seed {p['seed']:<5} macro {p['none']:.4f} -> {p['inv']:.4f}  Δ={p['diff']:+.4f}"
          f"   r1 H@1 {p['h1_none']:.4f} -> {p['h1_inv']:.4f}  Δ={p['h1_inv']-p['h1_none']:+.4f}")

d5 = float(np.mean([p["diff"] for p in pair]))
win = sum(1 for p in pair if p["diff"] > 0)
sd_inv = tbl[("inv", "macro5")]["sd"]

# ── 判线 ───────────────────────────────────────────────────────────────────
w1a, w1b, w1c = d5 >= NOISE, sd_inv <= NOISE, win >= W1C_MIN
print("\n── 预注册判线套用 ──")
print(f"  W1a 配对增益 Δ5 = {d5:+.4f} ≥ {NOISE}            : {'PASS' if w1a else 'FAIL'}")
print(f"  W1b inv 五种子 macro5 sd = {sd_inv:.4f} ≤ {NOISE}  : {'PASS' if w1b else 'FAIL'}")
print(f"  W1c 配对 inv>none {win}/5 ≥ {W1C_MIN}/5            : {'PASS' if w1c else 'FAIL'}")
if w1a and w1b and w1c:
    verdict = "W1 换表线全过 ⇒ 主表 ours 行换成 inv 档（表 2/3/4/图 2 同步，§13.4 清单）"
elif w1a:
    failed = [n for n, ok in (("W1b sd", w1b), ("W1c 同号数", w1c)) if not ok]
    verdict = (f"W2 ⇒ 不换主表。inv 进消融小节按逐种子原始值列，正文只写方向不写量级"
               f"（Δ5={d5:+.4f} 过线；不过的是 {'、'.join(failed)}：sd={sd_inv:.4f}，同号 {win}/5）")
else:
    verdict = f"W3 ⇒ 判负。主表保持 none，§39 的改进降级为\"3 种子上的方向\"（Δ5={d5:+.4f} < {NOISE}）"
print(f"\n  判定：{verdict}")

# ── 换表候选行（无论判线过不过都算出来，写着好看清代价）────────────────────
print("\n── 表 2 ours 行的两套候选写法（×100，mean ± 五种子 sd）──")
for mode in ("none", "inv"):
    c = [f"{100*tbl[(mode, m)]['mean']:.2f} ± {100*tbl[(mode, m)]['sd']:.2f}" for m in
         ("mrr_filter_ent", "h1_filter_ent", "r1_mrr", "r1_h1", "macro5")]
    print(f"  {mode:<5} 总MRR {c[0]}   总H@1 {c[1]}   r1MRR {c[2]}   r1H@1 {c[3]}   macro {c[4]}")
print("  逐类 macro 分量（inv）: " + "  ".join(f"{c} {100*tbl[('inv', c)]['mean']:.2f}" for c in CLS5))
print("  逐类 macro 分量（none）: " + "  ".join(f"{c} {100*tbl[('none', c)]['mean']:.2f}" for c in CLS5))

if problems:
    print("\n!! 问题清单：")
    for p in problems:
        print("   -", p)

out = dict(rows=rows, pair=pair, stats={f"{m}|{k}": v for (k, m), v in tbl.items()},
           lines=dict(NOISE=NOISE, W1A=d5, W1B=sd_inv, W1C=win, W1C_MIN=W1C_MIN),
           verdict=verdict, swap=(w1a and w1b and w1c), problems=problems)
with open(os.path.join(EXT, "cw5_summary.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\n已存 -> {EXT}/cw5_summary.json")
sys.exit(0 if not problems else 5)
