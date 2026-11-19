#!/usr/bin/env python3
"""图 2 / 图 3 重画脚本 —— 只用 xcad_v2（修好全局编号）上的 final-test 读数。

为什么要有这个脚本：手稿旧版图 3 的两条序列（"RE-GCN backbone macro 0.20 / T4=1.00"、
"+ compatibility head macro 0.52 / P100=0.04, V100M32=0.03"）与图 2 的 "+Compat 0.519"
都出自 6 月的 legacy 撞号编号，在 xcad_v2 上对不上任何一个臂。本脚本不写死数字：它从日志的
**最后一个** `Using best epoch` 之后的 final-test 块里解析逐类 Hits@1 与 macro，先过断言
（GATE：期望值来自两份 night_summary.tsv，类频 n 也断言），再画图。

用法（服务器上）：
  python3 make_fig_freq_prior_v2.py --root ~/xcad/outputs/reports/baselines_idspace \
      --out   .../figs_v2/fig_freq_prior_v2.pdf \
      --out-macro .../figs_v2/fig_macro_compare_v2.pdf
"""
import argparse
import os
import re
import sys

CLS = ["T4", "MISC", "P100", "V100", "V100M32"]
N_EXPECT = {"T4": 5032, "MISC": 5246, "P100": 552, "V100": 811, "V100M32": 195}

# (label, log 文件, 期望 macro, 期望总 MRR)  —— 期望值抄自
# docs/reviews/2026-11-17_idspace_main_table.md §3 的新主表，逐条断言，不匹配就退出。
SOURCES = [
    ("regcn",  "v2_regcn.log",        0.2029, 0.050227),
    ("cen",    "v2_cen_test_r2.log",   0.2000, 0.049006),
    ("retia",  "v2_retia_curtrain.log", 0.2000, 0.047344),
    ("ours42", "diag_joint_v2.log",    0.5085, 0.065962),
    ("ours123", "v2_joint_s123.log",   0.5069, 0.066374),
    ("ours2024", "v2_joint_s2024.log", 0.5069, 0.065321),
]


def parse_final_block(path):
    """返回 (逐类 Hits@1 dict, macro, 总 MRR)。只取最后一个 `Using best epoch` 之后。"""
    text = open(path, errors="ignore").read().replace("\r", "\n")
    marks = [m.start() for m in re.finditer(r"Using best epoch", text)]
    if not marks:
        raise SystemExit(f"FATAL {path}: 没有 eval 块")
    tail = text[marks[-1]:]
    mrr = re.findall(r"MRR \(filter_ent\):\s*([0-9.]+)", tail)
    mac = re.findall(r"Per-GPU Hits@1 Macro \(unweighted\)\s*=\s*([0-9.]+)", tail)
    per = {}
    for c in CLS:
        found = re.findall(rf"{c}:\s*Hits@1=([0-9.]+)\s*\(n=(\d+)\)", tail)
        if not found:
            raise SystemExit(f"FATAL {path}: 缺 {c} 的逐类 Hits@1")
        v, n = found[-1]
        if int(n) != N_EXPECT[c]:
            raise SystemExit(f"FATAL {path}: {c} 的 n={n} != {N_EXPECT[c]}（不是 xcad_v2 的类频）")
        per[c] = float(v)
    if not mrr or not mac:
        raise SystemExit(f"FATAL {path}: 缺总 MRR 或 macro")
    return per, float(mac[-1]), float(mrr[-1])


# 图 2（各配置 per-GPU macro）用的 12 个臂：4 配置 × 3 种子。
# label -> [(log 相对 --root 的路径, 期望 macro, 期望总 MRR)]，期望值抄自
# outputs/reports/baselines_idspace/ablate_v2/night_summary.tsv 与
# night_2026_11_18/night_summary.tsv，逐条断言。
MACRO_ARMS = {
    "① backbone": [
        ("ablate_v2/a1_base.log",                 0.0621, 0.003520),
        ("../night_2026_11_18/s4_base_s123.log",  0.1879, 0.046644),
        ("../night_2026_11_18/s4_base_s2024.log", 0.2004, 0.043395),
    ],
    "② +Hetero": [
        ("ablate_v2/a3_het.log",                  0.2000, 0.049246),
        ("../night_2026_11_18/s4_het_s123.log",   0.2000, 0.049829),
        ("../night_2026_11_18/s4_het_s2024.log",  0.2000, 0.049009),
    ],
    "③ +Compat": [
        ("ablate_v2/a2_compat.log",               0.5099, 0.061799),
        ("../night_2026_11_18/s4_compat_s123.log", 0.5214, 0.062992),
        ("../night_2026_11_18/s4_compat_s2024.log", 0.4982, 0.062594),
    ],
    "④ xCAD full": [
        ("ablate_v2/a4_joint.log",                0.5085, 0.065962),
        ("v2_joint_s123.log",                     0.5069, 0.066374),
        ("v2_joint_s2024.log",                    0.5069, 0.065321),
    ],
}
SEEDS = ("42", "123", "2024")


def macro_from_logs(root):
    """解析 12 臂的 macro，并顺带报告每臂恒答哪一类（供图注与正文核对）。"""
    out = {}
    latch = {}
    for label, arms in MACRO_ARMS.items():
        macs = []
        for fname, exp_mac, exp_mrr in arms:
            per, mac, mrr = parse_final_block(os.path.normpath(os.path.join(root, fname)))
            if abs(mac - exp_mac) > 1e-4 or abs(mrr - exp_mrr) > 1e-6:
                raise SystemExit(
                    f"GATE FAIL {fname}: macro {mac} (期望 {exp_mac}) / MRR {mrr} (期望 {exp_mrr})")
            top = [c for c in CLS if per[c] >= 0.999]
            latch[fname] = (top[0] if len(top) == 1 else "mixed") if max(per.values()) >= 0.999 else "none"
            macs.append(mac)
            print(f"GATE PASS {fname:40s} macro={mac:.4f} mrr={mrr:.6f} latch={latch[fname]}")
        out[label] = macs
    return out, latch


def draw_macro_compare(out_path, macs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(MACRO_ARMS)
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(3.7, 2.5))
    colors = ["#1F6188", "#4C8FB5", "#9FAEBB"]
    for i, seed in enumerate(SEEDS):
        ys = [macs[l][i] for l in labels]
        ax.bar(x + (i - 1) * w, ys, w, label=f"seed {seed}",
               color=colors[i], edgecolor="white", linewidth=0.4)
    # 逐种子读数：三档四舍五入后相同则只标一个居中值，避免文字叠字
    for j, l in enumerate(labels):
        txt = ["%.2f" % v for v in macs[l]]
        if len(set(txt)) == 1:
            ax.text(j, macs[l][0] + 0.014, txt[0], ha="center", va="bottom", fontsize=5.8)
        else:
            for i, t in enumerate(txt):
                ax.text(j + (i - 1) * w, macs[l][i] + 0.014, t, ha="center",
                        va="bottom", fontsize=5.2)
    ax.axhline(0.20, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("r1 per-GPU macro Hits@1", fontsize=8)
    ax.set_ylim(0, 0.70)
    ax.tick_params(axis="y", labelsize=7)
    from matplotlib.lines import Line2D
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=f"seed {s}")
               for c, s in zip(colors, SEEDS)]
    handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=0.8,
                          label="mode-guess = 0.200"))
    ax.legend(handles=handles, fontsize=6.0, frameon=False, ncol=2,
              loc="upper left", handlelength=1.1, columnspacing=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    print("WROTE", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-macro", default=None, help="额外输出图 2（各配置 macro 逐种子）")
    args = ap.parse_args()

    vals = {}
    for key, fname, exp_mac, exp_mrr in SOURCES:
        path = os.path.join(args.root, fname)
        per, mac, mrr = parse_final_block(path)
        if abs(mac - exp_mac) > 1e-4 or abs(mrr - exp_mrr) > 1e-6:
            raise SystemExit(f"GATE FAIL {fname}: macro {mac} (期望 {exp_mac}) / MRR {mrr} (期望 {exp_mrr})")
        vals[key] = per
        print(f"GATE PASS {fname:24s} macro={mac:.4f} mrr={mrr:.6f} "
              + " ".join(f"{c}={per[c]:.4f}" for c in CLS))

    # CEN 与 RETIA 都恒答同一个高频类（T4=1.0000、其余 0）⇒ 合并成一条序列，先断言再合并
    for k in ("cen", "retia"):
        if not (vals[k]["T4"] == 1.0 and all(vals[k][c] == 0.0 for c in CLS[1:])):
            raise SystemExit(f"GATE FAIL {k}: 不是恒答 T4，不能与另一条合并")
    degenerate = {c: vals["cen"][c] for c in CLS}
    ours = {c: sum(vals[f"ours{s}"][c] for s in (42, 123, 2024)) / 3.0 for c in CLS}
    ours_macro = sum(ours[c] for c in CLS) / len(CLS)  # 无权重逐类均值 = macro

    series = [
        ("RE-GCN  (macro %.3f)" % (sum(vals["regcn"][c] for c in CLS) / len(CLS)),
         [vals["regcn"][c] for c in CLS], "#9FAEBB"),
        ("CEN / RETIA  (macro %.3f)" % (sum(degenerate[c] for c in CLS) / len(CLS)),
         [degenerate[c] for c in CLS], "#4C8FB5"),
        ("xCAD full  (macro %.3f)" % ours_macro, [ours[c] for c in CLS], "#1F6188"),
    ]
    print("ours 3-seed mean macro = %.4f" % ours_macro)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(CLS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for i, (label, ys, color) in enumerate(series):
        bars = ax.bar(x + (i - 1) * w, ys, w, label=label, color=color,
                      edgecolor="white", linewidth=0.4)
        for b, v in zip(bars, ys):
            if v > 0.005:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CLS, fontsize=8)
    ax.set_xlabel("GPU class", fontsize=8)
    ax.set_ylabel("Per-GPU Hits@1", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", handlelength=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, format="pdf")
    print("WROTE", args.out)

    if args.out_macro:
        macs, _ = macro_from_logs(args.root)
        for label, ys in macs.items():
            m = sum(ys) / len(ys)
            sd = (sum((v - m) ** 2 for v in ys) / (len(ys) - 1)) ** 0.5
            print(f"MEAN {label:14s} macro = {m:.4f} +- {sd:.4f}   raw = " +
                  " ".join(f"{v:.4f}" for v in ys))
        draw_macro_compare(args.out_macro, macs)


if __name__ == "__main__":
    main()
