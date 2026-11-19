#!/usr/bin/env python3
"""图 4（随机森林上界探测）重画脚本 —— 只用 xcad_v2 上带日志的读数。

为什么要有这个脚本：手稿 §1.3/§5.5 引用的 0.58 / 0.59 长期**没有任何日志支撑**
（`outputs/reports` 下找不到 sklearn/RF 的产物），旧图 fig_rf_ceiling.pdf 是 6 月
legacy 撞号编号时期的产物。本脚本不写死数字：它解析 diagnose_sklearn.py 的六份
（2 切分 × 3 种子）日志 + 三份 ours 旗舰 final-test 日志，先过断言（GATE），
再画图并打印"剩余差距按类分解"。

用法（服务器上）：
  python3 make_fig_rf_ceiling_v2.py \
      --rf-dir   ~/xcad/outputs/reports/rf_ceiling_v2 \
      --idspace-root ~/xcad/outputs/reports/baselines_idspace \
      --out      ~/xcad/outputs/reports/figs_v2/fig_rf_ceiling_v2.pdf
"""
import argparse
import os
import re

CLS = ["T4", "MISC", "P100", "V100", "V100M32"]
N_EXPECT = {"T4": 5032, "MISC": 5246, "P100": 552, "V100": 811, "V100M32": 195}
SEEDS = ("42", "123", "2024")

# ours 旗舰三种子（与图 2/图 3 同源），期望值抄自
# docs/reviews/2026-11-17_idspace_main_table.md §3
OURS_ARMS = [
    ("diag_joint_v2.log", 0.5085, 0.065962),
    ("v2_joint_s123.log", 0.5069, 0.066374),
    ("v2_joint_s2024.log", 0.5069, 0.065321),
]


def parse_final_block(path):
    """ours 侧：返回 (逐类 Hits@1, macro, 总 MRR)，只取最后一个 `Using best epoch` 之后。"""
    text = open(path, errors="ignore").read().replace("\r", "\n")
    marks = [m.start() for m in re.finditer(r"Using best epoch", text)]
    if not marks:
        raise SystemExit(f"FATAL {path}: 没有 eval 块")
    tail = text[marks[-1]:]
    per = {}
    for c in CLS:
        found = re.findall(rf"{c}:\s*Hits@1=([0-9.]+)\s*\(n=(\d+)\)", tail)
        if not found:
            raise SystemExit(f"FATAL {path}: 缺 {c} 的逐类 Hits@1")
        v, n = found[-1]
        if int(n) != N_EXPECT[c]:
            raise SystemExit(f"FATAL {path}: {c} 的 n={n} != {N_EXPECT[c]}（不是 xcad_v2 类频）")
        per[c] = float(v)
    mac = re.findall(r"Per-GPU Hits@1 Macro \(unweighted\)\s*=\s*([0-9.]+)", tail)
    mrr = re.findall(r"MRR \(filter_ent\):\s*([0-9.]+)", tail)
    if not mac or not mrr:
        raise SystemExit(f"FATAL {path}: 缺 macro 或总 MRR")
    return per, float(mac[-1]), float(mrr[-1])


def parse_sklearn_log(path, model="RandomForest"):
    """diagnose_sklearn.py 的日志 -> {macro, per-class, n, 众数基线, 特征重要性}。"""
    text = open(path, errors="ignore").read()
    split = re.findall(r"切分模式:\s*(.+)", text)[0].strip()
    lines = text.splitlines()
    idx = [k for k, ln in enumerate(lines) if ln.strip().startswith(model)]
    if not idx:
        raise SystemExit(f"FATAL {path}: 找不到 {model} 块")
    k = idx[0] + 1
    while k < len(lines) and lines[k].strip() and set(lines[k].strip()) <= set("─"):
        k += 1
    body_lines = []
    while k < len(lines) and not (lines[k].strip().startswith("──")
                                  or lines[k].strip().startswith("==")):
        body_lines.append(lines[k])
        k += 1
    body = "\n".join(body_lines)
    macro = float(re.search(r"macro avg:\s*([0-9.]+)%", body).group(1))
    acc = float(re.search(r"总准确率:\s*([0-9.]+)%", body).group(1))
    per, ns = {}, {}
    for cid, name, v, n in re.findall(
            r"type\s+(\d+)\s+\((\w+)\s*\):\s+([0-9.]+)%\s+\(n=\s*(\d+)\)", body):
        if name in CLS:
            per[name] = float(v)
            ns[name] = int(n)
    if sorted(per) != sorted(CLS):
        raise SystemExit(f"FATAL {path}: 逐类缺类，实际 {sorted(per)}")
    maj = re.findall(r"\[(训练|测试)集众数=type (\d+)\] 总准确率=([0-9.]+)%\s+macro=([0-9.]+)%", text)
    imp = re.findall(r"^\s*\d+\.\s+(\w+)\s+([0-9.]+)\s*$", text, re.M)
    imp = [(k, float(v)) for k, v in imp if k in
           ("avg_plan_cpu", "std_plan_cpu", "min_plan_cpu", "max_plan_cpu",
            "avg_plan_mem", "std_plan_mem", "min_plan_mem", "max_plan_mem",
            "avg_plan_gpu", "std_plan_gpu", "min_plan_gpu", "max_plan_gpu",
            "instance_count")]
    return dict(split=split, macro=macro, acc=acc, per=per, n=ns,
                majority=maj, imp=imp, total_test=sum(ns.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf-dir", required=True)
    ap.add_argument("--idspace-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # ---- GATE 1：tau-split 的测试类频必须等于主表的 macro5 类频 ----
    rf = {}
    for split, stem in (("tau", "v2_tau"), ("algo", "v2_algo")):
        for seed in SEEDS:
            fname = f"{stem}.log" if seed == "42" else f"{stem}_s{seed}.log"
            d = parse_sklearn_log(os.path.join(args.rf_dir, fname))
            if split == "tau":
                if d["n"] != N_EXPECT:
                    raise SystemExit(f"GATE FAIL {fname}: tau-split 测试类频 {d['n']} != {N_EXPECT}")
                if d["total_test"] != 11836:
                    raise SystemExit(f"GATE FAIL {fname}: 测试 r1 边 {d['total_test']} != 11836")
            if abs(d["macro"] - sum(d["per"][c] for c in CLS) / 5.0) > 0.02:
                raise SystemExit(f"GATE FAIL {fname}: macro 与逐类均值不自洽")
            for kind, cid, _, mac in d["majority"]:
                if abs(float(mac) - 20.00) > 0.01:
                    raise SystemExit(f"GATE FAIL {fname}: {kind}众数 macro={mac} != 20.00")
            rf.setdefault(split, {})[seed] = d
            print(f"GATE PASS {fname:22s} split={d['split']:18s} RF macro={d['macro']:.2f} "
                  f"acc={d['acc']:.2f} n_test={d['total_test']}")

    # ---- GATE 2：legacy 与 v2 的上界必须逐位相同（上界只吃特征+类别，不吃实体编号）----
    for split, fname in (("tau", "legacy_tau.log"), ("algo", "legacy_algo.log")):
        d = parse_sklearn_log(os.path.join(args.rf_dir, fname))
        v2 = rf[split]["42"]
        if abs(d["macro"] - v2["macro"]) > 1e-9:
            raise SystemExit(f"GATE FAIL {fname}: legacy RF macro {d['macro']} != v2 {v2['macro']}")
        print(f"GATE PASS {fname:22s} 与 v2 逐位相同（RF macro={d['macro']:.2f}）"
              f" => 上界对实体编号不敏感")

    # ---- GATE 3：ours 三种子，期望值钉死 ----
    ours_per, ours_mac = [], []
    for fname, exp_mac, exp_mrr in OURS_ARMS:
        per, mac, mrr = parse_final_block(os.path.join(args.idspace_root, fname))
        if abs(mac - exp_mac) > 1e-4 or abs(mrr - exp_mrr) > 1e-6:
            raise SystemExit(f"GATE FAIL {fname}: macro {mac} (期望 {exp_mac}) / MRR {mrr} (期望 {exp_mrr})")
        ours_per.append(per)
        ours_mac.append(mac)
        print(f"GATE PASS {fname:22s} ours macro={mac:.4f} mrr={mrr:.6f}")

    def ms(vals):
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        return m, sd

    stats = {}
    for split in ("tau", "algo"):
        macs = [rf[split][s]["macro"] for s in SEEDS]
        m, sd = ms(macs)
        stats[split] = (m, sd, macs)
        print(f"RF {split:4s} macro = {m:.2f} +- {sd:.2f}  raw = " + " ".join(f"{v:.2f}" for v in macs))
    ours_m, ours_sd = ms(ours_mac)
    print(f"ours macro = {ours_m*100:.2f} +- {ours_sd*100:.2f}")

    # ---- 剩余差距按类分解（同一测试边：tau-split）----
    rf_per_mean = {c: ms([rf["tau"][s]["per"][c] for s in SEEDS])[0] for c in CLS}
    ours_per_mean = {c: sum(p[c] for p in ours_per) / 3.0 * 100.0 for c in CLS}
    print("\n剩余差距按类分解（RF tau-split 三种子均值 - ours 三种子均值，×100）")
    diffs = {}
    for c in CLS:
        diffs[c] = rf_per_mean[c] - ours_per_mean[c]
        print(f"  {c:9s} RF {rf_per_mean[c]:6.2f}  ours {ours_per_mean[c]:6.2f}  差 {diffs[c]:+6.2f}")
    total_gap = sum(diffs.values()) / 5.0
    rare = (diffs["P100"] + diffs["V100M32"]) / 5.0
    print(f"  macro 差合计 = {total_gap:.2f} 点；其中 P100+V100M32 = {rare:.2f} 点"
          f"（占 {100*rare/total_gap:.1f}%）")
    print("  RF 特征重要性 top5(tau,seed42): " +
          ", ".join(f"{k}={v:.4f}" for k, v in rf["tau"]["42"]["imp"][:5]))

    # ---- 画图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = [("By edge\n(tau-split)", "tau"), ("By algorithm\n(unseen algos)", "algo")]
    x = np.arange(len(groups))
    w = 0.30
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.bar(x - w / 2, [0.20, 0.20], w, label="Majority-guess", color="#9FAEBB",
           edgecolor="white", linewidth=0.4)
    heights = [stats["tau"][0] / 100.0, stats["algo"][0] / 100.0]
    errs = [stats["tau"][1] / 100.0, stats["algo"][1] / 100.0]
    ax.bar(x + w / 2, heights, w, yerr=errs, capsize=2,
           label="Random forest (profile only)", color="#1F6188",
           edgecolor="white", linewidth=0.4, error_kw=dict(lw=0.6))
    for xi, h in zip(x - w / 2, [0.20, 0.20]):
        ax.text(xi, h + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)
    for i, (xi, h) in enumerate(zip(x + w / 2, heights)):
        ax.text(xi, h + errs[i] + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([g for g, _ in groups], fontsize=7)
    ax.set_ylabel("Per-GPU macro Hits@1", fontsize=8)
    ax.set_ylim(0, 0.72)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6.0, frameon=False, loc="upper left", handlelength=1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, format="pdf")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
