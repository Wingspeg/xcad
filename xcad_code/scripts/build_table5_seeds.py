#!/usr/bin/env python3
"""build_table5_seeds.py — 把论文表 5（兼容性头 λ/β）钉成 v2 上的真三种子表。

口径：只取每个日志最后一次 "Load Model name … Using best epoch : N" 之后的 final test 块
（与 2026-11-17_idspace_main_table.md §1 第 4 条一致），读 Per-GPU macro 与逐类 H@1。
读数一律来自日志，不手写、不插值；缺档直接打 None 并让汇总失败，不猜。

判线是 2026-11-18 开卡之前写死的三条（见 scripts/t5_2026_11_18.sh 头部）：
  L1 不回退线：任一臂的 macro 不得比**同 seed 的旗舰（λ5β0.5）**低超过 NOISE=0.02
  L2 结论线：同一格内三种子跨度 ≤ NOISE 才许写"不敏感"
  L3 β 方向：手稿称"β=0 方差略大故取 0.5"，本版按实测判它成立/不成立
"""
import os
import re

NOISE = 0.02
ROOT = os.path.expanduser("~/xcad/outputs/reports/")

# cell -> {seed: 相对路径}
SRC = {
    "lam5_beta0.5": {  # 主配置 = 表5 的 λ=5 格与 β=0.5 格（同一个配置）
        42:   "night_2026_11_18/p0_regression.log",
        123:  "baselines_idspace/v2_joint_s123.log",
        2024: "baselines_idspace/v2_joint_s2024.log",
    },
    "lam1": {
        42:   "night_2026_11_18/s3_joint_lam1.log",
        123:  "t5_2026_11_18/t5_lam1_s123.log",
        2024: "t5_2026_11_18/t5_lam1_s2024.log",
    },
    "lam10": {
        42:   "night_2026_11_18/s3_joint_lam10.log",
        123:  "t5_2026_11_18/t5_lam10_s123.log",
        2024: "t5_2026_11_18/t5_lam10_s2024.log",
    },
    "beta0": {
        42:   "night_2026_11_18/s3_joint_beta0.log",
        123:  "t5_2026_11_18/t5_beta0_s123.log",
        2024: "t5_2026_11_18/t5_beta0_s2024.log",
    },
}
MACRO = re.compile(r"Per-GPU Hits@1 Macro \(unweighted\)\s*=\s*([0-9.]+)")
CLS = re.compile(r"(T4|MISC|P100|V100|V100M32)=([0-9.]+)/\d+")
R1H1 = re.compile(r"r1_suits\s+MRR=[0-9.]+\s+H@1=([0-9.]+)")


def read(rel):
    """-> (macro5, r1_h1, {类: H@1})，只取 final test 块。"""
    txt = open(os.path.join(ROOT, rel), errors="ignore").read().replace("\r", "\n")
    hits = list(re.finditer(r"Load Model name", txt))
    if not hits:
        raise SystemExit("FATAL: %s 没有 final test 块" % rel)
    tail = txt[hits[-1].start():]
    m = MACRO.search(tail)
    if not m:
        raise SystemExit("FATAL: %s final test 块里没有 macro 行" % rel)
    cls = {k: float(v) for k, v in CLS.findall(tail)}
    r1 = R1H1.search(tail)
    return float(m.group(1)), (float(r1.group(1)) if r1 else None), cls


def msd(xs):
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, None
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, var ** 0.5


SEEDS = (42, 123, 2024)
val = {c: {s: read(p)[0] for s, p in src.items()} for c, src in SRC.items()}

print("== 表 5 重建（xcad_v2，h=6，macro ×100）==")
print("| 超参 | 取值 | seed 42 | seed 123 | seed 2024 | 三种子均值±sd | 跨度 |")
print("|---|---|---|---|---|---|---|")
order = [("lam1", "λ=1", "融合权重 λ（固定 β=0.5）"),
         ("lam5_beta0.5", "λ=5", "融合权重 λ（固定 β=0.5）"),
         ("lam10", "λ=10", "融合权重 λ（固定 β=0.5）"),
         ("beta0", "β=0", "辅助损失 β（固定 λ=5）"),
         ("lam5_beta0.5", "β=0.5", "辅助损失 β（固定 λ=5）")]
for key, label, group in order:
    vs = [val[key][s] for s in SEEDS]
    mean, sd = msd(vs)
    span = max(vs) - min(vs)
    cells = " | ".join("%.2f" % (v * 100) for v in vs)
    print("| %s | %s | %s | %.2f ± %.2f | %.2f |"
          % (group, label, cells, mean * 100, (sd or 0) * 100, span * 100))
print()
print("== 判线（开卡前写死）==")
flag = val["lam5_beta0.5"]
for key, label in [("lam1", "λ=1"), ("lam10", "λ=10"), ("beta0", "β=0")]:
    for s in SEEDS:
        d = val[key][s] - flag[s]
        tag = "FAIL" if d < -NOISE else "pass"
        print("  L1 %s seed %-4d macro %.4f vs 旗舰 %.4f  Δ%+.4f  %s(线 −%.2f)"
              % (label, s, val[key][s], flag[s], d, tag, NOISE))
        if tag == "FAIL":
            print("        逐类 H@1：%s" % {k: v for k, v in
                  sorted(read(SRC[key][s])[2].items())})
NAME = {"lam5_beta0.5": "λ=5 / β=0.5（主配置）", "lam1": "λ=1",
        "lam10": "λ=10", "beta0": "β=0"}
for key in SRC:
    vs = [val[key][s] for s in SEEDS]
    span = max(vs) - min(vs)
    print("  L2 %-16s 三种子跨度 %.4f(=%.2f 点)  %s(线 %.2f) -> 该格%s"
          % (NAME[key], span, span * 100, "pass" if span <= NOISE else "FAIL",
             NOISE, "可写不敏感" if span <= NOISE else "不可写不敏感"))

lam_means = [msd([val[k][s] for s in SEEDS])[0] for k in ("lam1", "lam5_beta0.5", "lam10")]
print("  λ 三点均值 %.4f / %.4f / %.4f -> 跨 λ 差 %.4f(=%.2f 点) %s"
      % (*lam_means, max(lam_means) - min(lam_means),
         (max(lam_means) - min(lam_means)) * 100,
         "在噪声内" if max(lam_means) - min(lam_means) <= NOISE else "超出噪声"))
b0 = [val["beta0"][s] for s in SEEDS]
b5 = [flag[s] for s in SEEDS]
m0, s0 = msd(b0)
m5, s5 = msd(b5)
print("  L3 β=0  %.2f±%.2f (跨度 %.2f)   β=0.5  %.2f±%.2f (跨度 %.2f)"
      % (m0 * 100, s0 * 100, (max(b0) - min(b0)) * 100,
         m5 * 100, s5 * 100, (max(b5) - min(b5)) * 100))
print("      -> 方差：%s   均值：%s"
      % ("β=0 更大，手稿那句方向成立" if s0 > s5 else "β=0 并不更大，手稿那句没有证据",
         "β=0 也更低（差 %.2f 点）" % ((m5 - m0) * 100) if m0 < m5
         else "β=0 不低"))
print("      逐类（β=0）: ", {s: read(SRC['beta0'][s])[2] for s in SEEDS})
print("      逐类（β=0.5）:", {s: read(SRC['lam5_beta0.5'][s])[2] for s in SEEDS})
