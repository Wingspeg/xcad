#!/usr/bin/env python3
"""analyze_night_2026_11_18.py — 把夜间队列 + 昨天已有的臂拼成四张可以直接抄进论文表的读数。

四张表:
  A 表 4 消融(仅骨架 / +点1 / +点2 / 完整)× seed {42,123,2024}, h=6
  B 图 5 的历史长度扫描: 完整模型 与 仅骨架, h ∈ {1,3,6,9,12}, seed 42
  C 表 5 的 λ/β 敏感性: λ ∈ {1,5,10,可学习} @ β=0.5、β=0 @ λ=5, seed 42
  D maas 旗舰(--base-prior --head-prior --prior-ts)× seed {42,43,44,45}

口径与 2026-11-17_idspace_main_table.md §1 一致: 只读最后一次 "Load Model name" 之后的
final test 块; mean = 无权重算术平均, sd = 样本标准差(n-1)。
判定阈值: v2020 一次运行的 run-to-run 噪声实测 0.0172(全体)/0.0196(全新 head),
所以任何小于 0.02 的差不能当结论 —— 脚本把 mean−sd、mean+sd 一起打出来就是为了这一眼。
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.expanduser("~/xcad/xcad_code/scripts"))
from summarize_night_logs import parse  # noqa: E402

NIGHT = os.path.expanduser("~/xcad/outputs/reports/night_2026_11_18")
BID = os.path.expanduser("~/xcad/outputs/reports/baselines_idspace")
MAAS = os.path.expanduser("~/xcad/xcad_code/xcad_model/src/maas_arms")

COLS = ["mrr_f", "h1_f", "all_h1", "r1_mrr", "r1_h1", "macro5", "rec_h1", "new_h1",
        "never_h1", "rec_edge_h1",
        "in_seg", "seg_h1", "seg_macro", "compat_acc", "place_mrr", "r2_h1",
        "T4", "MISC", "P100", "V100", "V100M32"]

TABLES = {
    "A 表 4 消融三种子(h=6, xcad_v2)": [
        ("仅骨架 base",   [("s42", BID + "/ablate_v2/a1_base.log"),
                           ("s123", NIGHT + "/s4_base_s123.log"),
                           ("s2024", NIGHT + "/s4_base_s2024.log")]),
        ("+点2 compat",   [("s42", BID + "/ablate_v2/a2_compat.log"),
                           ("s123", NIGHT + "/s4_compat_s123.log"),
                           ("s2024", NIGHT + "/s4_compat_s2024.log")]),
        ("+点1 het",      [("s42", BID + "/ablate_v2/a3_het.log"),
                           ("s123", NIGHT + "/s4_het_s123.log"),
                           ("s2024", NIGHT + "/s4_het_s2024.log")]),
        ("完整 xCAD",     [("s42", BID + "/ablate_v2/a4_joint.log"),
                           ("s123", BID + "/v2_joint_s123.log"),
                           ("s2024", BID + "/v2_joint_s2024.log")]),
    ],
    "B 图 5 历史长度扫描(seed 42)": [
        ("完整 h=1",  [("s42", NIGHT + "/s2_joint_h1.log")]),
        ("完整 h=3",  [("s42", NIGHT + "/s2_joint_h3.log")]),
        ("完整 h=6",  [("s42", BID + "/ablate_v2/a4_joint.log")]),
        ("完整 h=9",  [("s42", NIGHT + "/s2_joint_h9.log")]),
        ("完整 h=12", [("s42", NIGHT + "/s2_joint_h12.log")]),
        ("骨架 h=1",  [("s42", NIGHT + "/s1_base_h1.log")]),
        ("骨架 h=3",  [("s42", NIGHT + "/s1_base_h3.log")]),
        ("骨架 h=6",  [("s42", BID + "/ablate_v2/a1_base.log")]),
        ("骨架 h=9",  [("s42", NIGHT + "/s1_base_h9.log")]),
        ("骨架 h=12", [("s42", NIGHT + "/s1_base_h12.log")]),
    ],
    "C 表 5 λ/β 敏感性(完整模型, seed 42)": [
        ("λ=1  β=0.5",   [("s42", NIGHT + "/s3_joint_lam1.log")]),
        ("λ=5  β=0.5(论文)", [("s42", BID + "/ablate_v2/a4_joint.log")]),
        ("λ=10 β=0.5",   [("s42", NIGHT + "/s3_joint_lam10.log")]),
        ("λ=可学 β=0.5", [("s42", NIGHT + "/s3_joint_lamL.log")]),
        ("λ=5  β=0",     [("s42", NIGHT + "/s3_joint_beta0.log")]),
    ],
    "D maas 旗舰四种子(--base-prior --head-prior --prior-ts)": [
        ("maas 四种子合计", [("s42", MAAS + "/priorts.log"), ("s43", MAAS + "/priorts_s43.log"),
                             ("s44", NIGHT + "/s5_maas_priorts_s44.log"),
                             ("s45", NIGHT + "/s5_maas_priorts_s45.log")]),
        ("maas s42",  [("s42", MAAS + "/priorts.log")]),
        ("maas s43",  [("s43", MAAS + "/priorts_s43.log")]),
        ("maas s44",  [("s44", NIGHT + "/s5_maas_priorts_s44.log")]),
        ("maas s45",  [("s45", NIGHT + "/s5_maas_priorts_s45.log")]),
        ("maas 单通道(对照)", [("s42", MAAS + "/headprior.log")]),
        ("maas gate 臂",       [("s42", MAAS + "/priortsgate.log")]),
    ],
}


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "  -  "
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    if len(vals) == 1:
        return "%.4f" % m
    return "%.4f±%.4f" % (m, sd)


def main():
    out = ["# 2026-11-18 夜间队列读数汇总(每格 = 该配置各 seed 的 final test 均值 ± 样本标准差)\n",
           "口径: `xcad_v2` 全局四段编号 / gpu_base=124885 / 全 124,891 实体 time-aware filtered / "
           "r1 正向测试边 11,836。v2020 的单次运行噪声实测 0.0172(全体 H@1), "
           "**跨度小于 0.02 的差不能当结论**。\n"]
    for title, groups in TABLES.items():
        out.append("\n## " + title + "\n")
        head = ["配置", "n"] + COLS
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
        for label, items in groups:
            rows = []
            for _seed, path in items:
                if os.path.exists(path):
                    rows.append(parse(path))
                else:
                    sys.stderr.write("缺日志: %s (%s)\n" % (path, label))
            vals = {c: [float(r[c]) for r in rows if r.get(c, "-") not in ("-", "")] for c in COLS}
            cells = [label, str(len(rows))] + [fmt(vals[c]) for c in COLS]
            out.append("| " + " | ".join(cells) + " |")
        # 逐 seed 原值(审计用)
        out.append("\n<details><summary>逐 seed 原值</summary>\n")
        out.append("| 配置 | seed | " + " | ".join(COLS) + " |")
        out.append("|" + "---|" * (len(COLS) + 2))
        for label, items in groups:
            for seed, path in items:
                if not os.path.exists(path):
                    continue
                r = parse(path)
                out.append("| %s | %s | " % (label, seed) +
                           " | ".join(r.get(c, "-") for c in COLS) + " |")
        out.append("\n</details>\n")
    dst = os.path.join(NIGHT, "night_tables.md")
    with open(dst, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("写出 %s (%d 行)" % (dst, len(out)))


if __name__ == "__main__":
    main()
