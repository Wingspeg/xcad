#!/usr/bin/env python3
"""summarize_night_logs.py — 把无人值守队列的每个臂日志钉成一行读数。

口径与 docs/reviews/2026-11-17_idspace_main_table.md §1 第 4 条一致:
**只取最后一次 "Load Model name: ... Using best epoch : N" 之后的打印**(= final test 块),
训练循环里的 eval 块一概不读(那份类频是 12,600 那一套, 不是 11,836)。

用法: python summarize_night_logs.py <日志目录>
输出: <目录>/night_summary.tsv 和 .md; 缺字段打 "-"(不插值、不猜)。
"""
import os
import re
import sys

# (列名, 匹配模式, 在"最后一次 Load Model name 之后"的文本里取第几组)
FIELDS = [
    ("best_epoch", r"Using best epoch\s*:\s*(\d+)"),
    ("mrr_f",      r"MRR \(filter_ent\):\s*([0-9.]+)"),
    ("h1_f",       r"Hits \(filter_ent\) @ 1:\s*([0-9.]+)"),
    ("h3_f",       r"Hits \(filter_ent\) @ 3:\s*([0-9.]+)"),
    ("h10_f",      r"Hits \(filter_ent\) @ 10:\s*([0-9.]+)"),
    ("r1_mrr",     r"r1_suits\s+MRR=([0-9.]+)"),
    ("r1_h1",      r"r1_suits\s+MRR=[0-9.]+\s+H@1=([0-9.]+)"),
    ("macro5",     r"Per-GPU Hits@1 Macro \(unweighted\)\s*=\s*([0-9.]+)"),
    ("place_mrr",  r"placement\s+MRR=([0-9.]+)"),
    ("place_h1",   r"placement\s+MRR=[0-9.]+\s+H@1=([0-9.]+)"),
    ("r2_h1",      r"r2_requires\s+MRR=[0-9.]+\s+H@1=([0-9.]+)"),
    ("in_seg",     r"无约束 top-1 落在 GPU 段的比例\s*=\s*([0-9.]+)"),
    ("seg_h1",     r"GPU 段内\(filtered\) H@1\s*=\s*([0-9.]+)"),
    ("seg_macro",  r"段内 per-GPU macro5\s*=\s*([0-9.]+)"),
    ("compat_acc", r"compat 头直读.*acc=([0-9.]+)"),
    ("T4",         r"T4=([0-9.]+)/\d+"),
    ("MISC",       r"MISC=([0-9.]+)/\d+"),
    ("P100",       r"P100=([0-9.]+)/\d+"),
    ("V100",       r"V100=([0-9.]+)/\d+"),
    ("V100M32",    r"V100M32=([0-9.]+)/\d+"),
]

# 分层表里按"行"取的档: 用行关键字 + "读数行"必须同时含 H@1=, 免得抓到说明行。
# v2020 的标签是"复发 head / 全新 head", maas 是"head 在 history 以 rel0 复发 / 从未出现 /
# 边 (head,tail) 在 history 里" —— 两套都留, 缺的打 "-"(maas 主表用的 0.7364 是"边复发"那一档)。
STRATA = [
    ("all_h1",      ("全部 test 边",              r"H@1=([0-9.]+)")),
    ("rec_h1",      ("复发",                      r"H@1=([0-9.]+)")),
    ("new_h1",      ("全新",                      r"H@1=([0-9.]+)")),
    ("never_h1",    ("从未出现",                  r"H@1=([0-9.]+)")),
    ("rec_edge_h1", ("边 (head,tail) 在 history", r"H@1=([0-9.]+)")),
]


def final_block(text):
    """返回最后一次 'Load Model name' 之后的文本; 没有则退回全文末尾 400 行。"""
    idx = text.rfind("Load Model name")
    if idx < 0:
        return "\n".join(text.splitlines()[-400:])
    return text[idx:]


def parse(path):
    with open(path, "rb") as fh:
        blk = final_block(fh.read().decode("utf-8", "replace").replace("\r", "\n"))
    row = {}
    for col, pat in FIELDS:
        m = re.search(pat, blk)
        row[col] = m.group(1) if m else "-"
    lines = blk.splitlines()
    for col, (key, pat) in STRATA:
        val = "-"
        for ln in lines:
            # 必须是"读数行"(同时含关键字和 H@1=), 分层口径说明行里也提到复发/全新
            if key in ln and "H@1=" in ln:
                m = re.search(pat, ln)
                if m:
                    val = m.group(1)
                    break
        row[col] = val
    m = re.search(r"Sanity Check: stat name\s*:\s*(\S+)", blk)
    row["stat_name"] = m.group(1) if m else "-"
    m = re.search(r"gpu_base=(\d+)", blk)
    row["gpu_base"] = m.group(1) if m else "-"
    return row


def main(log_dir):
    names = sorted(f for f in os.listdir(log_dir) if f.endswith(".log"))
    rows = [{"tag": f[:-4], **parse(os.path.join(log_dir, f))} for f in names]
    cols = ["tag"] + [c for c, _ in FIELDS] + [c for c, _ in STRATA] + ["gpu_base", "stat_name"]

    with open(os.path.join(log_dir, "night_summary.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "-")) for c in cols) + "\n")

    keep = ["tag", "best_epoch", "mrr_f", "h1_f", "r1_mrr", "r1_h1", "macro5",
            "rec_h1", "new_h1", "in_seg", "seg_h1", "compat_acc", "gpu_base"]
    with open(os.path.join(log_dir, "night_summary.md"), "w") as fh:
        fh.write("# 2026-11-18 夜间队列读数(每行 = 一次 final test, ×1 原值)\n\n")
        fh.write("口径: `xcad_v2` 全局四段编号(gpu_base=124885)/ 全 124,891 实体 filtered / "
                 "final test 块(r1 正向边 11,836)。maas 臂没有 GPU 段报表, 相应列打 `-`。"
                 "`in_seg` = r1 的 top-1 落在 6 张卡内的比例(骨架行的候选段逃逸就看这一列)。\n\n")
        fh.write("| " + " | ".join(keep) + " |\n")
        fh.write("|" + "---|" * len(keep) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(r.get(c, "-")) for c in keep) + " |\n")
        fh.write("\n全列(逐类 H@1、placement、r2、段内 macro)见 `night_summary.tsv`。\n")

    print("臂数 = %d  -> %s/night_summary.{tsv,md}" % (len(rows), log_dir))
    bad = [r["tag"] for r in rows if r["h1_f"] == "-"]
    if bad:
        print("!! 没有 final test 指标的臂(要人看日志): %s" % ", ".join(bad))
    legacy = [r["tag"] for r in rows if r["gpu_base"] not in ("-", "124885")]
    if legacy:
        print("!! 编号口径不是 GLOBAL(gpu_base!=124885)的臂: %s" % ", ".join(legacy))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/xcad/outputs/reports/night_2026_11_18"))
