#!/usr/bin/env python3
"""cw_log_parse.py — 从 xCAD 训练日志的 final-test 块抽取"类别加权臂"需要的全部读数。

口径纪律（与 §31/§37/§38 一致）：
  * 只读**最终 test** 那一块。做法：整份日志按 "MRR (raw_ent):" 切块，取**最后一块**含
    "Per-GPU Hits@1 Macro" 的 —— 训练循环里每 evaluate-every 轮打印的 eval 块一概不读。
  * 判错块会被逐类 n 断言拦住：final test 必须是 5032/5246/552/811/195（n=11836，与主表
    同一把尺子）；训练循环里那份是 valid 口径（总数 12,600），逐类值不同 ⇒ 直接拒读，
    不静默用错数。
"""
import os
import re

N_EXPECT = {"T4": 5032, "MISC": 5246, "P100": 552, "V100": 811, "V100M32": 195}
BLOCK = re.compile(r"^MRR \(raw_ent\):", re.M)
PERCLS = re.compile(r"^\s*(\w+):\s+Hits@1=([0-9.]+)\s+\(n=\s*(\d+)\)", re.M)
MACRO = "Per-GPU Hits@1 Macro (unweighted)"


def _one(pattern, blk, cast=float):
    m = re.search(pattern, blk)
    return cast(m.group(1)) if m else None


def final_block(path):
    """返回 (final-test 块, 全文)。日志没有可识别的块 ⇒ (None, 全文)。"""
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    starts = [m.start() for m in BLOCK.finditer(txt)]
    if not starts:
        return None, txt
    chunks = [txt[starts[i]: starts[i + 1] if i + 1 < len(starts) else len(txt)]
              for i in range(len(starts))]
    cand = [c for c in chunks if MACRO in c]
    return (cand[-1] if cand else None), txt


def parse(path):
    """返回 dict；日志缺失/无 final-test 块/逐类 n 不符 ⇒ {'ok': False, 'why': ...}。"""
    if not os.path.exists(path):
        return {"ok": False, "why": "日志不存在"}
    blk, full = final_block(path)
    if blk is None:
        return {"ok": False, "why": "无含 macro5 行的 final test 块"}
    per = {k: (float(v), int(n)) for k, v, n in PERCLS.findall(blk)}
    if set(per) != set(N_EXPECT):
        return {"ok": False, "why": f"逐类行不全: {sorted(per)}"}
    for k, n in N_EXPECT.items():
        if per[k][1] != n:
            return {"ok": False, "why": f"{k} n={per[k][1]} != {n}（不是主表那把尺子，拒读）"}
    r1 = re.search(r"r1_suits\s+MRR=([0-9.]+)\s+H@1=([0-9.]+)\s+H@3=([0-9.]+)\s+H@10=([0-9.]+)", blk)
    return {
        "ok": True,
        "path": path,
        "macro5": _one(re.escape(MACRO) + r" = ([0-9.]+)", blk),
        "per": {k: per[k][0] for k in per},
        "r1_mrr": float(r1.group(1)) if r1 else None,
        "r1_h1": float(r1.group(2)) if r1 else None,
        "r1_h3": float(r1.group(3)) if r1 else None,
        "r1_h10": float(r1.group(4)) if r1 else None,
        "inseg_h1": _one(r"GPU 段内\(filtered\) H@1 = ([0-9.]+)", blk),
        # 表 4 最后一列"top-1 落在 GPU 候选段"是这个比例，不是 inseg_h1。
        "in_seg_ratio": _one(r"无约束 top-1 落在 GPU 段的比例 = ([0-9.]+)", blk),
        "inseg_macro5": _one(r"段内 per-GPU macro5 = ([0-9.]+)", blk),
        "compat_macro5": _one(r"compat 头 per-GPU macro5 = ([0-9.]+)", blk),
        "agree": _one(r"compat argmax 与 模型段内 top-1 是同一个 GPU 的边 = ([0-9.]+)", blk),
        "mrr_raw_ent": _one(r"MRR \(raw_ent\): ([0-9.]+)", blk),
        "h1_raw_ent": _one(r"Hits \(raw_ent\) @ 1: ([0-9.]+)", blk),
        # 主表表 2 的"总 MRR / 总 H@1"两列取的是 **filter_ent** 口径（见
        # docs/reviews/2026-11-17_idspace_main_table.md §3 逐 seed 原始值 0.065962/0.059156），
        # 不是 raw_ent。两口径在本文图上差约 0.6 个点，混用会让 ours 行与基线行不同尺子。
        "mrr_filter_ent": _one(r"MRR \(filter_ent\): ([0-9.]+)", blk),
        "h1_filter_ent": _one(r"Hits \(filter_ent\) @ 1: ([0-9.]+)", blk),
        "best_epoch": (int(re.findall(r"Using best epoch\s*:\s*(\d+)", full)[-1])
                       if re.findall(r"Using best epoch\s*:\s*(\d+)", full) else None),
        "n_blocks": len([1 for m in BLOCK.finditer(full)]),
    }


def class_weight_lines(path):
    """[class-weight] 打印发生在建图阶段（final-test 块之前），必须全文扫。"""
    if not os.path.exists(path):
        return []
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    return [ln.strip() for ln in txt.splitlines() if ln.startswith("[class-weight]")]
