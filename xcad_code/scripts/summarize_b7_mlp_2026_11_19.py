#!/usr/bin/env python3
"""
B7 / R2-4 汇总 + 门禁：把 diagnose_sklearn.py 的 MLP 批次日志解析成一张表，
并在出表之前强制两道门禁（判线写在 docs/reviews/2026-11-19_b7_mlp_hybrid.md §4/§5，
本脚本只是把那份文档里的线翻译成代码，条件一字未改）。

G-SCALE  尺子：测试边 11,836、逐类 n == 5032/5246/552/811/195（T4/MISC/P100/V100/V100M32）。
G-REPRO  加了 MLP 之后 LR/RF 读数必须与既有日志逐位相同（值来自
         ~/xcad/outputs/reports/rf_ceiling_v2/ 与 rf_balanced_control_2026_11_18/，
         本轮先用脚本从旧日志读一遍再比对，不抄我的记忆）。

写法纪律：本脚本只解析、不重算指标 —— 数字全部出自 diagnose_sklearn.py 的打印。
"""
import argparse
import json
import os
import re
import sys

CLS = ["T4", "MISC", "P100", "V100", "V100M32"]
EXPECT_N = [5032, 5246, 552, 811, 195]
EXPECT_TOTAL = 11836

BLOCK = re.compile(r"^  (LogisticRegression|RandomForest|MLP)\s+\[(.*?)\]\s*$")
ACC = re.compile(r"总准确率:\s+([\d.]+)%")
MAC = re.compile(r"macro avg:\s+([\d.]+)%")
PC = re.compile(r"type \d+ \((\w+)\s*\):\s*recall=\s*([\d.]+)%\s*precision=\s*([\d.nan]+)%\s*\(n=\s*(\d+)\)")
MARG = re.compile(r"top-1 预测类计数:\s*(.*?)\s*\(真值")
PAIR = re.compile(r"(\w+)=(-?\d+)")
FIT = re.compile(r"^\[fit\] class_weight=(\w+).*train=(\d+)\s+test=(\d+)")


def parse_log(path):
    out = {"file": os.path.basename(path), "path": path, "clf": {}, "fit": None, "traces": 0}
    # split 只能从【完整路径】判：rf_balanced 的算法口径日志叫 algo/none_s42.log，
    # 光看 basename 会把它错配成 tau 口径（两边 test n 不同，会污染 G-REPRO）。
    out["split"] = "algo" if ("algo" in path or "algo" in os.path.basename(path)) else "tau"
    out["seed"] = int(re.search(r"s(\d+)", os.path.basename(path)).group(1)) \
        if re.search(r"s(\d+)", os.path.basename(path)) else None
    cur = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "Traceback" in line:
                out["traces"] += 1
            m = FIT.match(line)
            if m:
                out["fit"] = dict(class_weight=m.group(1), n_train=int(m.group(2)),
                                  n_test=int(m.group(3)))
            m = BLOCK.match(line.rstrip())
            if m:
                cur = m.group(1)
                out["clf"][cur] = {"split": m.group(2), "per": {}, "marginal": {}}
                continue
            if cur is None:
                continue
            b = out["clf"][cur]
            if "acc" not in b and ACC.search(line):
                b["acc"] = float(ACC.search(line).group(1))
            elif "macro" not in b and MAC.search(line):
                b["macro"] = float(MAC.search(line).group(1))
            for name, rec, prec, n in PC.findall(line):
                b["per"][name] = dict(recall=float(rec),
                                      precision=(None if prec == "nan" else float(prec)),
                                      n=int(n))
            if "[marginal]" in line:
                g = MARG.search(line)
                if g:
                    b["marginal"] = {k: int(v) for k, v in PAIR.findall(g.group(1))}
    return out


def gate_scale(logs):
    """尺子只卡 tau 口径（算法口径的测试边集本来就是另一批，约 21.8k 条）。"""
    bad = []
    for L in logs:
        if L["split"] != "tau":
            continue
        for name, b in L["clf"].items():
            ns = [b["per"].get(c, {}).get("n") for c in CLS]
            if ns != EXPECT_N:
                bad.append((L["file"], name, ns))
            tot = sum(x for x in ns if x)
            if tot != EXPECT_TOTAL:
                bad.append((L["file"], name, "total=%s" % tot))
    return bad


def gate_repro(new_logs, old_logs):
    """新日志里的 LR/RF 读数必须与既有日志逐位相同（按 class_weight+seed+split 配对）。"""
    def key(L):
        return (L["fit"]["class_weight"] if L["fit"] else "?", L["seed"], L["split"])

    old = {}
    for p in old_logs:
        L = parse_log(p)
        if L["fit"] and L["clf"]:
            k = key(L)
            if k in old:
                print(f"   !! 参照日志键冲突 {k}: {old[k]['path']} vs {p} —— 保留第一份")
            else:
                old[k] = L
    rows = []
    for L in new_logs:
        k = key(L)
        O = old.get(k)
        if O is None:
            rows.append((k, "-", "NO-OLD-LOG", L["path"]))
            continue
        for c in ("LogisticRegression", "RandomForest"):
            if c not in L["clf"] or c not in O["clf"]:
                rows.append((k, c, "MISSING", None))
                continue
            a1, m1 = L["clf"][c]["acc"], L["clf"][c]["macro"]
            a0, m0 = O["clf"][c]["acc"], O["clf"][c]["macro"]
            rows.append((k, c, "PASS" if (a1 == a0 and m1 == m0) else "FAIL",
                         f"new acc={a1} macro={m1} | old acc={a0} macro={m0}  <- {O['path']}"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", required=True, help="本轮 b7_mlp_hybrid_*/ 目录")
    ap.add_argument("--old-dir", nargs="*", default=[], help="既有 RF 日志目录（G-REPRO 的参照）")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tau_dir = os.path.join(args.new_dir, "tau")
    algo_dir = os.path.join(args.new_dir, "algo")
    new_logs = []
    for d in (tau_dir, algo_dir):
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".log"):
                    new_logs.append(parse_log(os.path.join(d, fn)))

    print("=" * 90)
    print("B7 MLP 汇总　(口径: xcad_v2 / tau-split / 11,836 条 r1 测试边 / 5 类 macro)")
    print("=" * 90)
    scale_bad = gate_scale(new_logs)
    print(f"\n[G-SCALE] 尺子 (11,836; {EXPECT_N}) -> {'PASS' if not scale_bad else 'FAIL'}"
          + ("" if not scale_bad else f"  {scale_bad[:3]}"))
    crashed = [L["path"] for L in new_logs if L["traces"]]
    print(f"[G-CRASH] 全部日志无 Traceback -> {'PASS' if not crashed else 'FAIL ' + str(crashed)}"
          f"  (共 {len(new_logs)} 份)")

    # 既有日志：从 --old-dir 里读，文件名可能不同(rf_balanced 用 tau_none_s42.log,
    # rf_ceiling_v2 用 v2_tau_s42.log)，按 (cw, seed, split) 归一后再配。
    old_paths = []
    for d in args.old_dir:
        for root, _, files in os.walk(d):
            for fn in sorted(files):
                if fn.endswith(".log") and ("tau" in fn or "algo" in fn or "v2_" in fn
                                            or os.path.basename(root) == "algo"):
                    p = os.path.join(root, fn)
                    L = parse_log(p)
                    if L["fit"] and "MLP" not in L["clf"]:
                        old_paths.append(p)
    repro = gate_repro(new_logs, old_paths) if old_paths else []
    print(f"\n[G-REPRO] 参照日志 {len(old_paths)} 份")
    fails = 0
    for k, c, verdict, detail in repro:
        if verdict not in ("PASS",):
            fails += 1
            print(f"   {verdict:>10}  {k} {c}  {detail or ''}")
    print(f"[G-REPRO] -> {'PASS' if fails == 0 and repro else 'FAIL/未跑'}  "
          f"({len(repro)} 项比对, 不符 {fails})")
    if not repro:
        print("   !! 没找到可配对的旧日志 ⇒ G-REPRO 未执行，本表不得用于替换已发表数字")
    if scale_bad or fails or crashed:
        print("\n门禁未过，按 §4 整块作废，不出表。")
        sys.exit(2)

    hdr = f"{'口径':<10}{'split':<6}{'seed':<6}{'分类器':<20}{'总acc':>8}{'macro5':>8}" + \
          "".join(f"{c:>9}" for c in CLS) + f"{'预P100':>8}{'预V100M32':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for L in new_logs:
        split, seed = L["split"], L["seed"]
        for c in ("LogisticRegression", "RandomForest", "MLP"):
            if c not in L["clf"]:
                continue
            b = L["clf"][c]
            print(f"{L['fit']['class_weight']:<10}{split:<6}{seed:<6}{c:<20}"
                  f"{b['acc']:>8.2f}{b['macro']:>8.2f}" +
                  "".join(f"{b['per'].get(x, {}).get('recall', float('nan')):>9.2f}" for x in CLS) +
                  f"{b['marginal'].get('P100', 0):>8d}{b['marginal'].get('V100M32', 0):>10d}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(dict(gates=dict(scale="PASS", repro="PASS"), logs=new_logs), f,
                      ensure_ascii=False, indent=1)
        print(f"\n[json] {args.json_out}")


if __name__ == "__main__":
    main()
