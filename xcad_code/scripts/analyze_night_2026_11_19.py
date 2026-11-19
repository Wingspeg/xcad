#!/usr/bin/env python3
# @File    : analyze_night_2026_11_19.py
"""night_2026_11_19 队列读数解析 + 预注册判线。

三条纪律（与 §31/§37/§41 一致）：
  1. 只读 final-test 块（cw_log_parse 的逐类 n 断言 = 5032/5246/552/811/195 兜底）。
  2. 比较前先跑 G-CAL：把**已发表**的读数从对应日志里重新解析出来，逐位对上才继续；
     对不上说明解析器与手稿不同尺子，本轮所有比较作废。
  3. 判线在看到数字前已写死，见 docs/reviews/2026-11-18_overnight_plan.md §1。

用法：
  ~/xcad/xcad_code/.venv/bin/python xcad_code/scripts/analyze_night_2026_11_19.py \
      --out-dir xcad_code/outputs/reports/night_2026_11_19
"""
import argparse
import json
import os
import re
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.join(_HERE, "..", "xcad_model", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
import cw_log_parse as C  # noqa: E402

R = os.path.expanduser("~/xcad/outputs/reports")
NIGHT = os.path.join(R, "night_2026_11_19")
BL_ID = os.path.join(R, "baselines_idspace")

# ── 已发表读数（docs/reviews/2026-11-17_idspace_main_table.md §3，用于 G-CAL）──
PUB_OURS = {
    42:   dict(mrr=0.065962, h1=0.059156, r1_mrr=0.927251, r1_h1=0.884843, macro=0.5085),
    123:  dict(mrr=0.066374, h1=0.059459, r1_mrr=0.923450, r1_h1=0.882308, macro=0.5069),
    2024: dict(mrr=0.065321, h1=0.058699, r1_mrr=0.924668, r1_h1=0.881801, macro=0.5069),
}
PUB_BASE = {
    "RE-GCN": dict(mrr=0.0502, h1=0.0300, r1_mrr=0.6904, r1_h1=0.4412, macro=0.2029),
    "CEN":    dict(mrr=0.0490, h1=0.0290, r1_mrr=0.6827, r1_h1=0.4251, macro=0.2000),
    "RETIA":  dict(mrr=0.0473, h1=0.0279, r1_mrr=0.6801, r1_h1=0.4251, macro=0.2000),
}
# 表 2 ours 三 seed 均值（判"变体是否掉出三种子带"用）
OURS_MEAN_MACRO = statistics.mean(v["macro"] for v in PUB_OURS.values())
OURS_SPAN_MACRO = max(v["macro"] for v in PUB_OURS.values()) - min(v["macro"] for v in PUB_OURS.values())

# v2020 run-to-run 噪声（§30/§42：口径是 r1 Hits@1）
NOISE_R1_H1 = 0.0172
NOISE_FLOOR = 0.02          # 低于此不许当差异说话


def get(path):
    """解析一臂的 final-test 读数。

    不直接用 C.parse：CEN 的 test-only 路径**只打印 filter_ent，没有 raw_ent**，
    而 cw_log_parse 是按 `^MRR (raw_ent):` 切块的 ⇒ 对 CEN test 日志必然失败。
    这里改成两种行首都认，其余纪律（最后一块、逐类 n 断言、filter_ent 口径）与
    cw_log_parse 完全一致，并保留对 C.parse 成功结果的复用。
    """
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    starts = [m.start() for m in re.finditer(r"^MRR \((?:raw|filter)_ent\):", txt, re.M)]
    if not starts:
        return None
    chunks = [txt[starts[i]: starts[i + 1] if i + 1 < len(starts) else len(txt)]
              for i in range(len(starts))]
    cand = [c for c in chunks if C.MACRO in c]
    if not cand:
        return None
    blk = cand[-1]
    per = {k: (float(v), int(n)) for k, v, n in C.PERCLS.findall(blk)}
    if set(per) != set(C.N_EXPECT):
        return None
    for k, n in C.N_EXPECT.items():
        if per[k][1] != n:      # 不是主表那把尺子(valid 口径) ⇒ 拒读
            return None
    r1 = re.search(r"r1_suits\s+MRR=([0-9.]+)\s+H@1=([0-9.]+)\s+H@3=([0-9.]+)\s+H@10=([0-9.]+)", blk)
    best = re.findall(r"Using best epoch\s*:\s*(\d+)", txt)
    return dict(
        mrr=C._one(r"MRR \(filter_ent\): ([0-9.]+)", blk),
        h1=C._one(r"Hits \(filter_ent\) @ 1: ([0-9.]+)", blk),
        r1_mrr=float(r1.group(1)) if r1 else None,
        r1_h1=float(r1.group(2)) if r1 else None,
        macro=C._one(re.escape(C.MACRO) + r" = ([0-9.]+)", blk),
        per={k: per[k][0] for k in per},
        best_epoch=int(best[-1]) if best else None,
        n_blocks=len(starts),
    )


def close(a, b, tol=5e-7):
    return a is not None and b is not None and abs(a - b) <= tol


def gcal():
    """门禁：从既有日志重解已发表读数，逐位对上才允许本轮比较。"""
    rep = {"ok": True, "rows": []}
    # ours 三个 shipped seed
    shipped_paths = {
        42: os.path.join(NIGHT, "p0_regression.log"),
        123: os.path.join(BL_ID, "v2_joint_s123.log"),
        2024: os.path.join(BL_ID, "v2_joint_s2024.log"),
    }
    for s, p in shipped_paths.items():
        g = get(p)
        exp = PUB_OURS[s]
        if g is None:
            rep["rows"].append(f"  seed{s}: 解析失败 {p}")
            rep["ok"] = False
            continue
        bad = [k for k in ("mrr", "h1", "r1_mrr", "r1_h1", "macro") if not close(g[k], exp[k], 1e-6)]
        rep["rows"].append(f"  ours s{s:<5} {'OK ' if not bad else 'MISMATCH ' + str(bad)}"
                           f" mrr={g['mrr']} h1={g['h1']} r1mrr={g['r1_mrr']} r1h1={g['r1_h1']} macro={g['macro']}")
        if bad:
            rep["ok"] = False
    # 基线 run-1
    # 注意 CEN：v2_cen_test.log 是**失败的首跑**（无 final-test 块），发表读数出自重试的
    # v2_cen_test_r2.log（0.049006/0.682748/0.425144/macro 0.2000，与表 2 的 4.90/68.27/42.51/20.00 逐位一致）。
    # CEN 一次运行 = pretrain + curriculum + **test-only 步**，表 2 取的是 test 步（curriculum 步
    # 打印的是 0.691788/0.443224，不同尺子），所以下面一律用 *_test 日志。
    base_paths = {"RE-GCN": os.path.join(BL_ID, "v2_regcn.log"),
                  "CEN": os.path.join(BL_ID, "v2_cen_test_r2.log"),
                  "RETIA": os.path.join(BL_ID, "v2_retia_curtrain.log")}
    for m, p in base_paths.items():
        g = get(p)
        if g is None:
            rep["rows"].append(f"  {m}: 解析失败 {p}")
            rep["ok"] = False
            continue
        # 表 2 的基线行只精确到 ×100 后两位 ⇒ 容差 5e-5
        exp = PUB_BASE[m]
        bad = [k for k in ("mrr", "r1_mrr", "r1_h1", "macro") if not close(g[k], exp[k], 5e-5)]
        rep["rows"].append(f"  {m:<7} {'OK ' if not bad else 'MISMATCH ' + str(bad)}"
                           f" mrr={g['mrr']:.4f} r1h1={g['r1_h1']:.4f} macro={g['macro']:.4f}"
                           f" (表2 {exp['mrr']:.4f}/{exp['r1_h1']:.4f}/{exp['macro']:.4f})")
        if bad:
            rep["ok"] = False
    rep["shipped_paths"] = shipped_paths
    rep["base_paths"] = base_paths
    return rep


def b2():
    """归一化口径变体 vs shipped，逐 seed 配对。"""
    out = {"arms": {}, "delta": {}, "pairs": {}}
    for s in (42, 123, 2024):
        p = os.path.join(NIGHT, f"b2_tc_s{s}.log")
        g = get(p)
        out["arms"][s] = g
        if g is None:
            out.setdefault("failed", []).append(s)
            continue
        exp = PUB_OURS[s]
        d = {k: g[k] - exp[k] for k in ("mrr", "h1", "r1_mrr", "r1_h1", "macro")}
        out["delta"][s] = d
        out["pairs"][s] = dict(shipped=exp, variant={k: g[k] for k in ("mrr", "h1", "r1_mrr", "r1_h1", "macro")},
                               delta=d)
    ds = [v["macro"] for v in out["delta"].values()]
    if ds:
        out["macro_delta_mean"] = statistics.mean(ds)
        out["macro_delta_min"] = min(ds)
        out["macro_delta_max"] = max(ds)
        out["same_sign"] = all(x > 0 for x in ds) or all(x < 0 for x in ds)
    # 判线（事前写死）
    if ds:
        md = out["macro_delta_mean"]
        if abs(md) <= NOISE_FLOOR:
            out["line"] = ("|Δmacro| ≤ 0.02 ⇒ 该维度可表述为『两个非因果要素里已测的那个，"
                           "实测不改变结论（≤噪声）』")
        elif md < -NOISE_FLOOR:
            out["line"] = "Δ < −0.02 ⇒ 必须改正文 §3.1/§6 T2 并如实报掉多少"
        else:
            out["line"] = "Δ > +0.02 ⇒ 变体更好，需如实报并检查是否口径引入新信息"
    return out


def b3():
    """外部基线三次运行：极差判线。"""
    runs = {
        "RE-GCN": [os.path.join(BL_ID, "v2_regcn.log"),
                   os.path.join(NIGHT, "b3_regcn_r2.log"),
                   os.path.join(NIGHT, "b3_regcn_r3.log")],
        # CEN 一次运行 = pretrain + curriculum + test-only，读数取 test 步
        "CEN": [os.path.join(BL_ID, "v2_cen_test_r2.log"),
                os.path.join(NIGHT, "b3_cen_r2_test.log"),
                os.path.join(NIGHT, "b3_cen_r3_test.log")],
        "RETIA": [os.path.join(BL_ID, "v2_retia_curtrain.log"),
                  os.path.join(NIGHT, "b3_retia_r2.log"),
                  os.path.join(NIGHT, "b3_retia_r3.log")],
    }
    out = {}
    for m, ps in runs.items():
        gs = []
        for p in ps:
            g = get(p)
            if g is None:
                out[m] = {"error": f"解析失败 {p}"}
                break
            gs.append(dict(path=os.path.basename(p), **{k: g[k] for k in ("mrr", "h1", "r1_mrr", "r1_h1", "macro")},
                           per=g["per"], best_epoch=g["best_epoch"]))
        else:
            rng = {k: max(x[k] for x in gs) - min(x[k] for x in gs)
                   for k in ("mrr", "h1", "r1_mrr", "r1_h1", "macro")}
            sd = {k: (statistics.stdev([x[k] for x in gs]) if len(gs) > 1 else 0.0)
                  for k in ("mrr", "h1", "r1_mrr", "r1_h1", "macro")}
            out[m] = dict(runs=gs, range=rng, sd=sd,
                          verdict=("极差 ≤0.02 ⇒ '常数预测器的 macro 对种子不变' 可升级为实测"
                                   if rng["macro"] <= NOISE_FLOOR else
                                   f"macro 极差 {rng['macro']:.4f} >0.02 ⇒ 撤那句，表 2 基线行改 mean±sd"),
                          # macro 恰好等于常数猜测器 0.2000 的臂 = 退化预测器，可主张"只命中一个高频类"
                          collapsed=[bool(max(x["per"].values()) >= 0.999 and
                                            sum(1 for v in x["per"].values() if v > 0.02) == 1)
                                     for x in gs])
    return out


def b4():
    """代价实测：墙钟来自 queue.out；显存本轮无轮询 ⇒ 明确报缺口。

    口径：CEN 的"一次运行"= pretrain + curriculum + test-only 三步之和（与表 2 的
    "一行一次运行"同源），所以按模型汇总时三步要相加，不能只算最后那 30 秒。
    """
    q = os.path.join(NIGHT, "queue.out")
    if not os.path.exists(q):
        return {"error": "queue.out 不存在"}
    txt = open(q, errors="ignore").read()
    # queue.out 的行形如 "[12:17:38] b2_tc_s42 done   rc=0 用时=860s  final-test块=6"
    # 方括号里是**时间戳**不是 tag，tag 在它后面。
    dur = {t: int(s) for t, s in re.findall(
        r"\[\d\d:\d\d:\d\d\]\s+(\S+)\s+done\s+rc=\d+\s+用时=(\d+)s", txt)}

    def grp(prefixes):
        return sum(v for k, v in dur.items() if any(k.startswith(p) for p in prefixes))

    per_model = {
        "xCAD(ours, 变体口径)": grp(["b2_"]),
        "RE-GCN": grp(["b3_regcn"]),
        "CEN(3 步 = 1 次运行)": grp(["b3_cen"]),
        "RETIA": grp(["b3_retia"]),
    }
    # ours 每 epoch 墙钟：30 epoch 训练 + 6 次评测
    per_epoch = {}
    for tag in sorted(k for k in dur if k.startswith("b2_")):
        p = os.path.join(NIGHT, f"{tag}.log")
        if os.path.exists(p):
            per_epoch[tag] = dur[tag] / 30.0
    return {"durations_s": dur,
            "per_model_s": per_model,
            "ours_wallclock_per_epoch_s": per_epoch,
            "total_gpu_s": sum(dur.values()),
            "testonly_wallclock_s": {"P0 回归(旗舰 test-only)": 30, "B1 dump-preds": 29},
            "note": "显存峰值本轮未采样（脚本没挂轮询）⇒ B4 只能报墙钟，显存需另测一次 test-only。"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=NIGHT)
    args = ap.parse_args()
    out = {}
    out["gcal"] = gcal()
    print("==== G-CAL 门禁（先证明解析器与手稿同尺子）====")
    for ln in out["gcal"]["rows"]:
        print(ln)
    print(f"G-CAL: {'PASS' if out['gcal']['ok'] else 'FAIL'}")
    if not out["gcal"]["ok"]:
        print("!! 解析器与已发表读数不同口径，本轮比较作废")
        sys.exit(3)

    out["b2"] = b2()
    print("\n==== B2 归一化口径（train+valid 实体拟合）vs shipped，逐 seed 配对 ====")
    print(f"{'seed':<6}{'总MRR':>10}{'总H@1':>10}{'r1MRR':>10}{'r1H@1':>10}{'macro5':>10}")
    for s, g in sorted(out["b2"]["arms"].items(), key=lambda kv: str(kv[0])):
        if g is None:
            print(f"{s:<6}  解析失败")
            continue
        d = out["b2"]["delta"][s]
        print(f"{s:<6}" + "".join(f"{g[k]:>10.6f}" for k in ("mrr", "h1", "r1_mrr", "r1_h1", "macro")))
        print(f"{'Δ':<6}" + "".join(f"{d[k]:>+10.6f}" for k in ("mrr", "h1", "r1_mrr", "r1_h1", "macro")))
    if "macro_delta_mean" in out["b2"]:
        print(f"\nΔmacro 均值={out['b2']['macro_delta_mean']:+.6f} "
              f"区间=[{out['b2']['macro_delta_min']:+.6f},{out['b2']['macro_delta_max']:+.6f}] "
              f"同号={out['b2']['same_sign']} | shipped 三种子跨度={OURS_SPAN_MACRO:.4f} 均值={OURS_MEAN_MACRO:.4f}")
        print(f"判线: {out['b2']['line']}")

    out["b3"] = b3()
    print("\n==== B3 外部基线三次运行 ====")
    for m, r in out["b3"].items():
        if "error" in r:
            print(f"{m}: {r['error']}")
            continue
        print(f"\n{m}  (退化单类预测器: {r['collapsed']})")
        for x in r["runs"]:
            print(f"  {x['path']:<26} 总MRR={x['mrr']:.6f} r1MRR={x['r1_mrr']:.4f} "
                  f"r1H@1={x['r1_h1']:.4f} macro={x['macro']:.4f} ep={x['best_epoch']}")
        print(f"  极差: " + " ".join(f"{k}={v:.4f}" for k, v in r["range"].items()))
        print(f"  sd  : " + " ".join(f"{k}={v:.4f}" for k, v in r["sd"].items()))
        print(f"  判线: {r['verdict']}")

    out["b4"] = b4()
    print("\n==== B4 代价（墙钟）====")
    for m, s in out["b4"].get("per_model_s", {}).items():
        print(f"  {m:<26} {s:>6}s = {s/60:.1f} min")
    print("  ours 变体臂 每 epoch 墙钟: " +
          " ".join(f"{t}={v:.1f}s" for t, v in out["b4"].get("ours_wallclock_per_epoch_s", {}).items()))
    print(f"  test-only 参考: {out['b4'].get('testonly_wallclock_s')}")
    print(f"  GPU 合计 = {out['b4'].get('total_gpu_s', 0)/3600:.2f} h")
    print(f"  ⚠ {out['b4'].get('note','')}")

    j = os.path.join(args.out_dir, "night_2026_11_19_readings.json")
    with open(j, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[json] {j}")


if __name__ == "__main__":
    main()
