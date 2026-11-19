#!/usr/bin/env python3
"""summarize_rf_balanced_2026_11_18.py — 平衡口径 RF 对照的读数与判线（0 GPU）。

要回答的唯一问题（用户 2026-11-18 放行"先跑平衡口径的 RF 对照"）：
  §39 的 inv 臂 seed42 macro5 = 0.6023 超过了 §38 写进正文的随机森林上界 0.582，
  而我自己在 §5.5 写过"方法应逼近而不应超过它，超过即提示信息泄漏"。
  ⇒ 那个 0.582 是【按 accuracy 拟合】的 RF。把 RF 改成【按平衡口径拟合】（class_weight='balanced'），
     同一份 13 维画像、同一批 11,836 条测试边、同 seed，看它的 macro 变成多少。

尺子（与 §38 / 主表逐字相同）：tau 切分，train+valid 拟合、test 评测，测试边集 11,836，
逐类 n 必须是 T4 5032 / MISC 5246 / P100 552 / V100 811 / V100M32 195，不符直接 FAIL 不出表。

判线（看数字前写死，2026-11-18 12:4x 本机写下，未改）：
  已知 A = accuracy 口径 RF macro = 58.21（三种子 58.21/58.18/58.25）
       O = ours inv 档 = 60.23(单种子最好) / 55.58(三种子均值)
       N = ours 控制组 = 50.74
  R-L1 口径确证：B(平衡口径均值) >= 60.0 ⇒ "超过上界"这句话本身是口径错配，正文必须换成同口径对照，
                 inv_s42 的 60.23 落在 B 之内 ⇒ 泄漏疑虑由这条排除。
  R-L2 上界不动：B <= 59.0 ⇒ 60.23 高于我们复现过的任何 RF 口径 ⇒ 正文【不许】出现"超过上界"，
                 只能写"高于按 accuracy 拟合的画像上界"，并说明该上界不是模型信息集的上界。
  R-L3 中间带：59.0 < B < 60.0 ⇒ 不写"超过"也不写"之内"，只写"同口径下差距缩到 …点"。
  停止线：B < 50.74 ⇒ 平衡口径 RF 低于不加权的我们的臂 ⇒ 这个对照回答不了泄漏问题，
                 必须另找口径（正文里"上界"整段降级为"画像可分性探针"）。
  机制线（与判线同等重要）：
    M1 平衡 RF 的 P100/V100M32 预测边际是否远超真值(552/195) 且精确率低 ⇒ 印证"macro 涨=摊边际"
    M2 平衡 RF 的总 acc 相对 88.97 掉多少 ⇒ 与模型侧 r1 H@1 下跌同构（同一动作的两面）
    M3 平衡 RF 的 P100 recall 相对 12.14 的变化 ⇒ 画像里稀有类信号的可分度上限
"""
import json
import os
import re
import sys

ROOT = os.path.expanduser("~/xcad/outputs/reports/rf_balanced_control_2026_11_18/")
EXPECT_N = {"T4": 5032, "MISC": 5246, "P100": 552, "V100": 811, "V100M32": 195}
A_KNOWN = 58.21          # §38 accuracy 口径，三种子均值
# §38 逐位读数，按 seed 索引（G-REPRO 必须逐 seed 对上，不按文件名字母序）
EXPECT_NONE = {"42": 58.21, "123": 58.18, "2024": 58.25}
O_INV_BEST, O_INV_MEAN, O_NONE_MEAN = 60.23, 55.58, 50.74

CLSFROM = re.compile(r"^  (LogisticRegression|RandomForest)\s+\[([a-z0-9-]+)\]", re.M)
MACRO = re.compile(r"macro avg:\s*([0-9.]+)%")
ACC = re.compile(r"总准确率:\s*([0-9.]+)%")
PERCLS = re.compile(r"\((\w+)\s*\): recall=\s*([0-9.]+)%\s+precision=\s*(nan|[0-9.]+)%\s+\(n=\s*(\d+)\)")
MARG = re.compile(r"\[marginal\] top-1 预测类计数:(.*?)\s+\(真值", re.S)
FIT = re.compile(r"\[fit\] class_weight=(\w+).*?train=(\d+)\s+test=(\d+)")


def model_block(txt, model):
    """返回该模型段落正文；找不到返回 None。"""
    for m in CLSFROM.finditer(txt):
        if m.group(1) == model:
            nxt = CLSFROM.search(txt, m.end())
            end = nxt.start() if nxt else txt.find("RF 特征重要性", m.end())
            if end < 0:
                end = len(txt)
            return txt[m.end():end]
    return None


def parse(path, model="RandomForest"):
    txt = open(path, errors="ignore").read()
    blk = model_block(txt, model)
    out = {"log": os.path.basename(path), "ok": False, "why": ""}
    if blk is None:
        out["why"] = "no %s block" % model
        return out
    fm = FIT.search(txt)
    out["class_weight"] = fm.group(1) if fm else "?"
    out["n_train"] = int(fm.group(2)) if fm else -1
    out["n_test"] = int(fm.group(3)) if fm else -1
    mm, am = MACRO.search(blk), ACC.search(blk)
    if not mm or not am:
        out["why"] = "no macro/acc line"
        return out
    out["macro"] = float(mm.group(1))
    out["acc"] = float(am.group(1))
    per = {}
    for name, rec, prec, n in PERCLS.findall(blk):
        per[name] = {"recall": float(rec),
                 "precision": float("nan") if prec == "nan" else float(prec),
                 "n": int(n)}
    out["per"] = per
    if sum(v["n"] for v in per.values()) != 11836 or \
       any(per.get(k, {}).get("n") != v for k, v in EXPECT_N.items()):
        out["why"] = "RULER MISMATCH: %s" % {k: per.get(k, {}).get("n") for k in EXPECT_N}
        return out
    marg = {}
    gm = MARG.search(blk)
    if gm:
        for tok in gm.group(1).replace("\n", " ").split():
            if "=" in tok:
                k, v = tok.split("=")
                marg[k] = int(v)
    out["marginal"] = marg
    out["ok"] = True
    return out


def main():
    files = sorted(f for f in os.listdir(ROOT)) if os.path.isdir(ROOT) else []
    if not files:
        print("FAIL  目录为空：%s" % ROOT)
        return 1
    rows = []
    problems = []
    for f in files:
        if not f.endswith(".log"):
            continue
        r = parse(os.path.join(ROOT, f))
        if not r["ok"]:
            problems.append("%s: %s" % (f, r["why"]))
            continue
        rows.append(r)

    print("==== 平衡口径 RF 对照（尺子 = 主表 11,836 条测试边）====")
    hdr = "log                              cw        RF macro  RF acc    T4      MISC    P100    V100    V100M32   预测P100 预测V100M32"
    print(hdr)
    for r in rows:
        p = r["per"]
        print("%-32s %-9s %8.2f %8.2f  %6.2f %6.2f %6.2f %6.2f %6.2f   %6d %6d" % (
            r["log"], r["class_weight"], r["macro"], r["acc"],
            p["T4"]["recall"], p["MISC"]["recall"], p["P100"]["recall"],
            p["V100"]["recall"], p["V100M32"]["recall"],
            r["marginal"].get("P100", -1), r["marginal"].get("V100M32", -1)))
    if problems:
        print("\n!! 不合格日志：")
        for x in problems:
            print("   ", x)

    def seed_of(name):
        m = re.search(r"s(\d+)", name)
        return m.group(1) if m else "?"

    def by_seed(cw):
        return {seed_of(r["log"]): round(r["macro"], 2)
                for r in rows if r["class_weight"] == cw and r["log"].startswith("tau_")}

    none_map, bal_map = by_seed("none"), by_seed("balanced")
    print("\n── 门禁与判线 ─────────────────────────────────────────────")
    got = [none_map.get(k) for k in EXPECT_NONE]
    want = [EXPECT_NONE[k] for k in EXPECT_NONE]
    gate_repro = got == want
    print("G-REPRO 默认路径逐 seed 复现 §38 : %s   seed{42,123,2024} 得到 %s 期望 %s"
          % ("PASS" if gate_repro else "FAIL -> 本轮读数全部作废", got, want))
    # G-PREC：打印的精确率必须与 (recall·真值n)/预测边际 自洽，防止把 recall 当 precision 印出去
    prec_problems = []
    for r in rows:
        for cls, v in r["per"].items():
            pred_n = r["marginal"].get(cls, 0)
            if pred_n <= 0:
                continue
            derived = 100.0 * (v["recall"] / 100.0) * v["n"] / pred_n
            if not (abs(derived - v["precision"]) <= 0.15 or v["precision"] != v["precision"]):
                prec_problems.append("%s %s printed=%.2f derived=%.2f"
                                     % (r["log"], cls, v["precision"], derived))
    print("G-PREC 精确率自洽检查 : %s" % ("PASS" if not prec_problems else
                                          "FAIL -> 逐类精确率不可用\n   " + "\n   ".join(prec_problems)))
    if not bal_map:
        print("没有 balanced 日志，判线不出。")
        return 1
    B = sum(bal_map.values()) / len(bal_map)
    print("accuracy 口径 B_none = %s   balanced 口径 B_bal = %.2f  %s"
          % (list(none_map.values()), B, list(bal_map.values())))
    if B < O_NONE_MEAN:
        line = "停止线：B < 50.74 ⇒ 该对照回答不了泄漏问题 ⇒ 正文『上界』整段降级为『画像可分性探针』"
    elif B >= 60.0:
        line = "R-L1 口径确证：B>=60.0 ⇒ 『超过上界』是口径错配；inv_s42=60.23 落在 B 之内 ⇒ 泄漏疑虑排除"
    elif B <= 59.0:
        line = "R-L2 上界不动：B<=59.0 ⇒ 正文不许出现『超过上界』，只写『高于按 accuracy 拟合的画像上界』"
    else:
        line = "R-L3 中间带：只写『同口径下差距缩到 … 点』，不写『超过』也不写『之内』"
    print("判线：%s" % line)
    print("对照量：A=58.21(accuracy 口径)  B=%.2f(平衡口径)  ours inv 60.23/55.58  ours none 50.74" % B)
    for r in rows:
        if r["class_weight"] == "balanced":
            p = r["per"]
            print("  [M1/M2/M3] %s  Δacc=%.2f  P100 recall %.2f (vs 12.14)  "
                  "P100 precision %.2f  V100M32 recall %.2f (vs 23.08)" % (
                      r["log"], r["acc"] - 88.97, p["P100"]["recall"],
                      p["P100"]["precision"], p["V100M32"]["recall"]))
    with open(os.path.join(ROOT, "readings.json"), "w") as fh:
        json.dump({"rows": rows, "gate_repro": gate_repro,
                   "prec_problems": prec_problems,
                   "none_by_seed": none_map, "balanced_by_seed": bal_map,
                   "B_balanced": B, "line": line}, fh,
                  ensure_ascii=False, indent=1)
    print("\n已存 -> %sreadings.json" % ROOT)
    return 0 if (gate_repro and not prec_problems and not problems) else 1


sys.exit(main())
