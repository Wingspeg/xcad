#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_xcad_head_prior.py -- (D) 在 xcad_v2 上的 0-GPU 读数与 §29.2 判线核对。

同一批 11,836 条 rel=1 test 边、同一把 filtered 尺子（`filt_rank` = gold 在 GPU 段内的
1-based 排名）把已发表模型(`joint_recheck`)、时序口径先验臂(`headprior_ts` 及 seed43)、
静态口径对照臂(`headprior_static`)并排，再叠上 §28 已量的闭式通道（head 历史查表 / 属性 RF）。

判线阈值全部取 §29.2 的预注册值（写在跑之前），本脚本只核对、不修改：
    ① 修复线  复发层 H@1 >= 0.90          ② 追平线  复发层 H@1 >= 0.997
    ③ 属性线  全体   H@1 >= 0.889659      ④ 不回退  全体 >= 0.874843 且 全新层 >= 0.892525
    ⑤ 天花板  全体 >= 0.901（拼装靶子按同一批边现场重算）
    ⑥ 成色    复发层与闭式查表 top-1 同率 >= 0.95，"表答错/表为 0"两档分别报数
    ⑦ 噪声    seed42 vs 43：复发 >= 0.03、全体/全新 >= 0.01 才允许写强弱
    ⑧ 停止线  复发层 < 0.804496 ⇒ (D) 判负

用法(server): ~/xcad/xcad_code/.venv/bin/python xcad_code/scripts/analyze_xcad_head_prior.py
产物: outputs/reports/analyze_xcad_head_prior.{log,json}
"""
import json
import os
import re

import numpy as np
import torch

D = os.path.expanduser("~/xcad/xcad_code/xcad_model")
DATA = os.path.join(D, "data/xcad_v2")
ARMS = os.path.join(D, "src/xcad_v2_arms")
MODELS = os.path.join(D, "models")
REP = os.path.expanduser("~/xcad/xcad_code/outputs/reports")
REL = 1

REF = {                                     # §29.1 的参照数（跑之前就有）
    "published": {"all": 0.884843, "rec": 0.697947, "novel": 0.902525},
    "lookup": {"all": 0.478033, "rec": 0.999022, "novel": 0.428743},
    "rf_attr": {"all": 0.889659, "rec": 0.804496, "novel": 0.897716},
    "oracle": {"all": 0.918214, "rec": 1.000000, "novel": 0.910570},
}
LINE = {"fix": 0.90, "tie": 0.997, "attr": 0.889659, "hold_all": 0.884843 - 0.01,
        "hold_novel": 0.902525 - 0.01, "ceil": 0.910865 - 0.01, "agree": 0.95,
        "stop": 0.804496, "noise_rec": 0.03, "noise_big": 0.01}

n_ents = int(open(os.path.join(DATA, "stat.txt")).read().split()[0])

# ── rel=1 边（head, 卡型列, τ）；列序 = tail 实体 id 升序，与 exporter 同一约定 ──
E = {}
for s in ("train", "valid", "test"):
    a = np.loadtxt(os.path.join(DATA, f"{s}.txt"), dtype=np.int64)
    r = a[a[:, 1] == REL]
    E[s] = np.stack([r[:, 0], r[:, 2], r[:, 3]], axis=1)
tails = sorted(set(np.concatenate([E[s][:, 1] for s in E]).tolist()))
n_types = len(tails)
gpu_base = tails[0]
assert gpu_base == n_ents - n_types, "rel1 的 tail 段不是末尾连续段"
col = np.zeros(n_ents, dtype=np.int64)
col[tails] = np.arange(n_types)
H = np.concatenate([E["train"], E["valid"]])
TE = E["test"]
hd, tc = TE[:, 0], col[TE[:, 1]]

# ── 闭式 head 记忆通道：每 head 历史众数卡型；并列按全局频次、再按类号（= §28 mode_of）──
cnt = np.zeros((n_ents, n_types), dtype=np.int64)
np.add.at(cnt, (H[:, 0], col[H[:, 1]]), 1)
glob_cnt = cnt.sum(axis=0)
rank_of_cls = np.empty(n_types, dtype=np.int64)
rank_of_cls[np.argsort(-glob_cnt, kind="stable")] = np.arange(n_types)
sc = (cnt.astype(np.float64) * 1e8 + (n_types - rank_of_cls)[None, :] * 1e4
      + (n_types - np.arange(n_types))[None, :])
mode = sc.argmax(axis=1)
is_rec = cnt[hd].sum(axis=1) > 0
tbl_ok = (mode[hd] == tc)
ONE = np.ones(len(hd), dtype=bool)
print(f"[{os.path.basename(__file__)}] xcad_v2 rel=1: test 边={len(TE)} "
      f"复发={int(is_rec.sum())} ({is_rec.mean()*100:.2f}%) 全新={int((~is_rec).sum())} "
      f"卡型段={tails}")
for nm, m in (("all", ONE), ("rec", is_rec), ("novel", ~is_rec)):
    got = float(tbl_ok[m].mean())
    assert abs(got - REF["lookup"][nm]) <= 1.5e-4, f"闭式查表 {nm} 读数 {got:.6f} 与 §28 不符"
print(f"[自检] 闭式查表复现 §28.1 = {tbl_ok[ONE].mean():.6f}/{tbl_ok[is_rec].mean():.6f}/"
      f"{tbl_ok[~is_rec].mean():.6f}  OK")


def metrics(rank, mask):
    r = rank[mask]
    return {"n": int(mask.sum()), "mrr": float((1.0 / r).mean()),
            "h1": float((r <= 1).mean()), "h3": float((r <= 3).mean()),
            "h10": float((r <= 10).mean())}


STRATA = (("all", ONE), ("rec", is_rec), ("novel", ~is_rec))
TAGS = [t for t in ("joint_recheck", "headprior_ts", "headprior_ts_s43", "headprior_static",
                    "headprior_ts_smoke") if os.path.exists(os.path.join(ARMS, f"{t}.pt"))]
if not TAGS:
    raise SystemExit("没有任何 dump，先跑 scripts/run_xcad_head_prior.sh")
arms = {}
for tag in TAGS:
    d = torch.load(os.path.join(ARMS, f"{tag}.pt"), map_location="cpu")
    assert np.array_equal(d["heads"].numpy(), hd), f"{tag}: 行序/head 不一致"
    assert np.array_equal(col[d["tails"].numpy()], tc), f"{tag}: gold 卡型不一致"
    arms[tag] = {"rank": d["filt_rank"].numpy().astype(np.float64),
                 "top1": (d["model_top1"].numpy() - gpu_base) if "model_top1" in d else None}
    print(f"[dump] {tag:20s} n={len(arms[tag]['rank'])}  keys={sorted(d.keys())}")

out = {"dataset": "xcad_v2", "rel": REL, "gpu_base": int(gpu_base), "n_test": int(len(hd)),
       "reference": REF, "lines_thresholds": LINE}

print("\n=== Q0 各臂三档 H@1/H@3/H@10/MRR（同一批边、同一把尺子）===")
print(f"  {'arm':20s} {'档':6s} {'n':>6s}  {'H@1':>9s} {'H@3':>9s} {'H@10':>9s} {'MRR':>9s}")
res = {}
for tag in TAGS:
    res[tag] = {nm: metrics(arms[tag]["rank"], m) for nm, m in STRATA}
    for nm, mm in res[tag].items():
        print(f"  {tag:20s} {nm:6s} {mm['n']:6d}  {mm['h1']:.6f} {mm['h3']:.6f} "
              f"{mm['h10']:.6f} {mm['mrr']:.6f}")
out["arms"] = res
if "joint_recheck" in res:
    for nm in ("all", "rec", "novel"):
        got = res["joint_recheck"][nm]["h1"]
        assert abs(got - REF["published"][nm]) <= 1.5e-6, \
            f"回归失败：joint_recheck {nm} H@1={got:.6f} != 已发表 {REF['published'][nm]:.6f}"
    print("[回归自检] joint_recheck 三档与已发表逐位相同 0.884843/0.697947/0.902525  OK")

print("\n=== Q1 §29.2 判线核对（正式臂 = headprior_ts；缺它就用第一个可用臂并标明）===")
lines = {}


def verdict(name, ok, detail):
    print(f"  [{name}] {'PASS' if ok else 'FAIL'}  {detail}")
    lines[name] = {"pass": bool(ok), "detail": detail}
    return ok


main_tag = "headprior_ts" if "headprior_ts" in res else TAGS[0]
if main_tag != "headprior_ts":
    print(f"  !! 正式臂 headprior_ts 还没跑，本轮判线记在 {main_tag} 名下（冒烟读数值不判线）")
a = res[main_tag]
verdict("1 修复线 复发>=0.90", a["rec"]["h1"] >= LINE["fix"],
        f"{main_tag} 复发层 H@1={a['rec']['h1']:.6f}（模型 0.697947 / 查表 0.999022 / RF 0.804496）")
verdict("2 追平线 复发>=0.997", a["rec"]["h1"] >= LINE["tie"],
        f"{main_tag} 复发层 H@1={a['rec']['h1']:.6f} —— 过了就等于『基本是那张表』，必须与⑥同报")
verdict("3 属性线 全体>=0.889659", a["all"]["h1"] >= LINE["attr"],
        f"{main_tag} 全体 H@1={a['all']['h1']:.6f} vs 离线属性 RF 0.889659")
verdict("4a 不回退(全体)", a["all"]["h1"] >= LINE["hold_all"],
        f"{a['all']['h1']:.6f} vs 0.884843-0.01={LINE['hold_all']:.6f}")
verdict("4b 不回退(全新层)", a["novel"]["h1"] >= LINE["hold_novel"],
        f"{a['novel']['h1']:.6f} vs 0.902525-0.01={LINE['hold_novel']:.6f}（该层表恒 0，是接线自检）")

# ⑤ 的手工拼装靶子：复发层取闭式查表、全新层取本臂模型 —— 只看 H@1（其余档对拼装无定义）
mdl_ok = arms[main_tag]["rank"] <= 1
sp_h1 = float((tbl_ok[is_rec].sum() + mdl_ok[~is_rec].sum()) / len(hd))
out["splice_manual_h1"] = sp_h1
verdict("5 天花板 全体>=0.901", a["all"]["h1"] >= LINE["ceil"],
        f"本臂全体 {a['all']['h1']:.6f} vs 现场拼装(复发查表+全新模型) {sp_h1:.6f}"
        f"（判线 = 拼装-0.01 = {LINE['ceil']:.6f}；ORACLE 查表∪RF = {REF['oracle']['all']:.6f}，"
        f"单通道注入到不了，不拿它当线）")

if "headprior_ts_s43" in res:
    b = res["headprior_ts_s43"]
    d_ = {nm: float(abs(a[nm]["h1"] - b[nm]["h1"])) for nm in ("all", "rec", "novel")}
    out["seed_diff"] = d_
    verdict("7 噪声带(只报数)", True,
            f"|42-43| 全体={d_['all']:.6f}(线 0.01) 复发={d_['rec']:.6f}(线 0.03) "
            f"全新={d_['novel']:.6f}(线 0.01) —— 差小于线时任何强弱陈述都不许写")
verdict("8 停止线未触发(复发>=0.804496)", a["rec"]["h1"] >= LINE["stop"],
        f"{main_tag} 复发层 H@1={a['rec']['h1']:.6f} vs 停止线 {LINE['stop']}")
out["lines"] = lines

print("\n=== Q2 成色：这一臂是不是『只是把那张表搬过来』（判线⑥）===")
mech = {}
for tag in TAGS:
    t1 = arms[tag]["top1"]
    if t1 is None:
        print(f"  {tag}: dump 无 model_top1，跳过同率")
        continue
    in_seg = (t1 >= 0) & (t1 < n_types)
    same = np.where(in_seg, t1 == mode[hd], False)
    m_ok = arms[tag]["rank"] <= 1
    wrong_rec = is_rec & ~tbl_ok
    mech[tag] = {
        "top1_agree_all": float(same.mean()),
        "top1_agree_rec": float(same[is_rec].mean()),
        "top1_agree_novel": float(same[~is_rec].mean()),
        "top1_outside_segment_rows": int((~in_seg).sum()),
        "only_model_right_rec": int((m_ok & ~tbl_ok & is_rec).sum()),
        "only_table_right_rec": int((tbl_ok & ~m_ok & is_rec).sum()),
        "n_table_wrong_rec": int(wrong_rec.sum()),
        "model_h1_where_table_wrong_rec": (float(m_ok[wrong_rec].mean())
                                           if wrong_rec.any() else float("nan")),
        "model_h1_where_table_zero_novel": float(m_ok[~is_rec].mean()),
        "oracle_model_or_table_all": float((m_ok | tbl_ok).mean()),
        "oracle_model_or_table_rec": float((m_ok | tbl_ok)[is_rec].mean()),
    }
    g = mech[tag]
    print(f"  {tag:20s} top-1 同率 全体={g['top1_agree_all']:.6f} 复发={g['top1_agree_rec']:.6f} "
          f"全新={g['top1_agree_novel']:.6f}  段外 top-1={g['top1_outside_segment_rows']} 行")
    print(f"  {'':20s} 复发层 只模型对={g['only_model_right_rec']} 只查表对={g['only_table_right_rec']}"
          f"  表答错的复发边 n={g['n_table_wrong_rec']} 上模型 H@1="
          f"{g['model_h1_where_table_wrong_rec']:.6f}  表为 0 的全新边 H@1="
          f"{g['model_h1_where_table_zero_novel']:.6f}")
    print(f"  {'':20}  (模型 ∪ 查表) 命中 = 全体 {g['oracle_model_or_table_all']:.6f} / "
          f"复发 {g['oracle_model_or_table_rec']:.6f}")
out["mechanism"] = mech
if main_tag in mech:
    verdict("6 成色 同率>=0.95", mech[main_tag]["top1_agree_rec"] >= LINE["agree"],
            f"{main_tag} 复发层与闭式查表 top-1 同率={mech[main_tag]['top1_agree_rec']:.6f}；"
            "同率高=分数由先验主导（要写进正文），同率低但 H@1 高=encoder 有独立贡献")

print("\n=== Q3 学到的 head_alpha（各臂 checkpoint 里的标量）===")
print("  注：stat 名里 flag 后缀在 seed 之前（...-compat-headprior-priorts-seed42），"
      "按文件名 glob 撞过 3 个同名臂 ⇒ 改成从各臂日志的 'Load Model name' 拿确切路径。")
alpha = {}
seen_ck = {}
RE_LOAD = re.compile(r"Load Model name: (\S+?)/?\.? Using best epoch : (\d+)")
for tag in TAGS:
    log = os.path.join(ARMS, f"{tag}.log")
    hits = []
    if os.path.exists(log):
        txt = open(log, errors="replace").read().replace("\r", "\n")
        hits = [(m.group(1), int(m.group(2))) for m in RE_LOAD.finditer(txt)]
    names = {os.path.basename(h[0]) for h in hits}
    if len(names) != 1:
        print(f"  {tag:20s} 日志里的 checkpoint 名不唯一 {sorted(names)}，跳过")
        continue
    name, best_epoch = hits[-1]
    path = os.path.join(MODELS, os.path.basename(name))
    if not os.path.exists(path):
        print(f"  {tag:20s} checkpoint 不存在：{os.path.basename(path)}")
        continue
    ck_tag = seen_ck.get(os.path.basename(path))
    if ck_tag is not None:
        print(f"  {tag:20s} !! 与 {ck_tag} 共用同一个 checkpoint（stat 名只含 flag+seed，"
              "不含 n-epochs）⇒ 这一行的 alpha 是后跑那臂的，不作数")
        continue
    seen_ck[os.path.basename(path)] = tag
    ck = torch.load(path, map_location="cpu")
    st = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    ks = [k for k in st if "alpha" in k]
    alpha[tag] = {k: float(st[k]) for k in ks}
    alpha[tag]["_best_epoch"] = best_epoch
    alpha[tag]["_checkpoint"] = os.path.basename(path)
    print(f"  {tag:20s} best_epoch={best_epoch}  "
          + ("  ".join(f"{k}={float(st[k]):.4f}" for k in ks) or "（没有先验标量）")
          + f"   <- {os.path.basename(path)[-46:]}")
out["alpha"] = alpha

os.makedirs(REP, exist_ok=True)
p = os.path.join(REP, "analyze_xcad_head_prior.json")
with open(p, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n[written] {p}")
print("[done] 判线阈值全部取 §29.2 预注册值，本脚本不修改任何阈值。")
