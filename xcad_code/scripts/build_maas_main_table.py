#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_maas_main_table.py -- (G) 收尾：把 maas 上的全部臂与全部闭式通道钉成一张主表。

0 GPU：只读 `src/maas_arms/*.pt`（测试阶段 dump 下来的逐边过滤秩）+ `data/maas/*` +
各臂训练日志里的 val MRR / best epoch。不训练、不改任何数据。

口径与 §18–§26 完全一致：同一批 121,371 行 rel-0 测试边、同一个过滤式
（同一 (head, τ) 的其它已知真值先滤掉再排名）、同一三个分组（全体 / 全新 head /
复发边=硬线层）、同一个闭式排名实现（`generic_strata._ranks_from_cols`，即生产实现）。

产出三件东西（都写进 `outputs/reports/`）：
  maas_main_table.md    —— 论文可用的主表 + 成色表 + 判线表（Markdown）
  maas_main_table.csv   —— 逐方法×逐分组的长表（数字自己再拼也行）
  maas_main_table.json  —— 同上 + 元信息（日志/行数/断言结果）
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

D = os.path.expanduser("~/xcad/xcad_code/xcad_model")
ARMS = os.path.join(D, "src/maas_arms")
DATA = os.path.join(D, "data/maas")
OUT = os.path.expanduser("~/xcad/xcad_code/outputs/reports")
sys.path.insert(0, os.path.join(D, "src"))
import generic_strata as prod                                        # noqa: E402

# ── 方法清单：tag -> (块, 一句话说明) ────────────────────────────────────
MODEL_ARMS = [
    ("base",         "论文配置直接搬到 maas（无任何先验）"),
    ("hetero",       "论文配置 + 关系异构消息基（§18 消融）"),
    ("static",       "论文配置去掉 GRU 时间演化（§18 消融）"),
    ("baseemb",      "head 表征 += 可学习的 base 共享嵌入（§21）"),
    ("basefeat",     "head 表征 += 注入的 base 历史卡型直方图（§22）"),
    ("baseprior",    "输出层 += α·log1p(base 历史卡型计数)（§23）"),
    ("headprior",    "输出层 += base 与 head 两条先验通道（静态表，§24）"),
    ("priorgate",    "双通道 + 按行 softplus 信任门控（静态表，§25）"),
    ("priorts",      "双通道 + 时序口径先验表（每快照只用严格更早的边，§26）"),
    ("priortsgate",  "时序口径 + 按行信任门控（§26）"),
    ("priorts_s43",  "priorts 的第二个 seed（只为量 run-to-run 噪声）"),
]
CHAN = [
    ("tail_prior",   "恒定答历史最热卡型（常数）"),
    ("entity_mem",   "背历史：该 head 自己的历史众数卡型（复发通道上界）"),
    ("base_hist",    "父节点历史：base_id 的历史卡型直方图（属性通道）"),
    ("add_hist",     "两表 log1p 直接相加（不学习的合并）"),
]
SEED_PAIR = ("priorts", "priorts_s43")
HEADLINE = ("priorts", "priortsgate")
PRIOR_ARMS = ("baseprior", "headprior", "priorgate", "priorts", "priortsgate",
              "priorts_s43")
# checkpoint 名里的旗标串是固定顺序拼出来的，与 dump tag 不同名
TAG2CK = {"baseprior": "-baseprior-seed42",
          "baseemb": "-baseemb-seed42", "basefeat": "-basefeat-seed42",
          "hetero": "-het2-", "static": "-staticrel-seed42",
          "headprior": "-headprior-seed42",
          "priorgate": "-priorgate-seed42",
          "priorts": "-headprior-priorts-seed42",
          "priortsgate": "-priorgate-priorts-seed42",
          "priorts_s43": "-headprior-priorts-seed43"}

# ── 数据侧：行序、分组、闭式通道（与 analyze_maas_base_prior.py 同一套算式）──
n_ents = int(open(os.path.join(DATA, "stat.txt")).read().split()[0])
base_of = np.load(os.path.join(DATA, "base_of.npy"))
pad = int(base_of.max())

tails = set()
for fn in ("train.txt", "valid.txt", "test.txt"):
    for ln in open(os.path.join(DATA, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            tails.add(int(p[2]))
gpu_base = min(tails)
n_types = len(tails)
assert max(tails) == n_ents - 1 and n_types == n_ents - gpu_base, "卡型段不是编号末尾连续段"

hrows, rows = [], []
for fn in ("train.txt", "valid.txt"):
    for ln in open(os.path.join(DATA, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            hrows.append((int(p[0]), int(p[2]) - gpu_base))
for ln in open(os.path.join(DATA, "test.txt")):
    p = ln.split()
    if len(p) >= 4 and int(p[1]) == 0:
        rows.append((int(p[0]), int(p[2]) - gpu_base, int(p[3])))
T = pd.DataFrame(rows, columns=["hd", "tc", "tau"])
H = pd.DataFrame(hrows, columns=["hd", "tc"])
hist_pairs = set(map(tuple, H.values))
hist_heads = set(H.hd)
gold = T.tc.to_numpy()
n = len(T)

per = {}
for h, tau, t in zip(T.hd, T.tau, T.tc):
    per.setdefault((int(h), int(tau)), set()).add(int(t))
filt = [{x for x in per[(int(h), int(tau))] if x != int(g)}
        for h, tau, g in zip(T.hd, T.tau, gold)]


def chan_ranks(row_tables):
    """任意"每行一张 {卡型: 分}"表 -> 同尺过滤排名（生产实现同一个函数）。"""
    uniq, index_of, rk_id = [], {}, np.empty(n, dtype=np.int64)
    for i, t in enumerate(row_tables):
        k = frozenset(t.items())
        j = index_of.get(k)
        if j is None:
            j = index_of[k] = len(uniq)
            uniq.append(t)
        rk_id[i] = j
    return prod._ranks_from_cols(rk_id, uniq, np.asarray(gold), filt, n_types,
                                 np.random.default_rng(7),
                                 (0, n_types - 1)).astype(np.float64)


Hb = H.assign(b=base_of[H.hd.values])
tab_b = {int(b): {int(c): int(v) for c, v in sub.groupby("tc")["n"].sum().items()}
         for b, sub in Hb.groupby(["b", "tc"]).size().rename("n").reset_index().groupby("b")}
tab_h = {int(h): {int(c): int(v) for c, v in sub.groupby("tc")["n"].sum().items()}
         for h, sub in H.groupby(["hd", "tc"]).size().rename("n").reset_index().groupby("hd")}
tab_t = {int(c): int(v) for c, v in H.groupby("tc").size().items()}
bh_rank = chan_ranks([tab_b.get(int(b), {}) for b in base_of[T.hd.values]])
em_rank = chan_ranks([tab_h.get(int(h), {}) for h in T.hd.values])
tp_rank = chan_ranks([tab_t] * n)
_bl = {b: {c: float(np.log1p(v)) for c, v in d.items()} for b, d in tab_b.items()}
_hl = {h: {c: float(np.log1p(v)) for c, v in d.items()} for h, d in tab_h.items()}


def _add_tbl(h, b):
    d = dict(_bl.get(b, {}))
    for c, v in _hl.get(h, {}).items():
        d[c] = d.get(c, 0.0) + v
    return d


add_rank = chan_ranks([_add_tbl(int(h), int(b))
                       for h, b in zip(T.hd.values, base_of[T.hd.values])])

new_head = ~T.hd.isin(hist_heads).values
recur = np.array([(int(h), int(t)) in hist_pairs
                  for h, t in zip(T.hd.values, T.tc.values)])
COH = [("all", "全部 test 边", np.ones(n, bool)),
       ("novel", "全新 head", new_head),
       ("recurrent", "复发边（硬线层）", recur)]

# ── 学习臂：读 dump 的逐边过滤秩，并校验行序 ──
ranks, present, absent = {}, [], []
for tag, desc in MODEL_ARMS:
    path = os.path.join(ARMS, f"{tag}.pt")
    if not os.path.exists(path):
        absent.append(tag)
        continue
    d = torch.load(path, map_location="cpu")
    assert np.array_equal(d["heads"].numpy(), T.hd.values), f"{tag} 行序不一致"
    assert np.array_equal(d["tails"].numpy() - gpu_base, gold), f"{tag} gold 不一致"
    # 注意：dump 里的 `edge_in_history` 是"边在编码器最近快照窗口里"(n=17,378)，与本表的
    # 复发定义 (head, 卡型) ∈ train∪valid (n=17,525) **不是同一把尺子**，故不做相等断言。
    ranks[tag] = d["filt_rank"].numpy().astype(np.float64)
    present.append(tag)
M = dict(ranks)
M.update({"tail_prior": tp_rank, "entity_mem": em_rank,
          "base_hist": bh_rank, "add_hist": add_rank})

# ── 自检：与生产 dump 自带的闭式秩对一遍（复发层应逐位接近）──
_ck = torch.load(os.path.join(ARMS, "baseprior.pt"), map_location="cpu")
selfcheck = []
for nm, mine, theirs in (
        ("entity_mem", em_rank, np.asarray(_ck["baseline"]["entity_memory"], dtype=np.float64)),
        ("tail_prior", tp_rank, np.asarray(_ck["baseline"]["tail_prior"], dtype=np.float64))):
    selfcheck.append({"channel": nm, "cohort": "recurrent",
                      "mine_H@1": round(float(np.mean(mine[recur] <= 1)), 4),
                      "prod_H@1": round(float(np.mean(theirs[recur] <= 1)), 4),
                      "rank_diff_rows": int((mine[recur] != theirs[recur]).sum())})

# ── 训练日志里的 val MRR / best epoch + checkpoint 里的学到的信任度 ──
_re_mrr = re.compile(r"Best MRR\s+([0-9.]+)")
_re_ep = re.compile(r"Using best epoch\s*:\s*(\d+)")
sel = {}
for tag in present:
    log = os.path.join(ARMS, f"{tag}.log")
    if not os.path.exists(log):
        continue
    txt = open(log, errors="ignore").read().replace("\r", "\n")
    mm, ee = _re_mrr.findall(txt), _re_ep.findall(txt)
    if mm:
        sel[tag] = {"val_mrr": float(mm[-1]),
                    "best_epoch": int(ee[-1]) if ee else None}

# ── 学到的信任度（读 checkpoint 里的标量/门控参数，0 GPU）──
alphas = {}
mdir = os.path.join(D, "models")
for tag in present:
    if tag not in PRIOR_ARMS:
        continue
    hits = [f for f in sorted(os.listdir(mdir)) if TAG2CK[tag] in f]
    if not hits:
        continue
    sd = torch.load(os.path.join(mdir, hits[0]), map_location="cpu")["state_dict"]
    e = {"ckpt": hits[0]}
    for k in ("prior_alpha", "head_alpha"):
        if k in sd:
            e[k] = round(float(sd[k]), 4)
    if "gate_w" in sd:
        e["gate_w"] = [[round(x, 4) for x in row] for row in sd["gate_w"].float().tolist()]
        e["gate_c"] = [round(x, 4) for x in sd["gate_c"].float().tolist()]
    alphas[tag] = e

# ── 主表：方法 × 分组 -> H@1 / H@3 / H@10 / MRR ──
long_rows = []


def stats(r, mask):
    x = r[mask]
    return {"n": int(mask.sum()), "H@1": float(np.mean(x <= 1)),
            "H@3": float(np.mean(x <= 3)), "H@10": float(np.mean(x <= 10)),
            "MRR": float(np.mean(1.0 / x)), "med": float(np.median(x))}


ORDER = [("model", t, d) for t, d in MODEL_ARMS if t in M] + \
        [("chan", t, d) for t, d in CHAN]
for kind, tag, desc in ORDER:
    for cid, cname, mask in COH:
        s = stats(M[tag], mask)
        long_rows.append({"block": "learned" if kind == "model" else "closed_form",
                          "method": tag, "desc": desc, "cohort": cid,
                          "cohort_cn": cname, **{k: round(v, 6) for k, v in s.items()},
                          "val_mrr": sel.get(tag, {}).get("val_mrr"),
                          "best_epoch": sel.get(tag, {}).get("best_epoch")})
# oracle 两行（任一通道 top-1 命中即算命中）——非学习合并的天花板
for oid, oname, orm in (("oracle_base_mem", "oracle(base∪mem)：两通道任一命中",
                         (bh_rank <= 1) | (em_rank <= 1)),
                        ("oracle_all", "oracle(+tail)：三通道任一命中",
                         (bh_rank <= 1) | (em_rank <= 1) | (tp_rank <= 1))):
    for cid, cname, mask in COH:
        x = (mask & orm).sum()
        long_rows.append({"block": "ceiling", "method": oid, "desc": oname,
                          "cohort": cid, "cohort_cn": cname, "n": int(mask.sum()),
                          "H@1": float(x / mask.sum()), "H@3": None, "H@10": None,
                          "MRR": None, "med": None,
                          "val_mrr": None, "best_epoch": None})

# ── 成色表：先验臂与两条闭式通道的关系 ──
mech = []
for arm in [a for a in PRIOR_ARMS if a in M]:
    row = {"arm": arm}
    for cid, cname, mask in COH:
        mh, ph, eh = M[arm][mask] <= 1, bh_rank[mask] <= 1, em_rank[mask] <= 1
        both_wrong = ~ph & ~eh
        row[f"agree_base_{cid}"] = float(np.mean((mh & ph) | (~mh & ~ph)))
        row[f"agree_mem_{cid}"] = float(np.mean((mh & eh) | (~mh & ~eh)))
        row[f"gain_over_base_{cid}"] = int((mh & ~ph).sum())
        row[f"loss_vs_base_{cid}"] = int((ph & ~mh).sum())
        row[f"bothwrong_n_{cid}"] = int(both_wrong.sum())
        row[f"bothwrong_H@1_{cid}"] = float(np.mean(mh[both_wrong])) if both_wrong.any() else float("nan")
    mech.append(row)

# ── seed 噪声 ──
noise = []
if all(t in M for t in SEED_PAIR):
    for cid, cname, mask in COH:
        a = float(np.mean(M[SEED_PAIR[0]][mask] <= 1))
        b = float(np.mean(M[SEED_PAIR[1]][mask] <= 1))
        noise.append({"cohort": cid, "cohort_cn": cname, "seed42_H@1": round(a, 4),
                      "seed43_H@1": round(b, 4), "abs_delta": round(abs(a - b), 4)})

# ── 判线表（§26.3 预注册，阈值写死在前，这里只填读数）──
def h1(tag, cid):
    mask = dict((c, msk) for c, _, msk in COH)[cid]
    return float(np.mean(M[tag][mask] <= 1))


LINES = [
    ("不回退线", "复发层 H@1 ≥ 0.7337（§24 的 0.7357 − 0.002）", 0.7337,
     {t: h1(t, "recurrent") for t in HEADLINE + ("priorts_s43",)}, "ge"),
    ("硬线（用户裁定）", "复发层 H@1 ≥ 0.3473（编队构成基线）", 0.3473,
     {t: h1(t, "recurrent") for t in HEADLINE}, "ge"),
    ("修复线", "全新 head H@1 ≥ 0.2877（§23 单通道读数）", 0.2877,
     {t: h1(t, "novel") for t in HEADLINE + ("priorts_s43",)}, "ge"),
    ("合格线", "全体 H@1 ≥ 0.3580（不学习的 add_hist）", 0.3580,
     {t: h1(t, "all") for t in HEADLINE + ("priorts_s43",)}, "ge"),
    ("停止线", "全体 H@1 < 0.3244（§24）即停止调参", 0.3244,
     {t: h1(t, "all") for t in HEADLINE + ("priorts_s43",)}, "lt_stop"),
    ("机制线", "两条闭式通道都答错的边上 H@1 ≥ 0.0700", 0.0700,
     {r["arm"]: r["bothwrong_H@1_all"] for r in mech if r["arm"] in HEADLINE + ("priorts_s43",)},
     "ge"),
]
verdicts = []
for name, defn, thr, readings, mode in LINES:
    for arm, v in readings.items():
        if mode == "lt_stop":
            ok = "未触发" if v >= thr else "触发"
        else:
            ok = "过" if v >= thr else "不过"
        verdicts.append({"line": name, "definition": defn, "threshold": thr,
                         "arm": arm, "readout": round(v, 4), "verdict": ok})

# ── 输出 ──
def fmt(x, nd=4):
    return "  -  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


CN = {"all": "全体 test 边", "novel": "全新 head", "recurrent": "复发边（硬线层）"}
md = []
md.append("# maas 主表（(G) 收尾产物，0 GPU）\n")
md.append(f"- 测试边：rel-0 共 {n:,} 行；分组 n = " +
          " / ".join(f"{CN[c]} {int(m.sum()):,}" for c, _, m in COH))
md.append(f"- 实体空间 {n_ents:,}（model head {gpu_base:,} ⊕ 卡型 {n_types}，"
          f"base {pad} 类 + 1 个 pad）；全新 head 占 {new_head.mean():.4f}，复发边占 {recur.mean():.4f}")
md.append("- 尺子：过滤式逐边排名（同一 (head, τ) 的其它已知真值先滤），与 §18–§26 全部读数同一实现\n")

md.append("## 表 A 学习臂（H@1 / H@3 / H@10 / MRR，分组）\n")
md.append("| 臂 | 全体 H@1 | 全体 H@3 | 全体 H@10 | 全体 MRR | 全新 H@1 | 全新 H@3 | 全新 H@10 | 全新 MRR | 复发 H@1 | 复发 H@3 | 复发 H@10 | 复发 MRR | best val MRR | best ep |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for kind, tag, desc in ORDER:
    if kind != "model":
        continue
    r = {x["cohort"]: x for x in long_rows if x["method"] == tag}
    cells = []
    for cid in ("all", "novel", "recurrent"):
        cells += [fmt(r[cid]["H@1"]), fmt(r[cid]["H@3"]), fmt(r[cid]["H@10"]), fmt(r[cid]["MRR"])]
    md.append(f"| `{tag}` | " + " | ".join(cells) +
              f" | {fmt(sel.get(tag, {}).get('val_mrr'), 4)} | "
              f"{sel.get(tag, {}).get('best_epoch', '—')} |")

md.append("\n## 表 B 非学习通道与天花板（同一把尺子）\n")
md.append("| 通道 | 全体 H@1 | 全体 H@3 | 全体 H@10 | 全体 MRR | 全新 H@1 | 全新 H@3 | 全新 H@10 | 全新 MRR | 复发 H@1 | 复发 H@3 | 复发 H@10 | 复发 MRR |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for kind, tag, desc in ORDER:
    if kind != "chan":
        continue
    r = {x["cohort"]: x for x in long_rows if x["method"] == tag}
    cells = []
    for cid in ("all", "novel", "recurrent"):
        cells += [fmt(r[cid]["H@1"]), fmt(r[cid]["H@3"]), fmt(r[cid]["H@10"]), fmt(r[cid]["MRR"])]
    md.append(f"| `{tag}` | " + " | ".join(cells) + " |")
for oid, oname in (("oracle_base_mem", "oracle(base∪mem)"), ("oracle_all", "oracle(+tail)")):
    r = {x["cohort"]: x for x in long_rows if x["method"] == oid}
    md.append(f"| **{oname}** | {fmt(r['all']['H@1'])} | — | — | — | "
              f"{fmt(r['novel']['H@1'])} | — | — | — | {fmt(r['recurrent']['H@1'])} | — | — | — |")
md.append("\n> oracle 两行是 hit-level 天花板（任一通道 top-1 命中即记命中），只出 H@1；"
          "它不是模型，任何只在通道之间做选择的方法都越不过它。")

md.append("\n## 表 C 成色：先验臂与两条闭式通道的逐边关系\n")
md.append("| 臂 | 分组 | 与 base_hist top-1 一致率 | 与 entity_mem top-1 一致率 | 臂对&表错 | 表对&臂错 | 两通道都错 n | 该子集上臂 H@1 |")
md.append("|---|---|---|---|---|---|---|---|")
for row in mech:
    for cid in ("all", "novel", "recurrent"):
        md.append(f"| `{row['arm']}` | {CN[cid]} | {row[f'agree_base_{cid}']:.4f} | "
                  f"{row[f'agree_mem_{cid}']:.4f} | {row[f'gain_over_base_{cid}']:,} | "
                  f"{row[f'loss_vs_base_{cid}']:,} | {row[f'bothwrong_n_{cid}']:,} | "
                  f"{row[f'bothwrong_H@1_{cid}']:.4f} |")

md.append("\n## 表 D run-to-run 噪声（唯一一对同配置不同 seed）\n")
md.append("| 分组 | seed 42 H@1 | seed 43 H@1 | 绝对差 |")
md.append("|---|---|---|---|")
for r in noise:
    md.append(f"| {r['cohort_cn']} | {r['seed42_H@1']:.4f} | {r['seed43_H@1']:.4f} | {r['abs_delta']:.4f} |")

md.append("\n## 表 E 预注册判线（阈值在 §26.3 写死，这里只填读数）\n")
md.append("| 判线 | 定义 | 阈值 | 臂 | 读数 | 判定 |")
md.append("|---|---|---|---|---|---|")
for v in verdicts:
    md.append(f"| {v['line']} | {v['definition']} | {v['threshold']:.4f} | `{v['arm']}` | "
              f"{v['readout']:.4f} | **{v['verdict']}** |")

md.append("\n## 表 F 学到的信任度（checkpoint 标量；门控臂的逐行 α 分离见 Q6）\n")
md.append("| 臂 | prior_alpha (base 通道) | head_alpha (head 通道) | gate_w / gate_c |")
md.append("|---|---|---|---|")
for tag in [t for t, _ in MODEL_ARMS if t in alphas]:
    e = alphas[tag]
    g = ""
    if "gate_w" in e:
        g = f"w={e['gate_w']} c={e['gate_c']}"
    md.append(f"| `{tag}` | {e.get('prior_alpha', '—')} | {e.get('head_alpha', '—')} | {g or '—'} |")

md.append("\n## 表 0 方法清单（臂=同一超参/同一 seed/按 val MRR 选 checkpoint）\n")
md.append("| 名称 | 类型 | 一句话 |")
md.append("|---|---|---|")
for tag, desc in MODEL_ARMS:
    md.append(f"| `{tag}` | 学习臂{'（缺失）' if tag not in M else ''} | {desc} |")
for tag, desc in CHAN:
    md.append(f"| `{tag}` | 非学习通道 | {desc} |")

os.makedirs(OUT, exist_ok=True)
pd.DataFrame(long_rows).to_csv(os.path.join(OUT, "maas_main_table.csv"), index=False)
with open(os.path.join(OUT, "maas_main_table.md"), "w") as f:
    f.write("\n".join(md) + "\n")
meta = {"n_test_edges": n, "n_ents": n_ents, "gpu_base": gpu_base, "n_types": n_types,
        "n_bases_excl_pad": pad, "novel_share": round(float(new_head.mean()), 6),
        "recurrent_share": round(float(recur.mean()), 6),
        "arms_present": present, "arms_missing": absent,
        "cohort_n": {c: int(m.sum()) for c, _, m in COH},
        "selfcheck_closed_form_vs_prod": selfcheck,
        "seed_noise": noise, "verdicts": verdicts, "mechanism": mech,
        "learned_alphas": alphas, "val_mrr_and_epoch": sel}
with open(os.path.join(OUT, "maas_main_table.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)

print("\n".join(md))
print("\n[files] " + "  ".join(os.path.join(OUT, x) for x in
                               ("maas_main_table.md", "maas_main_table.csv",
                                "maas_main_table.json")))
print("[selfcheck] " + json.dumps(selfcheck, ensure_ascii=False))
print("[missing arms] " + (", ".join(absent) if absent else "无"))
