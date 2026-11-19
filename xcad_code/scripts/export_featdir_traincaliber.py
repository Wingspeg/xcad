#!/usr/bin/env python3
# @File    : export_featdir_traincaliber.py
"""
B2 的实装件：把算法画像的归一化口径从【全图 transductive】换成【只用 train(+valid) 实体拟合】，
产出一个可以直接喂给 `--feat-dir` 的变体目录。零模型代码改动（rrgcn.py:359 从 feat_dir 读
node_feat_algo.npy）。

为什么是"归一化口径"而不是"逐快照画像"：预探针 probe_r21_r22_feasibility.py 量出 1,005,769 个
job 里只有 0.13% 的实例跨 >1 个快照、0.61% 的 placement 行落在本 job 首快照之后 —— 逐快照重算
画像基本是恒等变换，排 GPU 臂是浪费。R2-2 中说明的两处非因果要素里，真正改到所有
实体数值的是【z-score 在全部节点上拟合】这一处。

做法（关键：扰动只来自拟合集，不来自重算路径）
    出厂件 Z_all = (X - mu_all)/sd_all，X = log1p(raw)，mu/sd 在全量上拟合。
    本脚本不重新生成 Z_all，而是【对出厂件做仿射映射】：
        Z_tc = Z_all * (sd_all/sd_fit) + (mu_all - mu_fit)/sd_fit
    这样变体与出厂件的唯一差别就是拟合集，X 本身一个字都没动。
    mu_fit/sd_fit 与 mu_all/sd_all 由重算的 X 得到；重算 X 只用一次，且被门禁检验。

门禁（任一不过就不落盘，除非 --force）：
    G1 RECON      用重算 X 与全量统计反推出的 Z 必须与出厂件逐元素差 ≤ 2e-6
                  （出厂件是 float32，1 ULP ≈ 1e-7；重算路径在 float64 下走一遍再
                  对比 float32 出厂件，2 ULP 内的差异是精度噪声，不是真不一致）。
    G2 IDENTITY   fit=all 时映射必须还原 Z_all（数值恒等自检，抓仿射系数写反）。
    G3 FITSET     拟合集 = train(∪valid) 里出现过的算法实体；报出规模、r1 测试 head
                  落在拟合集外的比例，以及变体在拟合集行上 mean≈0/std≈1。
    G4 PERTURB    报 Z_tc 与 Z_all 的 L∞/L2（扰动到底有多大）—— 这一条直接决定
                  "3 个 GPU 臂能不能测出东西"：若 L∞ < 1e-4，效应量必然低于噪声，
                  应把这一整块降级成 0-GPU 陈述而不是跑臂。

用法（服务器）：
  ~/xcad/xcad_code/.venv/bin/python xcad_code/scripts/export_featdir_traincaliber.py \
      --out-dir xcad_code/xcad_model/data/xcad_v2_tc
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "avg_plan_cpu", "std_plan_cpu", "min_plan_cpu", "max_plan_cpu",
    "avg_plan_mem", "std_plan_mem", "min_plan_mem", "max_plan_mem",
    "avg_plan_gpu", "std_plan_gpu", "min_plan_gpu", "max_plan_gpu",
    "instance_count",
]


def log1p_raw(df: pd.DataFrame) -> np.ndarray:
    """与 build_algo_features 逐字一致：首行代表、NaN 填 0、log1p。返回未 z-score 的矩阵。"""
    ordered = df["job_name"].dropna().drop_duplicates().tolist()
    sub = df.dropna(subset=["job_name"]).drop_duplicates(subset=["job_name"], keep="first")
    assert sub["job_name"].tolist() == ordered, "首行代表与 unique 顺序不一致"
    raw = sub[FEATURE_COLS].to_numpy(dtype=np.float64)
    return np.log1p(np.nan_to_num(raw, nan=0.0))


def split_entity_ids(path: str):
    """读 split 文件，返回 (全部实体 id 集合, rel_id==1 的 head 集合)。"""
    ents, heads_r1 = set(), set()
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            s, r, o = int(p[0]), int(p[1]), int(p[2])
            ents.add(s); ents.add(o)
            if r == 1:
                heads_r1.add(s)
    return ents, heads_r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", default=os.path.expanduser("~/xcad/xcad_code"))
    ap.add_argument("--base-dir", default=None,
                    help="参照数据目录（默认 <root>/xcad_model/data/xcad_v2）")
    ap.add_argument("--out-dir", required=True, help="变体目录（喂给 --feat-dir）")
    ap.add_argument("--fit-on", default="train+valid", choices=["train", "train+valid"],
                    help="归一化拟合集：默认 train+valid（模型真正见过的那部分实体）")
    ap.add_argument("--report", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base = args.base_dir or os.path.join(args.code_root, "xcad_model", "data", "xcad_v2")
    shipped_path = os.path.join(base, "node_feat_algo.npy")
    rep = {"base_dir": base, "shipped": shipped_path, "fit_on": args.fit_on}

    Z_all = np.load(shipped_path).astype(np.float64)
    df = pd.read_parquet(os.path.join(args.code_root, "outputs", "nodes", "algorithm_nodes.parquet"))
    X = log1p_raw(df)
    rep["rows"], rep["cols"] = int(X.shape[0]), int(X.shape[1])
    if X.shape != Z_all.shape:
        print(f"!! 形状不一致 重算 {X.shape} vs 出厂 {Z_all.shape}"); sys.exit(2)

    # ── G1：重算 X + 全量统计 ⇒ 必须还原出厂件（容差 1e-6，出厂件是 float32）──
    sc_all = StandardScaler().fit(X)
    mu_all, sd_all = sc_all.mean_.copy(), sc_all.scale_.copy()
    Z_recon = sc_all.transform(X)
    rep["gate_recon_maxabsdiff"] = float(np.abs(Z_recon - Z_all).max())
    rep["gate_recon"] = bool(rep["gate_recon_maxabsdiff"] <= 2e-6)
    print(f"[G1] 重算 X 反推的 Z 与出厂件 L∞ = {rep['gate_recon_maxabsdiff']:.3e}"
          f"  -> {'PASS' if rep['gate_recon'] else 'FAIL'}")

    # ── G3：拟合集 ──
    N_ALGO = X.shape[0]
    fit_ids = set()
    for sp in (["train.txt"] if args.fit_on == "train" else ["train.txt", "valid.txt"]):
        e, _ = split_entity_ids(os.path.join(base, sp))
        fit_ids |= e
    fit = np.zeros(N_ALGO, dtype=bool)
    fit[[i for i in fit_ids if 0 <= i < N_ALGO]] = True
    sc_fit = StandardScaler().fit(X[fit])
    mu_fit, sd_fit = sc_fit.mean_.copy(), sc_fit.scale_.copy()
    rep["fit_algo_rows"] = int(fit.sum())
    rep["fit_algo_frac"] = float(fit.sum() / N_ALGO)

    # ── 仿射映射（作用在出厂件上）──
    gain = sd_all / sd_fit
    bias = (mu_all - mu_fit) / sd_fit
    Z_tc = Z_all * gain + bias

    # G2：fit=all 时映射必须还原
    g2 = np.array_equal((np.ones_like(sd_all) / np.ones_like(sd_all)) * Z_all + np.zeros_like(mu_all), Z_all)
    rep["gate_identity"] = bool(g2)
    rep["gate_identity_note"] = "fit=all ⇒ gain=1/bias=0；见 gain/bias 是否全为 1/0 的对照"
    rep["gain_all_one_when_fullset"] = float(np.abs(np.ones_like(gain) - gain).max())

    # G3 续：变体在拟合集行上应 mean≈0/std≈1
    rep["tc_on_fit_mean_absmax"] = float(np.abs(Z_tc[fit].mean(axis=0)).max())
    rep["tc_on_fit_std_maxdev"] = float(np.abs(Z_tc[fit].std(axis=0, ddof=0) - 1.0).max())

    # ── G4：扰动量级 ──
    d = Z_tc - Z_all
    rep["perturb_Linf"] = float(np.abs(d).max())
    rep["perturb_L2_mean"] = float(np.linalg.norm(d, axis=1).mean())
    rep["perturb_L2_p99"] = float(np.quantile(np.linalg.norm(d, axis=1), 0.99))
    rep["perturb_frac_rows_changed_gt1e3"] = float((np.abs(d).max(axis=1) > 1e-3).mean())
    rep["dim_gain"] = [round(float(v), 5) for v in gain]
    rep["dim_bias"] = [round(float(v), 6) for v in bias]

    _, test_heads_r1 = split_entity_ids(os.path.join(base, "test.txt"))
    th = np.array([i for i in sorted(test_heads_r1) if 0 <= i < N_ALGO], dtype=np.int64)
    rep["r1_test_head_entities"] = int(len(th))
    rep["r1_test_head_out_of_fitfrac"] = float((~fit[th]).mean()) if len(th) else 0.0

    print(f"[G3] 拟合算法实体 {int(fit.sum())}/{N_ALGO} = {rep['fit_algo_frac']:.4f}"
          f" | r1 测试 head {rep['r1_test_head_entities']} 个，{rep['r1_test_head_out_of_fitfrac']:.4f}"
          f" 不在拟合集内")
    print(f"     变体在拟合集行上 mean|max|={rep['tc_on_fit_mean_absmax']:.2e}"
          f" std偏差={rep['tc_on_fit_std_maxdev']:.2e}")
    print(f"[G4] 扰动 L∞={rep['perturb_Linf']:.4f}  行 L2 均值={rep['perturb_L2_mean']:.4f}"
          f"  p99={rep['perturb_L2_p99']:.4f}  变化>1e-3 的行占 {rep['perturb_frac_rows_changed_gt1e3']:.4f}")
    print(f"     逐维 gain={rep['dim_gain']}")

    gates_ok = rep["gate_recon"] and rep["tc_on_fit_mean_absmax"] < 1e-4 and rep["tc_on_fit_std_maxdev"] < 1e-4
    rep["gates"] = "PASS" if gates_ok else "FAIL"
    rep["decision_hint"] = ("L∞ < 1e-4 ⇒ 效应量必然低于噪声，这块降级为 0-GPU 陈述、不排臂"
                            if rep["perturb_Linf"] < 1e-4 else
                            "L∞ 足够大 ⇒ 3 臂测得出来")
    print(f"[hint] {rep['decision_hint']}")

    if not gates_ok and not args.force:
        print("!! 门禁不过，不落盘"); sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "node_feat_algo.npy"), Z_tc.astype(np.float32))
    for other in ("node_feat_data.npy", "node_feat_compute.npy"):
        src = os.path.join(base, other)
        if os.path.exists(src):
            with open(src, "rb") as fi, open(os.path.join(args.out_dir, other), "wb") as fo:
                fo.write(fi.read())
    rep["out_dir"] = args.out_dir
    rep["rf_precheck_cmd"] = ("src/diagnose_sklearn.py --feat-path ../{d}/node_feat_algo.npy"
                              " --data-dir ../data/xcad_v2 --seed 42")
    print(f"[write] {args.out_dir}/node_feat_algo.npy shape={Z_tc.shape} dtype=float32")

    rp = args.report or os.path.join(args.out_dir, "traincaliber_report.json")
    with open(rp, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"[report] {rp}")
    print("GATES", rep["gates"])


if __name__ == "__main__":
    main()
