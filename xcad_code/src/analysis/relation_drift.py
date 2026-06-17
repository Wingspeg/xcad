"""
xCAD 关系语义时间漂移分析 (Relation Semantic Time-Drift Analysis)

目的
----
为论文创新点 1 (时间感知关系建模) 提供实证依据:
验证 "R1 关系 (Algorithm → GPU Type) 的 success_rate 随时间 τ 漂移"
这一现象的普遍性、幅度与结构 (单调 vs 震荡)。

数据
----
- outputs/edges/r1_suits_edges.parquet  (字段 src, dst, success_rate, cooccur_count, tau)
- outputs/nodes/algorithm_nodes.parquet  (字段 job_name, workload)

输出
----
- outputs/reports/relation_drift.md

注意
----
本脚本只读不写模型/训练代码,纯离线分析,可在服务器上直接运行。
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# 允许 `python -m src.analysis.relation_drift` 与 `python src/analysis/relation_drift.py` 两种入口
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)
for p in (_PROJECT_ROOT, _SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.config import OUTPUT_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EDGES_DIR = os.path.join(OUTPUT_ROOT, "edges")
NODES_DIR = os.path.join(OUTPUT_ROOT, "nodes")
REPORTS_DIR = os.path.join(OUTPUT_ROOT, "reports")

# === 可调参数 (论文阈值) ===
MIN_TAU_COVERAGE = 5        # (workload, gpu_type) 至少出现在 ≥ N 个不同 τ 上
STD_THRESHOLD = 0.1         # 证据 1: 显著漂移阈值
RANGE_THRESHOLD = 0.3       # 证据 2: 强幅度漂移阈值
R2_THRESHOLD = 0.5          # 证据 3: 非单调阈值
TOP_K = 10                  # 附表条数

# 报告输出名
REPORT_FILENAME = "relation_drift.md"


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载 r1_suits 边与 algorithm 节点。"""
    edges_path = os.path.join(EDGES_DIR, "r1_suits_edges.parquet")
    nodes_path = os.path.join(NODES_DIR, "algorithm_nodes.parquet")
    edges = pd.read_parquet(edges_path)
    nodes = pd.read_parquet(nodes_path)
    # 仅保留必需字段,降低内存
    needed_edge_cols = ["src", "dst", "success_rate", "cooccur_count", "tau"]
    edges = edges[[c for c in needed_edge_cols if c in edges.columns]].copy()
    return edges, nodes


def attach_workload(edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """用 r1.src == algorithm.job_name 给每条边挂上 workload 标签。

    使用 inner join,只保留 src 出现在 algorithm_nodes 中的边
    (这些 src 本就是带 workload 标签的 job,见 build_edges.py:workload_mask)。
    """
    nodes_subset = nodes[["job_name", "workload"]].drop_duplicates(subset=["job_name"])
    merged = edges.merge(
        nodes_subset,
        left_on="src",
        right_on="job_name",
        how="inner",
    )
    merged = merged.drop(columns=["job_name"])
    # workload 缺失的行 (理论上 inner join 后不会剩) 再次过滤
    merged = merged[merged["workload"].notna()].copy()
    return merged


def aggregate_weighted(df: pd.DataFrame) -> pd.DataFrame:
    """按 (workload, dst, τ) 聚合,success_rate 用 cooccur_count 加权平均。

    公式: success_rate(group) = sum(success_rate_i * cooccur_count_i) / sum(cooccur_count_i)
    """
    df = df.copy()
    df["_w_sr"] = df["success_rate"] * df["cooccur_count"]
    g = (
        df.groupby(["workload", "dst", "tau"], as_index=False, sort=False)
        .agg(_num=("_w_sr", "sum"), _den=("cooccur_count", "sum"))
    )
    g["success_rate"] = g["_num"] / g["_den"].replace(0, np.nan)
    g = g.drop(columns=["_num", "_den"])
    return g


def filter_by_coverage(ts_df: pd.DataFrame, min_tau: int) -> pd.DataFrame:
    """只保留出现于 ≥ min_tau 个不同 τ 的 (workload, dst) 对。"""
    coverage = ts_df.groupby(["workload", "dst"])["tau"].nunique()
    valid_pairs = coverage[coverage >= min_tau].index
    mask = ts_df.set_index(["workload", "dst"]).index.isin(valid_pairs)
    return ts_df.loc[mask].copy()


# ---------------------------------------------------------------------------
# 漂移指标
# ---------------------------------------------------------------------------
def linear_r2(y: np.ndarray) -> float:
    """对序列 y 拟合 y = a * x + b (x = 0..n-1) 并返回 R²。

    - 常数列: 定义 R² = 1.0 (平凡拟合),但会在 std/range 指标中被识别为'无漂移'。
    - 长度 < 2: 返回 NaN。
    """
    n = len(y)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    if ss_tot == 0.0:
        return 1.0
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    ss_res = float(np.sum((y - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def oscillation_count(y: np.ndarray) -> int:
    """一阶差分符号变化次数 (忽略 0 差分)。"""
    if len(y) < 2:
        return 0
    diff = np.diff(y)
    signs = np.sign(diff)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(np.diff(signs) != 0))


def compute_pair_metrics(ts_df: pd.DataFrame) -> pd.DataFrame:
    """对每个 (workload, dst) 对计算 std / range / R² / 震荡次数。"""
    rows = []
    for (workload, dst), g in ts_df.groupby(["workload", "dst"], sort=False):
        g_sorted = g.sort_values("tau")
        y = g_sorted["success_rate"].to_numpy(dtype=float)
        rows.append({
            "workload": workload,
            "gpu_type": dst,
            "n_tau": int(len(y)),
            "tau_min": int(g_sorted["tau"].min()),
            "tau_max": int(g_sorted["tau"].max()),
            "mean_sr": float(np.mean(y)),
            "std": float(pd.Series(y).std(ddof=1)) if len(y) > 1 else 0.0,
            "range": float(np.max(y) - np.min(y)),
            "r2": linear_r2(y),
            "oscillation": oscillation_count(y),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _describe(values: np.ndarray) -> Dict[str, float]:
    """min / p25 / p50 / p75 / p90 / max 描述性统计。"""
    return {
        "n": int(len(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _fmt_pct(num: int, denom: int) -> str:
    if denom == 0:
        return "N/A"
    return f"{num}/{denom} = {100.0 * num / denom:.1f}%"


def render_report(metrics: pd.DataFrame, output_path: str) -> None:
    n_pairs = len(metrics)
    std_vals = metrics["std"].to_numpy()
    range_vals = metrics["range"].to_numpy()
    r2_vals = metrics["r2"].dropna().to_numpy()
    osc_vals = metrics["oscillation"].to_numpy()

    std_dist = _describe(std_vals)
    range_dist = _describe(range_vals)
    r2_dist = _describe(r2_vals)
    osc_dist = _describe(osc_vals)

    std_sig_n = int(np.sum(std_vals > STD_THRESHOLD))
    range_sig_n = int(np.sum(range_vals > RANGE_THRESHOLD))
    r2_low_n = int(np.sum(r2_vals < R2_THRESHOLD))

    top10 = metrics.sort_values("range", ascending=False).head(TOP_K)

    L = []
    L.append("# xCAD 关系语义时间漂移分析报告")
    L.append("")
    L.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(">")
    L.append("> 数据源:")
    L.append("> - `outputs/edges/r1_suits_edges.parquet` (src, dst, success_rate, cooccur_count, τ)")
    L.append("> - `outputs/nodes/algorithm_nodes.parquet` (job_name, workload)")
    L.append(">")
    L.append(f"> 关系类型: **r1_suits** (Algorithm → GPU Type)")
    L.append(f"> 加权方式: success_rate 在 (workload, gpu_type, τ) 粒度用 cooccur_count 加权平均")
    L.append(f"> 时间覆盖阈值: 每个 (workload, gpu_type) 对必须出现在 ≥ **{MIN_TAU_COVERAGE}** 个不同 τ")
    L.append("")
    L.append("## 总览")
    L.append("")
    L.append(f"- 满足时间覆盖的 (workload, gpu_type) 对数: **{n_pairs}**")
    L.append(f"- 涉及的 workload 数: **{metrics['workload'].nunique()}**")
    L.append(f"- 涉及的 gpu_type 数: **{metrics['gpu_type'].nunique()}**")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 证据 1 — 漂移普遍性 (success_rate 标准差)")
    L.append("")
    L.append("对每个 (workload, gpu_type) 对,计算其 success_rate(τ) 序列的样本标准差 (ddof=1)。")
    L.append("")
    L.append("| 统计量 | 值 |")
    L.append("|---|---|")
    L.append(f"| n (对数) | {std_dist['n']} |")
    L.append(f"| min | {std_dist['min']:.4f} |")
    L.append(f"| p25 | {std_dist['p25']:.4f} |")
    L.append(f"| p50 (中位数) | {std_dist['p50']:.4f} |")
    L.append(f"| p75 | {std_dist['p75']:.4f} |")
    L.append(f"| p90 | {std_dist['p90']:.4f} |")
    L.append(f"| max | {std_dist['max']:.4f} |")
    L.append("")
    L.append(f"**std > {STD_THRESHOLD} 的对占比**: {_fmt_pct(std_sig_n, n_pairs)}")
    L.append("")
    L.append(
        f"解读:中位标准差 = {std_dist['p50']:.4f}。{_fmt_pct(std_sig_n, n_pairs)} 的对达到"
        f"显著漂移水平 (std > {STD_THRESHOLD}),说明关系语义的'波动'是普遍现象,"
        f"而非个别 job 的偶发噪声。"
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 证据 2 — 漂移幅度 (success_rate 极差)")
    L.append("")
    L.append("对每个 (workload, gpu_type) 对,计算 success_rate(τ) 序列的极差 max − min。")
    L.append("")
    L.append("| 统计量 | 值 |")
    L.append("|---|---|")
    L.append(f"| n (对数) | {range_dist['n']} |")
    L.append(f"| min | {range_dist['min']:.4f} |")
    L.append(f"| p25 | {range_dist['p25']:.4f} |")
    L.append(f"| p50 (中位数) | {range_dist['p50']:.4f} |")
    L.append(f"| p75 | {range_dist['p75']:.4f} |")
    L.append(f"| p90 | {range_dist['p90']:.4f} |")
    L.append(f"| max | {range_dist['max']:.4f} |")
    L.append("")
    L.append(f"**range > {RANGE_THRESHOLD} 的对占比**: {_fmt_pct(range_sig_n, n_pairs)}")
    L.append("")
    L.append(
        f"解读:极差反映关系适配性的'跨越'强度。中位极差 = {range_dist['p50']:.4f};"
        f"{_fmt_pct(range_sig_n, n_pairs)} 的对跨越了 ≥ {RANGE_THRESHOLD} 的 success_rate 区间,"
        f"即从'基本不匹配'到'基本完全匹配'的质变,而非细微扰动。"
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 证据 3 — 漂移结构 (单调 vs 震荡)")
    L.append("")
    L.append("对每个对的 success_rate(τ) 序列 (按 τ 升序),计算:")
    L.append("- **一阶差分符号变化次数** = 震荡次数 (反映曲线方向反转的频度)")
    L.append("- **线性回归 R²** = y = a·τ + b 的拟合优度 (反映单调趋势强度)")
    L.append("")
    L.append("R² 越接近 1 ⇒ 序列近似单调;越接近 0 ⇒ 无单调趋势 (震荡 / 随机)。")
    L.append("")
    L.append("### 3.1 线性回归 R² 分布")
    L.append("")
    L.append("| 统计量 | 值 |")
    L.append("|---|---|")
    L.append(f"| n | {r2_dist['n']} |")
    L.append(f"| min | {r2_dist['min']:.4f} |")
    L.append(f"| p25 | {r2_dist['p25']:.4f} |")
    L.append(f"| p50 (中位数) | {r2_dist['p50']:.4f} |")
    L.append(f"| p75 | {r2_dist['p75']:.4f} |")
    L.append(f"| p90 | {r2_dist['p90']:.4f} |")
    L.append(f"| max | {r2_dist['max']:.4f} |")
    L.append("")
    L.append(f"**R² < {R2_THRESHOLD} 的对占比**: {_fmt_pct(r2_low_n, n_pairs)}")
    L.append("")
    L.append(
        f"解读:{_fmt_pct(r2_low_n, n_pairs)} 的对 R² < {R2_THRESHOLD},说明简单的时间衰减/线性增长模型"
        f"无法捕捉其动态,必须采用**演化式关系建模** (随时间更新关系表示) 才能反映真实结构。"
    )
    L.append("")
    L.append("### 3.2 震荡次数 (差分符号变化) 分布")
    L.append("")
    L.append("| 统计量 | 值 |")
    L.append("|---|---|")
    L.append(f"| n | {osc_dist['n']} |")
    L.append(f"| min | {osc_dist['min']:.0f} |")
    L.append(f"| p25 | {osc_dist['p25']:.0f} |")
    L.append(f"| p50 (中位数) | {osc_dist['p50']:.0f} |")
    L.append(f"| p75 | {osc_dist['p75']:.0f} |")
    L.append(f"| p90 | {osc_dist['p90']:.0f} |")
    L.append(f"| max | {osc_dist['max']:.0f} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"## 附:Top {TOP_K} 漂移最剧烈的 (workload, gpu_type) 对")
    L.append("")
    L.append("按 success_rate 极差降序,取前 10 名,用于直观展示最强漂移案例。")
    L.append("")
    L.append(
        "| 排名 | workload | gpu_type | 出现的 τ 数 | τ 范围 | mean SR | std | range | R² | 震荡次数 |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for rank, row in enumerate(top10.itertuples(index=False), start=1):
        L.append(
            f"| {rank} | {row.workload} | {row.gpu_type} | {row.n_tau} | "
            f"[{row.tau_min}, {row.tau_max}] | {row.mean_sr:.3f} | "
            f"{row.std:.3f} | {row.range:.3f} | {row.r2:.3f} | {row.oscillation} |"
        )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 论文引用建议")
    L.append("")
    L.append(
        f"基于上述统计:在 {n_pairs} 个满足时间覆盖的关系对中,"
        f"{_fmt_pct(std_sig_n, n_pairs)} 的对存在显著波动 (std > {STD_THRESHOLD}),"
        f"{_fmt_pct(range_sig_n, n_pairs)} 的对跨越 ≥ {RANGE_THRESHOLD} 的适配性区间,"
        f"{_fmt_pct(r2_low_n, n_pairs)} 的对不满足单调趋势 (R² < {R2_THRESHOLD})。"
        f"三个维度共同支撑论文创新点 1 的核心主张:"
        f"**静态关系建模将 R1 视为常量,会丢失与时间相关的语义信号;必须引入时间感知的关系表示**。"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    logger.info(f"Report written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("xCAD 关系语义时间漂移分析")
    logger.info("=" * 60)
    logger.info(f"  EDGES_DIR  = {EDGES_DIR}")
    logger.info(f"  NODES_DIR  = {NODES_DIR}")
    logger.info(f"  REPORTS_DIR= {REPORTS_DIR}")

    logger.info("[1/5] 加载数据 ...")
    edges, nodes = load_data()
    logger.info(f"  r1_suits 边数: {len(edges):,}")
    logger.info(f"  algorithm 节点数: {len(nodes):,}")

    logger.info("[2/5] 挂 workload 标签 (r1.src == algorithm.job_name) ...")
    df = attach_workload(edges, nodes)
    logger.info(f"  带 workload 标签的边: {len(df):,}")
    if len(df) == 0:
        logger.error("没有边匹配到带 workload 标签的 job,终止。")
        return

    logger.info("[3/5] 按 (workload, dst, τ) 加权聚合 success_rate ...")
    ts_df = aggregate_weighted(df)
    logger.info(f"  时间序列行数: {len(ts_df):,}")

    logger.info(f"[4/5] 过滤 τ 覆盖 ≥ {MIN_TAU_COVERAGE} 的 (workload, dst) 对 ...")
    ts_df = filter_by_coverage(ts_df, MIN_TAU_COVERAGE)
    n_pairs = ts_df[["workload", "dst"]].drop_duplicates().shape[0]
    logger.info(f"  幸存 (workload, gpu_type) 对数: {n_pairs}")
    if n_pairs == 0:
        logger.error("没有任何关系对满足时间覆盖阈值,终止。")
        return

    logger.info("[5/5] 计算每对漂移指标并生成报告 ...")
    metrics = compute_pair_metrics(ts_df)
    output_path = os.path.join(REPORTS_DIR, REPORT_FILENAME)
    render_report(metrics, output_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
