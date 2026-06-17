"""
xCAD Edge Builder

Builds time-varying edges with tau annotation for each time window.
Key fixes from review:
1. placement: src is worker_name → fixed to job_name via instance table mapping
2. r1/r2/r3: Now per-τ aggregation (each window generates its own edges)

Each parquet output contains tau column, with multiple edges per τ for time-varying relations.
"""

import os
import sys
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.load_tables import load_all_tables
from src.utils.schema import GPU_TYPE_ORDER, STATUS_SUCCESS
from src.utils.config import OUTPUT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = f"{OUTPUT_ROOT}/edges"
RELATIVE_TIME_GRANULARITY = "day"
R4_SHIFTS_THRESHOLD = 0.05
NUM_TIME_WINDOWS = 69


def compute_tau(timestamps: pd.Series, min_ts: float) -> pd.Series:
    """Convert absolute timestamps to relative time window τ ∈ [1, 69]."""
    tau = ((timestamps - min_ts) / 86400).astype(float)
    tau = tau.fillna(-1)
    tau = tau.clip(lower=1, upper=NUM_TIME_WINDOWS)
    tau = tau.round().astype(int)
    return tau


def build_placement_edges(instance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build placement edges: job → machine.
    
    Key fix: Map worker_name → job_name, create edges at job level.
    """
    logger.info("Building placement edges (job → machine)...")

    df = instance_df.copy()
    
    if "start_time" in df.columns:
        min_ts = df["start_time"].min()
        df["tau"] = compute_tau(df["start_time"], min_ts)
    
    df["worker_to_job"] = df["job_name"]
    
    edges = df[["job_name", "machine", "tau"]].drop_duplicates()
    edges = edges.rename(columns={
        "job_name": "src",
        "machine": "dst"
    })
    edges["edge_type"] = "placement"
    edges["weight"] = 1.0

    logger.info(f"  Placement edges: {len(edges):,}")
    tau_dist = edges["tau"].value_counts().sort_index()
    logger.info(f"  tau distribution (sample):")
    for tau_val in [1, 10, 20, 30, 40, 50, 60, 69]:
        if tau_val in tau_dist.index:
            logger.info(f"    τ={tau_val}: {tau_dist[tau_val]:,} edges")
    
    return edges


def build_r1_suits_edges_per_tau(
    group_tag: pd.DataFrame,
    task_df: pd.DataFrame,
    instance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build r1_suits edges: Algorithm → GPU Type (per time window).
    
    For each τ, aggregate instances within that window to compute success_rate.
    """
    logger.info("Building r1_suits edges (per τ aggregation)...")

    workload_mask = group_tag["workload"].notna()
    workload_inst = group_tag[workload_mask][["inst_id", "workload"]].copy()
    
    inst_to_job = instance_df[["inst_id", "job_name", "start_time", "worker_name"]].copy()
    inst_to_job = inst_to_job.drop_duplicates(subset=["inst_id"])
    
    workload_with_job = workload_inst.merge(inst_to_job, on="inst_id", how="inner")
    
    if "start_time" in workload_with_job.columns and workload_with_job["start_time"].notna().any():
        min_ts = workload_with_job["start_time"].min()
        workload_with_job["tau"] = compute_tau(workload_with_job["start_time"], min_ts)
    
    workload_jobs = set(workload_with_job["job_name"].unique())
    logger.info(f"  Jobs with workload labels: {len(workload_jobs):,}")
    
    job_gpu = instance_df[instance_df["job_name"].isin(workload_jobs)].merge(
        task_df[["job_name", "task_name", "gpu_type"]],
        on=["job_name", "task_name"],
        how="left"
    )
    
    if "start_time" in job_gpu.columns and job_gpu["start_time"].notna().any():
        min_ts = job_gpu["start_time"].min()
        job_gpu["tau"] = compute_tau(job_gpu["start_time"], min_ts)
    
    all_edges = []
    
    for tau_val in range(1, NUM_TIME_WINDOWS + 1):
        window_data = job_gpu[job_gpu["tau"] == tau_val]
        
        if len(window_data) == 0:
            continue
        
        group_stats = window_data.groupby(["job_name", "gpu_type"]).agg({
            "worker_name": "count",
            "status": lambda x: (x == STATUS_SUCCESS).sum()
        }).reset_index()
        group_stats.columns = ["job_name", "gpu_type", "cooccur_count", "terminated_count"]
        group_stats["success_rate"] = group_stats["terminated_count"] / group_stats["cooccur_count"]
        
        edges = group_stats[group_stats["cooccur_count"] > 0][
            ["job_name", "gpu_type", "success_rate", "cooccur_count"]
        ].copy()
        edges["tau"] = tau_val
        edges = edges.rename(columns={
            "job_name": "src",
            "gpu_type": "dst",
        })
        edges["edge_type"] = "r1_suits"
        edges["weight"] = edges["success_rate"]
        
        all_edges.append(edges)
    
    if all_edges:
        result = pd.concat(all_edges, ignore_index=True)
        logger.info(f"  r1_suits edges: {len(result):,}")
        logger.info(f"    τ distribution (sample):")
        for tau_val in [1, 10, 20, 30, 40, 50, 60]:
            count = len(result[result["tau"] == tau_val])
            if count > 0:
                logger.info(f"      τ={tau_val}: {count:,} edges")
        return result
    else:
        return pd.DataFrame(columns=["src", "dst", "success_rate", "cooccur_count", "tau", "edge_type", "weight"])


def build_r2_requires_edges_per_tau(
    group_tag: pd.DataFrame,
    instance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build r2_requires edges: Algorithm → GPU Type (per time window).
    """
    logger.info("Building r2_requires edges (per τ aggregation)...")

    gpu_spec_rows = group_tag[
        (group_tag["workload"].notna()) &
        (group_tag["gpu_type_spec"].notna()) &
        (group_tag["gpu_type_spec"] != "")
    ].copy()

    if len(gpu_spec_rows) == 0:
        return pd.DataFrame(columns=["src", "dst", "weight", "count", "tau", "edge_type"])

    inst_to_job = instance_df[["inst_id", "job_name", "start_time"]].drop_duplicates(subset=["inst_id"])
    gpu_spec_with_job = gpu_spec_rows.merge(inst_to_job, on="inst_id", how="inner")

    if "start_time" in gpu_spec_with_job.columns and gpu_spec_with_job["start_time"].notna().any():
        min_ts = gpu_spec_with_job["start_time"].min()
        gpu_spec_with_job["tau"] = compute_tau(gpu_spec_with_job["start_time"], min_ts)

    all_edges = []
    
    for tau_val in range(1, NUM_TIME_WINDOWS + 1):
        window_data = gpu_spec_with_job[gpu_spec_with_job["tau"] == tau_val]
        
        if len(window_data) == 0:
            continue
        
        co_occurrence = window_data.groupby(["job_name", "gpu_type_spec"]).agg({
            "inst_id": "count"
        }).reset_index()
        co_occurrence.columns = ["job_name", "gpu_type_spec", "count"]
        
        if co_occurrence["count"].nunique() == 1:
            co_occurrence["weight"] = co_occurrence["count"].astype(float)
        else:
            co_occurrence["weight"] = np.log1p(co_occurrence["count"])

        edges = co_occurrence[["job_name", "gpu_type_spec", "weight", "count"]].copy()
        edges["tau"] = tau_val
        edges = edges.rename(columns={
            "job_name": "src",
            "gpu_type_spec": "dst",
        })
        edges["edge_type"] = "r2_requires"
        
        all_edges.append(edges)
    
    if all_edges:
        result = pd.concat(all_edges, ignore_index=True)
        logger.info(f"  r2_requires edges: {len(result):,}")
        return result
    else:
        return pd.DataFrame(columns=["src", "dst", "weight", "count", "tau", "edge_type"])


def build_r3_drives_edges_per_tau(
    group_tag: pd.DataFrame,
    instance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build r3_drives edges: Data → Algorithm (per time window).
    """
    logger.info("Building r3_drives edges (per τ aggregation)...")

    inst_to_job = instance_df[["inst_id", "job_name", "start_time"]].drop_duplicates(subset=["inst_id"])
    group_with_job = group_tag.merge(inst_to_job, on="inst_id", how="inner")

    if "start_time" in group_with_job.columns and group_with_job["start_time"].notna().any():
        min_ts = group_with_job["start_time"].min()
        group_with_job["tau"] = compute_tau(group_with_job["start_time"], min_ts)

    all_edges = []
    
    for tau_val in range(1, NUM_TIME_WINDOWS + 1):
        window_data = group_with_job[
            (group_with_job["tau"] == tau_val) &
            (group_with_job["group"].notna()) &
            (group_with_job["workload"].notna())
        ]
        
        if len(window_data) == 0:
            continue
        
        co_occurrence = window_data.groupby(["group", "job_name"]).agg({
            "inst_id": "count"
        }).reset_index()
        co_occurrence.columns = ["group", "job_name", "count"]
        
        if co_occurrence["count"].nunique() == 1:
            co_occurrence["weight"] = co_occurrence["count"].astype(float)
        else:
            co_occurrence["weight"] = np.log1p(co_occurrence["count"])

        edges = co_occurrence[["group", "job_name", "weight", "count"]].copy()
        edges["tau"] = tau_val
        edges = edges.rename(columns={
            "group": "src",
            "job_name": "dst",
        })
        edges["edge_type"] = "r3_drives"
        
        all_edges.append(edges)
    
    if all_edges:
        result = pd.concat(all_edges, ignore_index=True)
        logger.info(f"  r3_drives edges: {len(result):,}")
        logger.info(f"    τ distribution (sample):")
        for tau_val in [1, 10, 20, 30, 40, 50, 60]:
            count = len(result[result["tau"] == tau_val])
            if count > 0:
                logger.info(f"      τ={tau_val}: {count:,} edges")
        return result
    else:
        return pd.DataFrame(columns=["src", "dst", "weight", "count", "tau", "edge_type"])


def build_r4_shifts_edges(
    instance_df: pd.DataFrame,
    task_df: pd.DataFrame,
    granularity: str = RELATIVE_TIME_GRANULARITY,
    threshold: float = R4_SHIFTS_THRESHOLD,
) -> pd.DataFrame:
    """
    Build r4_shifts edges: GPU Type → GPU Type (cross time window).
    """
    logger.info(f"Building r4_shifts edges...")

    instance_with_task = instance_df.merge(
        task_df[["job_name", "task_name", "gpu_type"]],
        on=["job_name", "task_name"],
        how="left"
    )

    instance_with_task = instance_with_task.dropna(subset=["start_time"])

    min_time = instance_with_task["start_time"].min()
    if granularity == "day":
        instance_with_task["time_window"] = (
            ((instance_with_task["start_time"] - min_time) / 86400).astype(int)
        )
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    gpu_load = instance_with_task.groupby(["time_window", "gpu_type"]).size().reset_index()
    gpu_load.columns = ["time_window", "gpu_type", "count"]

    window_totals = gpu_load.groupby("time_window")["count"].sum().reset_index()
    window_totals.columns = ["time_window", "total"]
    gpu_load = gpu_load.merge(window_totals, on="time_window")
    gpu_load["share"] = gpu_load["count"] / gpu_load["total"]

    windows = sorted(gpu_load["time_window"].unique())
    all_shifts = []

    for i in range(len(windows) - 1):
        w1, w2 = windows[i], windows[i + 1]

        share1 = gpu_load[gpu_load["time_window"] == w1][["gpu_type", "share"]].set_index("gpu_type")["share"]
        share2 = gpu_load[gpu_load["time_window"] == w2][["gpu_type", "share"]].set_index("gpu_type")["share"]

        all_gpus = set(share1.index) | set(share2.index)
        for gpu in all_gpus:
            s1 = share1.get(gpu, 0)
            s2 = share2.get(gpu, 0)
            change = abs(s2 - s1)
            all_shifts.append({
                "src": gpu,
                "dst": gpu,
                "weight": change,
                "window_pair": f"{w1}→{w2}"
            })

    shifts_df = pd.DataFrame(all_shifts) if all_shifts else pd.DataFrame()

    if len(shifts_df) > 0:
        shifts_df["edge_type"] = "r4_shifts"
        shifts_df["tau"] = -1

        filtered = shifts_df[shifts_df["weight"] > threshold].copy()

        if len(filtered) > 0:
            agg_weights = filtered.groupby(["src", "dst"])["weight"].mean().reset_index()
            agg_weights["edge_type"] = "r4_shifts"
            agg_weights["tau"] = -1

            logger.info(f"  r4_shifts edges: {len(agg_weights):,}")
            return agg_weights

    return pd.DataFrame(columns=["src", "dst", "weight", "edge_type", "tau"])


def save_edges(edges: pd.DataFrame, edge_type: str, output_dir: str = OUTPUT_DIR):
    """Save edges to parquet."""
    if len(edges) == 0:
        logger.warning(f"  No edges to save for {edge_type}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{edge_type}_edges.parquet")

    edges.to_parquet(output_path, index=False)
    logger.info(f"  Saved {edge_type} edges to {output_path}")
    logger.info(f"    Columns: {edges.columns.tolist()}")
    logger.info(f"    Shape: {edges.shape}")

    return output_path


def build_all_edges(
    granularity: str = RELATIVE_TIME_GRANULARITY,
    r4_threshold: float = R4_SHIFTS_THRESHOLD,
) -> Dict[str, pd.DataFrame]:
    """Build all five types of edges with time-varying tau."""
    logger.info("Loading all tables...")
    tables = load_all_tables()

    instance_df = tables["pai_instance_table"]
    task_df = tables["pai_task_table"]
    group_tag = tables["pai_group_tag_table"]

    logger.info("=" * 60)
    placement_edges = build_placement_edges(instance_df)
    logger.info("=" * 60)
    r1_edges = build_r1_suits_edges_per_tau(group_tag, task_df, instance_df)
    logger.info("=" * 60)
    r2_edges = build_r2_requires_edges_per_tau(group_tag, instance_df)
    logger.info("=" * 60)
    r3_edges = build_r3_drives_edges_per_tau(group_tag, instance_df)
    logger.info("=" * 60)
    r4_edges = build_r4_shifts_edges(instance_df, task_df, granularity, r4_threshold)
    logger.info("=" * 60)

    return {
        "placement": placement_edges,
        "r1_suits": r1_edges,
        "r2_requires": r2_edges,
        "r3_drives": r3_edges,
        "r4_shifts": r4_edges,
    }


def main():
    """Main entry point."""
    edges = build_all_edges()

    logger.info("\n" + "=" * 60)
    logger.info("Saving edges to parquet...")
    logger.info("=" * 60)

    for edge_type, df in edges.items():
        save_edges(df, edge_type)

    logger.info("\n" + "=" * 60)
    logger.info("Edge summary:")
    logger.info("=" * 60)
    for edge_type, df in edges.items():
        tau_info = f", τ range: [{df['tau'].min()}, {df['tau'].max()}]" if "tau" in df.columns else ""
        logger.info(f"  {edge_type}: {len(df):,} edges{tau_info}")


if __name__ == "__main__":
    main()