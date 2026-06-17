"""
xCAD Node Builder

Builds three types of nodes:
- Compute nodes (primary key: machine)
- Algorithm nodes (primary key: job_name) - 修正1: 作业级粒度，仅保留有 workload 标签的 job
- Data nodes (primary key: group) - 修正2: 加入 MIN_GROUP_SIZE 过滤
"""

import os
import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from src.utils.load_tables import load_all_tables
from src.utils.schema import GPU_TYPE_ORDER
from src.utils.config import OUTPUT_ROOT, MIN_GROUP_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = f"{OUTPUT_ROOT}/nodes"


def gpu_type_to_ordinal(gpu_type: str) -> int:
    """Convert gpu_type string to ordinal integer (older → newer)."""
    return GPU_TYPE_ORDER.get(gpu_type, 0)


def build_compute_nodes(
    machine_spec: pd.DataFrame,
    sensor_df: pd.DataFrame,
    machine_metric: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build Compute nodes with primary key = machine.

    Attributes:
    - Static: machine_spec columns (gpu_type, cap_cpu, cap_mem, cap_gpu)
    - Dynamic: sensor aggregated metrics (gpu_wrk_util, avg/max_gpu_wrk_mem, etc.)
              + machine_metric aggregated metrics (machine_load_1, machine_num_worker, etc.)
    """
    logger.info("Building Compute nodes...")

    spec_base = machine_spec.copy()
    spec_base["gpu_ordinal"] = spec_base["gpu_type"].apply(gpu_type_to_ordinal)

    sensor_agg = sensor_df.groupby("machine").agg({
        "gpu_wrk_util": ["mean", "max"],
        "avg_gpu_wrk_mem": "mean",
        "max_gpu_wrk_mem": "max",
        "avg_mem": "mean",
        "max_mem": "max",
        "cpu_usage": "mean",
        "read": "mean",
        "write": "mean",
    }).reset_index()
    sensor_agg.columns = [
        "machine", "avg_gpu_util", "max_gpu_util",
        "avg_gpu_mem", "max_gpu_mem",
        "avg_host_mem", "max_host_mem",
        "avg_cpu_usage", "avg_net_read", "avg_net_write"
    ]

    metric_agg = machine_metric.groupby("machine").agg({
        "machine_load_1": "mean",
        "machine_num_worker": "mean",
        "machine_cpu": "mean",
        "machine_gpu": "mean",
    }).reset_index()
    metric_agg.columns = [
        "machine", "avg_machine_load", "avg_num_worker",
        "avg_machine_cpu", "avg_machine_gpu"
    ]

    compute_nodes = spec_base.merge(sensor_agg, on="machine", how="left")
    compute_nodes = compute_nodes.merge(metric_agg, on="machine", how="left")

    compute_nodes = compute_nodes.rename(columns={
        "gpu_type": "machine_gpu_type",
        "cap_cpu": "machine_cap_cpu",
        "cap_mem": "machine_cap_mem",
        "cap_gpu": "machine_cap_gpu",
    })

    compute_nodes = compute_nodes.rename(columns={"gpu_ordinal": "gpu_type_ordinal"})

    logger.info(f"Compute nodes: {len(compute_nodes)} nodes")
    return compute_nodes


def build_algorithm_nodes(
    group_tag: pd.DataFrame,
    task_df: pd.DataFrame,
    instance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    修正1: Algorithm nodes with primary key = job_name.

    仅包含有 workload 标签的 job 子集 (~9.74%)。
    属性: workload 类别、task_name 角色构成、plan_cpu/mem/gpu 资源画像、实例数。
    注意: group_tag 表不含 job_name，需通过 instance_df 关联。
    """
    logger.info("Building Algorithm nodes (job-level granularity)...")

    workload_instances = group_tag[group_tag["workload"].notna()][["inst_id", "workload"]].copy()
    workload_inst_ids = set(workload_instances["inst_id"].unique())
    logger.info(f"  Instances with workload: {len(workload_inst_ids):,}")

    inst_to_job = instance_df[["inst_id", "job_name"]].drop_duplicates()
    workload_with_job = workload_instances.merge(inst_to_job, on="inst_id", how="inner")

    jobs_with_workload = workload_with_job[["job_name", "workload"]].drop_duplicates(subset=["job_name"])
    job_count = len(jobs_with_workload)
    logger.info(f"  Jobs with workload labels: {job_count:,}")

    if job_count > 200000:
        logger.warning(f"  ⚠️ Algorithm node count ({job_count:,}) significantly exceeds expected 'tens of thousands'. Stopping.")
        logger.warning("  Please verify the logic before proceeding.")
        return pd.DataFrame()

    workload_inst_ids = set(workload_instances["inst_id"].unique())
    instance_filtered = instance_df[instance_df["inst_id"].isin(workload_inst_ids)].copy()
    logger.info(f"  Filtered instances: {len(instance_filtered):,}")

    instance_with_job = instance_filtered.merge(
        jobs_with_workload,
        on="job_name",
        how="inner"
    )

    instance_with_task = instance_with_job.merge(
        task_df[["job_name", "task_name", "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]],
        on=["job_name", "task_name"],
        how="left",
        suffixes=("", "_task")
    )

    job_stats = instance_with_task.groupby("job_name").agg({
        "workload": "first",
        "plan_cpu": ["mean", "std", "min", "max"],
        "plan_mem": ["mean", "std", "min", "max"],
        "plan_gpu": ["mean", "std", "min", "max"],
        "worker_name": "count",
    }).reset_index()
    job_stats.columns = [
        "job_name", "workload",
        "avg_plan_cpu", "std_plan_cpu", "min_plan_cpu", "max_plan_cpu",
        "avg_plan_mem", "std_plan_mem", "min_plan_mem", "max_plan_mem",
        "avg_plan_gpu", "std_plan_gpu", "min_plan_gpu", "max_plan_gpu",
        "instance_count"
    ]

    role_dist_series = instance_with_task.groupby("job_name")["task_name"].apply(
        lambda x: x.value_counts().to_dict()
    )
    role_dist = pd.DataFrame({
        "job_name": role_dist_series.index,
        "role_distribution": role_dist_series.values
    })

    algorithm_nodes = job_stats.merge(role_dist, on="job_name", how="left")

    logger.info(f"Algorithm nodes: {len(algorithm_nodes):,} nodes (job-level, with workload label)")
    return algorithm_nodes


def build_data_nodes(
    group_tag: pd.DataFrame,
    sensor_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    修正2: Data nodes with primary key = group.

    仅保留实例数 >= MIN_GROUP_SIZE (5) 的 group。
    过滤 p50=1 的噪声单点。巨型 group (max=22642) 保留但标注为 outlier。
    属性: sensor IO aggregated metrics (read/write/read_count/write_count).
    """
    logger.info(f"Building Data nodes (MIN_GROUP_SIZE={MIN_GROUP_SIZE})...")

    group_instances = group_tag[group_tag["group"].notna()][["group", "inst_id"]].copy()
    all_groups = group_instances["group"].unique()
    logger.info(f"  Total unique groups (before filter): {len(all_groups):,}")

    group_sizes = group_instances.groupby("group")["inst_id"].nunique()
    logger.info(f"  Group size stats: p50={group_sizes.quantile(0.5):.0f}, p90={group_sizes.quantile(0.9):.0f}, max={group_sizes.max():,}")

    filtered_groups = set(group_sizes[group_sizes >= MIN_GROUP_SIZE].index)
    logger.info(f"  Groups with size >= {MIN_GROUP_SIZE}: {len(filtered_groups):,}")

    outlier_groups = set(group_sizes[group_sizes > 1000].index)
    if len(outlier_groups) > 0:
        logger.info(f"  Outlier groups (size > 1000): {len(outlier_groups)}")

    filtered_instances = group_instances[group_instances["group"].isin(filtered_groups)].copy()
    logger.info(f"  Filtered instances: {len(filtered_instances):,}")

    sensor_by_group = sensor_df.merge(
        filtered_instances[["inst_id", "group"]].drop_duplicates(),
        on="inst_id",
        how="inner"
    )

    data_node_stats = sensor_by_group.groupby("group").agg({
        "read": ["mean", "sum", "max"],
        "write": ["mean", "sum", "max"],
        "read_count": ["mean", "sum"],
        "write_count": ["mean", "sum"],
        "worker_name": "nunique",
    }).reset_index()
    data_node_stats.columns = [
        "group",
        "avg_read", "sum_read", "max_read",
        "avg_write", "sum_write", "max_write",
        "avg_read_count", "sum_read_count",
        "avg_write_count", "sum_write_count",
        "instance_count"
    ]

    data_node_stats["is_outlier"] = data_node_stats["group"].isin(outlier_groups)
    outlier_count = data_node_stats["is_outlier"].sum()
    logger.info(f"  Outlier nodes (size > 1000): {outlier_count}")

    logger.info(f"Data nodes: {len(data_node_stats):,} nodes (filtered by MIN_GROUP_SIZE={MIN_GROUP_SIZE})")
    return data_node_stats


def save_nodes(nodes: pd.DataFrame, node_type: str, output_dir: str = OUTPUT_DIR):
    """Save nodes to parquet with schema comments."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{node_type}_nodes.parquet")

    nodes.to_parquet(output_path, index=False)

    schema_comments = {
        "compute_nodes": {
            "machine": "主键，机器唯一标识",
            "machine_gpu_type": "GPU 代次类型 (MISC/V100/T4/P100...)",
            "gpu_type_ordinal": "GPU 代次序数 (老→新)",
            "machine_cap_cpu": "CPU 核数",
            "machine_cap_mem": "内存 GB",
            "machine_cap_gpu": "GPU 卡数 (0=纯CPU)",
            "avg_gpu_util": "GPU 使用率均值 (%)",
            "max_gpu_util": "GPU 使用率最大值 (%)",
            "avg_gpu_mem": "GPU 显存均值 (GB)",
            "max_gpu_mem": "GPU 显存最大值 (GB)",
            "avg_host_mem": "主机内存均值 (GB)",
            "max_host_mem": "主机内存最大值 (GB)",
            "avg_cpu_usage": "CPU 使用率均值 (%)",
            "avg_net_read": "网络读取均值 (字节)",
            "avg_net_write": "网络写入均值 (字节)",
            "avg_machine_load": "机器负载均值",
            "avg_num_worker": "共置实例数均值",
            "avg_machine_cpu": "机器 CPU 使用率均值",
            "avg_machine_gpu": "机器 GPU 使用率均值",
        },
        "algorithm_nodes": {
            "job_name": "主键，作业名称（仅包含有 workload 标签的 job）",
            "workload": "算法类型标签 (graphlearn/ctr/bert...)，约 9.74% 的 job 有此标签",
            "avg_plan_cpu": "计划 CPU 百分比均值",
            "std_plan_cpu": "计划 CPU 百分比标准差",
            "min_plan_cpu": "计划 CPU 百分比最小值",
            "max_plan_cpu": "计划 CPU 百分比最大值",
            "avg_plan_mem": "计划内存 GB 均值",
            "std_plan_mem": "计划内存 GB 标准差",
            "min_plan_mem": "计划内存 GB 最小值",
            "max_plan_mem": "计划内存 GB 最大值",
            "avg_plan_gpu": "计划 GPU 百分比均值",
            "std_plan_gpu": "计划 GPU 百分比标准差",
            "min_plan_gpu": "计划 GPU 百分比最小值",
            "max_plan_gpu": "计划 GPU 百分比最大值",
            "instance_count": "实例数量",
            "role_distribution": "角色分布字典 {ps/worker/evaluator: count}",
        },
        "data_nodes": {
            "group": "主键，数据源/汇分组",
            "avg_read": "网络读取均值 (字节)",
            "sum_read": "网络读取总和 (字节)",
            "max_read": "网络读取最大值 (字节)",
            "avg_write": "网络写入均值 (字节)",
            "sum_write": "网络写入总和 (字节)",
            "max_write": "网络写入最大值 (字节)",
            "avg_read_count": "读取次数均值",
            "sum_read_count": "读取次数总和",
            "avg_write_count": "写入次数均值",
            "sum_write_count": "写入次数总和",
            "instance_count": "实例数量",
            "is_outlier": "是否异常大 group (size > 1000)",
        }
    }

    schema_map = {
        "compute": "compute_nodes",
        "algorithm": "algorithm_nodes",
        "data": "data_nodes",
    }

    comment_path = os.path.join(output_dir, f"{node_type}_nodes.schema")
    with open(comment_path, "w") as f:
        schema_key = schema_map.get(node_type, f"{node_type}_nodes")
        if schema_key in schema_comments:
            for col, comment in schema_comments[schema_key].items():
                f.write(f"{col}: {comment}\n")

    logger.info(f"Saved {node_type} nodes to {output_path}")
    return output_path


def build_all_nodes() -> Dict[str, pd.DataFrame]:
    """Build all three types of nodes."""
    logger.info("Loading all tables...")
    tables = load_all_tables()

    machine_spec = tables["pai_machine_spec"]
    sensor_df = tables["pai_sensor_table"]
    machine_metric = tables["pai_machine_metric"]
    group_tag = tables["pai_group_tag_table"]
    task_df = tables["pai_task_table"]
    instance_df = tables["pai_instance_table"]

    logger.info("=" * 60)
    compute_nodes = build_compute_nodes(machine_spec, sensor_df, machine_metric)
    logger.info("=" * 60)
    algorithm_nodes = build_algorithm_nodes(group_tag, task_df, instance_df)
    logger.info("=" * 60)
    data_nodes = build_data_nodes(group_tag, sensor_df)
    logger.info("=" * 60)

    logger.info("\nGPU Type Ordinal Mapping (old → new):")
    for gpu_type, ordinal in sorted(GPU_TYPE_ORDER.items(), key=lambda x: x[1]):
        logger.info(f"  {gpu_type}: {ordinal}")

    return {
        "compute": compute_nodes,
        "algorithm": algorithm_nodes,
        "data": data_nodes,
    }


def main():
    """Main entry point."""
    nodes = build_all_nodes()

    logger.info("\n" + "=" * 60)
    logger.info("Saving nodes to parquet...")
    logger.info("=" * 60)

    for node_type, df in nodes.items():
        save_nodes(df, node_type)

    logger.info("\n" + "=" * 60)
    logger.info("Node summary:")
    logger.info("=" * 60)
    for node_type, df in nodes.items():
        logger.info(f"  {node_type}: {len(df):,} nodes, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
