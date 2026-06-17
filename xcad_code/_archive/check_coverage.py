"""
xCAD Coverage Report Generator

Generates coverage_report.md with:
- workload coverage in group_tag
- group size distribution
- gpu_type distribution
- status distribution
- join completeness check
"""

import os
import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from src.utils.load_tables import load_all_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def calculate_workload_coverage(group_tag: pd.DataFrame) -> Dict:
    """Calculate workload coverage statistics."""
    total_inst_ids = group_tag["inst_id"].nunique()
    workload_instances = group_tag[group_tag["workload"].notna()]
    workload_inst_ids = workload_instances["inst_id"].nunique()

    coverage = workload_inst_ids / total_inst_ids * 100 if total_inst_ids > 0 else 0

    workload_counts = (
        workload_instances["workload"]
        .value_counts()
        .head(20)
        .to_dict()
    )

    return {
        "total_inst_ids": total_inst_ids,
        "workload_inst_ids": workload_inst_ids,
        "coverage_percent": coverage,
        "top_20_workloads": workload_counts,
    }


def calculate_group_distribution(group_tag: pd.DataFrame) -> Dict:
    """Calculate group size distribution statistics."""
    group_counts = group_tag.groupby("group")["inst_id"].nunique()

    return {
        "total_groups": len(group_counts),
        "group_inst_count": {
            "p50": float(group_counts.quantile(0.5)),
            "p90": float(group_counts.quantile(0.9)),
            "p99": float(group_counts.quantile(0.99)),
            "max": float(group_counts.max()),
            "mean": float(group_counts.mean()),
            "std": float(group_counts.std()),
        },
        "large_group_check": group_counts.max() > group_counts.median() * 100,
    }


def calculate_gpu_type_distribution(task_df: pd.DataFrame, machine_spec: pd.DataFrame) -> Dict:
    """Calculate gpu_type distribution in task and machine_spec tables."""
    task_gpu = task_df["gpu_type"].value_counts().to_dict()

    spec_gpu = machine_spec["gpu_type"].value_counts().to_dict()

    return {
        "task_gpu_type_counts": task_gpu,
        "machine_spec_gpu_type_counts": spec_gpu,
    }


def calculate_status_distribution(
    job_df: pd.DataFrame, task_df: pd.DataFrame, instance_df: pd.DataFrame
) -> Dict:
    """Calculate status distribution across job/task/instance tables."""
    job_status = job_df["status"].value_counts().to_dict()
    task_status = task_df["status"].value_counts().to_dict()
    instance_status = instance_df["status"].value_counts().to_dict()

    job_terminated_pct = (
        job_status.get("Terminated", 0) / len(job_df) * 100 if len(job_df) > 0 else 0
    )
    task_terminated_pct = (
        task_status.get("Terminated", 0) / len(task_df) * 100 if len(task_df) > 0 else 0
    )
    instance_terminated_pct = (
        instance_status.get("Terminated", 0) / len(instance_df) * 100 if len(instance_df) > 0 else 0
    )

    return {
        "job_status": job_status,
        "task_status": task_status,
        "instance_status": instance_status,
        "terminated_percent": {
            "job": job_terminated_pct,
            "task": task_terminated_pct,
            "instance": instance_terminated_pct,
        },
    }


def calculate_join_completeness(
    group_tag: pd.DataFrame,
    job_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    machine_spec: pd.DataFrame,
    machine_metric: pd.DataFrame,
) -> Dict:
    """Calculate join completeness between tables."""
    group_inst_ids = set(group_tag["inst_id"].dropna().unique())
    job_inst_ids = set(job_df["inst_id"].dropna().unique())
    group_job_intersection = len(group_inst_ids & job_inst_ids)
    group_job_coverage = group_job_intersection / len(group_inst_ids) * 100 if group_inst_ids else 0

    instance_workers = set(instance_df["worker_name"].dropna().unique())
    sensor_workers = set(sensor_df["worker_name"].dropna().unique())
    instance_sensor_intersection = len(instance_workers & sensor_workers)
    instance_sensor_coverage = instance_sensor_intersection / len(instance_workers) * 100 if instance_workers else 0

    instance_machines = set(instance_df["machine"].dropna().unique())
    spec_machines = set(machine_spec["machine"].dropna().unique())
    instance_spec_intersection = len(instance_machines & spec_machines)
    instance_spec_coverage = instance_spec_intersection / len(instance_machines) * 100 if instance_machines else 0

    metric_workers = set(machine_metric["worker_name"].dropna().unique())
    metric_instance_intersection = len(instance_workers & metric_workers)
    metric_instance_coverage = metric_instance_intersection / len(instance_workers) * 100 if instance_workers else 0

    return {
        "group_tag_vs_job": {
            "group_tag_inst_ids": len(group_inst_ids),
            "job_inst_ids": len(job_inst_ids),
            "intersection": group_job_intersection,
            "coverage_percent": group_job_coverage,
        },
        "instance_vs_sensor": {
            "instance_workers": len(instance_workers),
            "sensor_workers": len(sensor_workers),
            "intersection": instance_sensor_intersection,
            "coverage_percent": instance_sensor_coverage,
        },
        "instance_vs_machine_spec": {
            "instance_machines": len(instance_machines),
            "spec_machines": len(spec_machines),
            "intersection": instance_spec_intersection,
            "coverage_percent": instance_spec_coverage,
        },
        "instance_vs_machine_metric": {
            "instance_workers": len(instance_workers),
            "metric_workers": len(metric_workers),
            "intersection": metric_instance_intersection,
            "coverage_percent": metric_instance_coverage,
        },
    }


def format_table(data: Dict, headers: Tuple[str, ...]) -> str:
    """Format data as markdown table."""
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"

    rows = []
    for key, value in data.items():
        rows.append(f"| {key} | {value} |")

    return "\n".join([header_line, separator] + rows)


def generate_coverage_report(output_path: str = "outputs/reports/coverage_report.md"):
    """Generate the coverage report markdown file."""
    logger.info("Loading all tables...")
    tables = load_all_tables()

    job_df = tables["pai_job_table"]
    task_df = tables["pai_task_table"]
    instance_df = tables["pai_instance_table"]
    sensor_df = tables["pai_sensor_table"]
    group_tag = tables["pai_group_tag_table"]
    machine_spec = tables["pai_machine_spec"]
    machine_metric = tables["pai_machine_metric"]

    logger.info("Calculating workload coverage...")
    workload_stats = calculate_workload_coverage(group_tag)

    logger.info("Calculating group distribution...")
    group_stats = calculate_group_distribution(group_tag)

    logger.info("Calculating gpu_type distribution...")
    gpu_stats = calculate_gpu_type_distribution(task_df, machine_spec)

    logger.info("Calculating status distribution...")
    status_stats = calculate_status_distribution(job_df, task_df, instance_df)

    logger.info("Calculating join completeness...")
    join_stats = calculate_join_completeness(
        group_tag, job_df, instance_df, sensor_df, machine_spec, machine_metric
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# xCAD Coverage Report\n\n")
        f.write("> Generated from Alibaba Cluster Trace GPU v2020\n\n")

        f.write("## 1. Table Row Counts\n\n")
        f.write("| Table | Rows |\n")
        f.write("|---|---|\n")
        for name, df in tables.items():
            f.write(f"| {name} | {len(df):,} |\n")
        f.write("\n")

        f.write("## 2. Workload Coverage\n\n")
        f.write(f"**Coverage:** {workload_stats['coverage_percent']:.2f}% ")
        f.write(f"({workload_stats['workload_inst_ids']:,} / {workload_stats['total_inst_ids']:,} inst_ids)\n\n")
        f.write("### Top 20 Workload Values\n\n")
        f.write("| Workload | Count |\n")
        f.write("|---|---|\n")
        for wl, count in workload_stats["top_20_workloads"].items():
            f.write(f"| {wl} | {count:,} |\n")
        f.write("\n")

        f.write("## 3. Group Size Distribution\n\n")
        f.write(f"**Total Groups:** {group_stats['total_groups']:,}\n\n")
        f.write("| Statistic | Value |\n")
        f.write("|---|---|\n")
        for stat, val in group_stats["group_inst_count"].items():
            f.write(f"| {stat} | {val:.2f} |\n")
        f.write(f"\n**Large Group Check:** {'⚠️ Potential dominance' if group_stats['large_group_check'] else '✅ OK'}\n\n")

        f.write("## 4. GPU Type Distribution\n\n")
        f.write("### Task Table\n\n")
        f.write("| GPU Type | Count |\n")
        f.write("|---|---|\n")
        for gpu, count in gpu_stats["task_gpu_type_counts"].items():
            f.write(f"| {gpu} | {count:,} |\n")
        f.write("\n### Machine Spec Table\n\n")
        f.write("| GPU Type | Count |\n")
        f.write("|---|---|\n")
        for gpu, count in gpu_stats["machine_spec_gpu_type_counts"].items():
            f.write(f"| {gpu} | {count:,} |\n")
        f.write("\n")

        f.write("## 5. Status Distribution\n\n")
        for level in ["job", "task", "instance"]:
            f.write(f"### {level.capitalize()} Table\n\n")
            f.write("| Status | Count | Percent |\n")
            f.write("|---|---|---|\n")
            status_dict = status_stats[f"{level}_status"]
            total = sum(status_dict.values())
            for status, count in status_dict.items():
                pct = count / total * 100 if total > 0 else 0
                f.write(f"| {status} | {count:,} | {pct:.2f}% |\n")
            f.write(f"\n**Terminated Rate:** {status_stats['terminated_percent'][level]:.2f}%\n\n")

        f.write("## 6. Join Completeness\n\n")
        f.write("### group_tag.inst_id vs job_table.inst_id\n\n")
        stats = join_stats["group_tag_vs_job"]
        f.write(f"- group_tag inst_ids: {stats['group_tag_inst_ids']:,}\n")
        f.write(f"- job inst_ids: {stats['job_inst_ids']:,}\n")
        f.write(f"- Intersection: {stats['intersection']:,}\n")
        f.write(f"- **Coverage: {stats['coverage_percent']:.2f}%**\n\n")

        f.write("### instance.worker_name vs sensor.worker_name\n\n")
        stats = join_stats["instance_vs_sensor"]
        f.write(f"- instance workers: {stats['instance_workers']:,}\n")
        f.write(f"- sensor workers: {stats['sensor_workers']:,}\n")
        f.write(f"- Intersection: {stats['intersection']:,}\n")
        f.write(f"- **Coverage: {stats['coverage_percent']:.2f}%**\n\n")

        f.write("### instance.machine vs machine_spec.machine\n\n")
        stats = join_stats["instance_vs_machine_spec"]
        f.write(f"- instance machines: {stats['instance_machines']:,}\n")
        f.write(f"- spec machines: {stats['spec_machines']:,}\n")
        f.write(f"- Intersection: {stats['intersection']:,}\n")
        f.write(f"- **Coverage: {stats['coverage_percent']:.2f}%**\n\n")

        f.write("### instance.worker_name vs machine_metric.worker_name\n\n")
        stats = join_stats["instance_vs_machine_metric"]
        f.write(f"- instance workers: {stats['instance_workers']:,}\n")
        f.write(f"- metric workers: {stats['metric_workers']:,}\n")
        f.write(f"- Intersection: {stats['intersection']:,}\n")
        f.write(f"- **Coverage: {stats['coverage_percent']:.2f}%**\n\n")

    logger.info(f"Coverage report written to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_coverage_report()
