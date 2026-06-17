"""
xCAD Time Distribution Report Generator

Generates time_distribution.md with:
- Day-of-week and hour distributions (UTC+8)
- Internal relative time span
- Candidate granularity statistics (day/week slicing)
"""

import os
import logging
from datetime import datetime
from typing import Dict
import pandas as pd
import numpy as np
from src.utils.load_tables import load_all_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TIMEZONE_OFFSET = 8 * 3600


def convert_to_utc8(seconds: float, base_timestamp: int = 1577836800) -> pd.Timestamp:
    """Convert desensitized seconds to UTC+8 timestamp."""
    if pd.isna(seconds):
        return pd.NaT
    return pd.Timestamp(base_timestamp + int(seconds) + TIMEZONE_OFFSET, unit="s")


def get_day_of_week(seconds: float) -> int:
    """Get day of week (0=Monday, 6=Sunday) from UTC+8 time."""
    if pd.isna(seconds):
        return -1
    ts = convert_to_utc8(seconds)
    return ts.dayofweek


def get_hour(seconds: float) -> int:
    """Get hour (0-23) from UTC+8 time."""
    if pd.isna(seconds):
        return -1
    ts = convert_to_utc8(seconds)
    return ts.hour


def calculate_time_span(instance_df: pd.DataFrame, metric_df: pd.DataFrame) -> Dict:
    """Calculate trace internal time span."""
    min_start = min(
        instance_df["start_time"].min(),
        metric_df["start_time"].min()
    )
    max_end = max(
        instance_df["end_time"].max(),
        metric_df["end_time"].max()
    )

    span_seconds = max_end - min_start
    span_days = span_seconds / 86400

    return {
        "min_start_time": min_start,
        "max_end_time": max_end,
        "span_seconds": span_seconds,
        "span_days": span_days,
    }


def calculate_dow_distribution(
    instance_df: pd.DataFrame,
    job_df: pd.DataFrame,
    metric_df: pd.DataFrame
) -> Dict:
    """Calculate day-of-week distribution."""
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    instance_dow = instance_df["start_time"].apply(get_day_of_week)
    instance_dow = instance_dow[instance_dow >= 0].value_counts().sort_index()
    instance_dow_dict = {
        dow_labels[i]: int(instance_dow.get(i, 0)) for i in range(7)
    }

    job_dow = job_df["start_time"].apply(get_day_of_week)
    job_dow = job_dow[job_dow >= 0].value_counts().sort_index()
    job_dow_dict = {
        dow_labels[i]: int(job_dow.get(i, 0)) for i in range(7)
    }

    return {
        "instance": instance_dow_dict,
        "job": job_dow_dict,
    }


def calculate_hour_distribution(
    instance_df: pd.DataFrame,
    job_df: pd.DataFrame,
    metric_df: pd.DataFrame
) -> Dict:
    """Calculate hour-of-day distribution."""
    instance_hours = instance_df["start_time"].apply(get_hour).value_counts().sort_index()
    instance_hour_dict = {i: int(instance_hours.get(i, 0)) for i in range(24)}

    job_hours = job_df["start_time"].apply(get_hour).value_counts().sort_index()
    job_hour_dict = {i: int(job_hours.get(i, 0)) for i in range(24)}

    return {
        "instance": instance_hour_dict,
        "job": job_hour_dict,
    }


def calculate_snapshot_statistics(
    instance_df: pd.DataFrame,
    granularity: str = "day"
) -> Dict:
    """Calculate snapshot statistics for given granularity."""
    instance_with_time = instance_df.copy()
    instance_with_time["ts_utc8"] = instance_with_time["start_time"].apply(convert_to_utc8)

    if granularity == "day":
        instance_with_time["window"] = instance_with_time["ts_utc8"].dt.date
    elif granularity == "week":
        instance_with_time["window"] = instance_with_time["ts_utc8"].dt.to_period("W").dt.start_time.dt.date
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    snapshot_counts = instance_with_time.groupby("window")["worker_name"].nunique()

    return {
        "total_snapshots": len(snapshot_counts),
        "avg_instances_per_snapshot": float(snapshot_counts.mean()),
        "p50_instances_per_snapshot": float(snapshot_counts.quantile(0.5)),
        "p90_instances_per_snapshot": float(snapshot_counts.quantile(0.9)),
        "min_instances_per_snapshot": int(snapshot_counts.min()),
        "max_instances_per_snapshot": int(snapshot_counts.max()),
    }


def generate_time_report(output_path: str = "outputs/reports/time_distribution.md"):
    """Generate the time distribution report markdown file."""
    logger.info("Loading all tables...")
    tables = load_all_tables()

    job_df = tables["pai_job_table"]
    instance_df = tables["pai_instance_table"]
    metric_df = tables["pai_machine_metric"]

    logger.info("Calculating time span...")
    time_span = calculate_time_span(instance_df, metric_df)

    logger.info("Calculating day-of-week distribution...")
    dow_stats = calculate_dow_distribution(instance_df, job_df, metric_df)

    logger.info("Calculating hour distribution...")
    hour_stats = calculate_hour_distribution(instance_df, job_df, metric_df)

    logger.info("Calculating daily snapshot statistics...")
    daily_stats = calculate_snapshot_statistics(instance_df, "day")

    logger.info("Calculating weekly snapshot statistics...")
    weekly_stats = calculate_snapshot_statistics(instance_df, "week")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# xCAD Time Distribution Report\n\n")
        f.write("> Generated from Alibaba Cluster Trace GPU v2020\n\n")

        f.write("## 1. Time Signal Reliability\n\n")
        f.write("| Signal | Reliability | Notes |\n")
        f.write("|---|---|---|\n")
        f.write("| Day of Week | ✅ Real | UTC+8 conversion based on desensitized offset |\n")
        f.write("| Hour of Day | ✅ Real | UTC+8 conversion |\n")
        f.write("| Absolute Date | ❌ Fake | Desensitized, do not use for long periods |\n")
        f.write("| Month/Year | ❌ Fake | Desensitized |\n\n")

        f.write("## 2. Trace Internal Time Span\n\n")
        f.write(f"- **Min Start Time (seconds):** {time_span['min_start_time']:,.0f}\n")
        f.write(f"- **Max End Time (seconds):** {time_span['max_end_time']:,.0f}\n")
        f.write(f"- **Total Span:** {time_span['span_seconds']:,.0f} seconds\n")
        f.write(f"- **Total Span:** {time_span['span_days']:.2f} days\n\n")

        f.write("## 3. Day-of-Week Distribution\n\n")
        f.write("### Instance Table\n\n")
        f.write("| Day | Count | Percent |\n")
        f.write("|---|---|---|\n")
        total_instances = sum(dow_stats["instance"].values())
        for day, count in dow_stats["instance"].items():
            pct = count / total_instances * 100 if total_instances > 0 else 0
            f.write(f"| {day} | {count:,} | {pct:.1f}% |\n")
        f.write("\n### Job Table\n\n")
        f.write("| Day | Count | Percent |\n")
        f.write("|---|---|---|\n")
        total_jobs = sum(dow_stats["job"].values())
        for day, count in dow_stats["job"].items():
            pct = count / total_jobs * 100 if total_jobs > 0 else 0
            f.write(f"| {day} | {count:,} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("## 4. Hour-of-Day Distribution\n\n")
        f.write("### Instance Table\n\n")
        f.write("| Hour | Count |\n")
        f.write("|---|---|\n")
        for hour, count in hour_stats["instance"].items():
            f.write(f"| {hour:02d} | {count:,} |\n")
        f.write("\n### Job Table\n\n")
        f.write("| Hour | Count |\n")
        f.write("|---|---|\n")
        for hour, count in hour_stats["job"].items():
            f.write(f"| {hour:02d} | {count:,} |\n")
        f.write("\n")

        f.write("## 5. Snapshot Granularity Candidates\n\n")
        f.write("### Daily Slicing\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Total Snapshots | {daily_stats['total_snapshots']:,} |\n")
        f.write(f"| Avg Instances/Snapshot | {daily_stats['avg_instances_per_snapshot']:,.0f} |\n")
        f.write(f"| P50 Instances/Snapshot | {daily_stats['p50_instances_per_snapshot']:,.0f} |\n")
        f.write(f"| P90 Instances/Snapshot | {daily_stats['p90_instances_per_snapshot']:,.0f} |\n")
        f.write(f"| Min Instances/Snapshot | {daily_stats['min_instances_per_snapshot']:,} |\n")
        f.write(f"| Max Instances/Snapshot | {daily_stats['max_instances_per_snapshot']:,} |\n")
        f.write("\n")

        f.write("### Weekly Slicing\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Total Snapshots | {weekly_stats['total_snapshots']:,} |\n")
        f.write(f"| Avg Instances/Snapshot | {weekly_stats['avg_instances_per_snapshot']:,.0f} |\n")
        f.write(f"| P50 Instances/Snapshot | {weekly_stats['p50_instances_per_snapshot']:,.0f} |\n")
        f.write(f"| P90 Instances/Snapshot | {weekly_stats['p90_instances_per_snapshot']:,.0f} |\n")
        f.write(f"| Min Instances/Snapshot | {weekly_stats['min_instances_per_snapshot']:,} |\n")
        f.write(f"| Max Instances/Snapshot | {weekly_stats['max_instances_per_snapshot']:,} |\n")
        f.write("\n")

        f.write("## 6. Granularity Recommendation\n\n")
        f.write("### For r4_shifts Edge (Compute → Compute)\n\n")
        f.write("Based on the time span and distribution:\n\n")
        f.write("| Granularity | Snapshots | Avg Instances | Recommended Threshold |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Day** | {daily_stats['total_snapshots']} | {daily_stats['avg_instances_per_snapshot']:,.0f} | 0.05 (5% load shift) |\n")
        f.write(f"| **Week** | {weekly_stats['total_snapshots']} | {weekly_stats['avg_instances_per_snapshot']:,.0f} | 0.10 (10% load shift) |\n\n")

        f.write("**Recommendation:** \n")
        f.write("- Use **relative time windows** within the trace (not calendar dates)\n")
        f.write(f"- For daily granularity: {daily_stats['total_snapshots']} snapshots, each ~{time_span['span_days']/daily_stats['total_snapshots']:.1f} days apart\n")
        f.write(f"- For weekly granularity: {weekly_stats['total_snapshots']} snapshots\n")
        f.write("- Threshold for r4_shifts should be configurable\n\n")

    logger.info(f"Time distribution report written to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_time_report()
