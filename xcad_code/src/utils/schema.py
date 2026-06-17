"""
xCAD Schema Definition

Based on official Alibaba Cluster Trace GPU v2020 README and corrected field_mapping.json.
This module defines all table schemas, join keys, and status values as Python constants.

Reference: datasets/alibaba-cluster-trace-gpu-v2020/processed/_meta/field_mapping.json
"""

from typing import Dict, List
from src.utils.config import DATA_ROOT

RAW_DATA_DIR = f"{DATA_ROOT}/raw"

# =============================================================================
# Table Column Definitions (by position, 1-indexed)
# =============================================================================

PAI_JOB_TABLE_COLS = [
    "job_name",    # 1: 主键，每个 job 唯一
    "inst_id",     # 2: 当作 job_id 用，用于 join group_tag 和 sensor
    "user",        # 3: 用户
    "status",      # 4: Running/Terminated/Failed/Waiting
    "start_time",  # 5: 秒，脱敏减常数
    "end_time",    # 6: 秒
]

PAI_TASK_TABLE_COLS = [
    "job_name",    # 1: 用于 join job_table
    "task_name",   # 2: 角色，ps/worker/evaluator
    "inst_num",    # 3: 实例数
    "status",      # 4: 任务状态
    "start_time",  # 5: 开始时间
    "end_time",    # 6: 结束时间
    "plan_cpu",    # 7: 百分比（600=6核）
    "plan_mem",    # 8: GB
    "plan_gpu",    # 9: 百分比（50=50% GPU）
    "gpu_type",    # 10: MISC/V100/T4/P100... MISC=老代次（K40m/K80/M60）
]

PAI_INSTANCE_TABLE_COLS = [
    "job_name",    # 1: 用于 join job_table
    "task_name",   # 2: 用于 join task_table
    "inst_name",   # 3: instance 的具体实例名
    "worker_name", # 4: 实例级细粒度键，用于 join sensor 和 machine_metric
    "inst_id",     # 5: 用于 join group_tag
    "status",      # 6: 实例状态
    "start_time",  # 7: 开始时间
    "end_time",    # 8: 结束时间
    "machine",     # 9: 实例所在机器，用于 join machine_spec 和 machine_metric
]

PAI_SENSOR_TABLE_COLS = [
    "job_name",         # 1: join 用
    "task_name",        # 2: join 用
    "worker_name",      # 3: 实例级细粒度键
    "inst_id",          # 4: join 用
    "machine",          # 5: join 用
    "gpu_name",         # 6: 设备号(/dev/nvidia0)，不是 gpu_type
    "cpu_usage",        # 7: CPU 使用百分比
    "gpu_wrk_util",     # 8: 实际 GPU 使用百分比
    "avg_mem",          # 9: 内存 GB（均值）
    "max_mem",         # 10: 内存 GB（最大）
    "avg_gpu_wrk_mem",  # 11: GPU 显存 GB（均值）
    "max_gpu_wrk_mem",  # 12: GPU 显存 GB（最大）
    "read",             # 13: 网络输入字节（数据 IO 画像）
    "write",            # 14: 网络输出字节（数据 IO 画像）
    "read_count",       # 15: 网络读次数（数据 IO 画像）
    "write_count",      # 16: 网络写次数（数据 IO 画像）
]

PAI_GROUP_TAG_TABLE_COLS = [
    "inst_id",      # 1: 用于 join job_table
    "user",         # 2: 用户
    "gpu_type_spec", # 3: 实例指定的 GPU 类型要求（空=未指定）→ r2 requires 边依据
    "group",        # 4: 数据节点主键，标记定制化输入（入口脚本、命令行参数、数据源/数据汇）相似的实例
    "workload",     # 5: DL 任务类型（graphlearn/ctr/bert...），约 9% 实例有 → 算法节点语义来源
]

PAI_MACHINE_SPEC_TABLE_COLS = [
    "machine",   # 1: 机器标识
    "gpu_type",  # 2: GPU 代次类型
    "cap_cpu",   # 3: CPU 核数
    "cap_mem",   # 4: 内存 GB
    "cap_gpu",   # 5: GPU 卡数（cap_gpu=0 表示纯 CPU 机器）
]

PAI_MACHINE_METRIC_TABLE_COLS = [
    "worker_name",       # 1: 实例级细粒度键，用于 join instance
    "machine",           # 2: 机器标识，用于 join machine_spec
    "start_time",        # 3: 开始时间
    "end_time",          # 4: 结束时间
    "machine_cpu_iowait", # 5: CPU iowait
    "machine_cpu_kernel", # 6: CPU kernel 时间
    "machine_cpu_usr",   # 7: CPU user 时间
    "machine_gpu",       # 8: GPU 使用
    "machine_load_1",    # 9: 1分钟负载
    "machine_net_receive", # 10: 网络接收
    "machine_num_worker", # 11: 共置实例数 = 资源竞争信号
    "machine_cpu",       # 12: CPU 使用率
]

# =============================================================================
# Join Key Mapping
# =============================================================================

JOIN_KEYS = {
    "job_task_instance": {
        "key": "job_name (+ task_name for instance)",
        "usage": "job ↔ task ↔ instance 的主关联键"
    },
    "to_sensor_and_machine_metric": {
        "key": "worker_name",
        "usage": "→ sensor / machine_metric 的细粒度实例级关联"
    },
    "to_machine_spec_and_machine_metric": {
        "key": "machine",
        "usage": "→ machine_spec / machine_metric 的机器级关联"
    },
    "to_group_tag": {
        "key": "inst_id",
        "usage": "→ group_tag (workload/group) 的作业级关联"
    }
}

TABLE_FILES = {
    "pai_job_table": f"{RAW_DATA_DIR}/pai_job_table.csv",
    "pai_task_table": f"{RAW_DATA_DIR}/pai_task_table.csv",
    "pai_instance_table": f"{RAW_DATA_DIR}/pai_instance_table.csv",
    "pai_sensor_table": f"{RAW_DATA_DIR}/pai_sensor_table.csv",
    "pai_group_tag_table": f"{RAW_DATA_DIR}/pai_group_tag_table.csv",
    "pai_machine_spec": f"{RAW_DATA_DIR}/pai_machine_spec.csv",
    "pai_machine_metric": f"{RAW_DATA_DIR}/pai_machine_metric.csv",
}

# =============================================================================
# Status Values
# =============================================================================

STATUS_VALUES = {
    "job": ["Running", "Terminated", "Failed", "Waiting"],
    "task": ["Running", "Terminated", "Failed", "Waiting"],
    "instance": ["Running", "Terminated", "Failed", "Waiting"],
}

STATUS_SUCCESS = "Terminated"

# =============================================================================
# GPU Type Mapping (代次序数化)
# =============================================================================

GPU_TYPE_ORDER: Dict[str, int] = {
    "K40m": 1,
    "K80": 2,
    "M60": 3,
    "P100": 4,
    "MISC": 5,
    "T4": 6,
    "V100": 7,
}

GPU_TYPE_NAMES = {
    "K40m": "NVIDIA Kepler K40m",
    "K80": "NVIDIA Kepler K80",
    "M60": "NVIDIA Maxwell M60",
    "P100": "NVIDIA Pascal P100",
    "MISC": "Legacy GPU (K40m/K80/M60)",
    "T4": "NVIDIA Turing T4",
    "V100": "NVIDIA Volta V100",
}

# =============================================================================
# Node Definitions
# =============================================================================

NODE_DEFINITIONS = {
    "Compute": {
        "primary_key": "machine",
        "sources": ["pai_machine_spec", "pai_machine_metric", "pai_sensor_table"],
    },
    "Algorithm": {
        "primary_key": "workload 或 workload×角色",
        "sources": ["pai_group_tag_table", "pai_task_table"],
    },
    "Data": {
        "primary_key": "group",
        "sources": ["pai_group_tag_table", "pai_sensor_table"],
    }
}

# =============================================================================
# Edge Definitions
# =============================================================================

EDGE_DEFINITIONS = {
    "placement": {"type": "锚边", "direction": "instance → machine"},
    "r1_suits": {"type": "Algorithm → Compute"},
    "r2_requires": {"type": "Algorithm → Compute"},
    "r3_drives": {"type": "Data → Algorithm"},
    "r4_shifts": {"type": "Compute → Compute (跨时间窗)"},
}

# =============================================================================
# Memory Optimization Dtypes
# =============================================================================

DTYPES_SENSOR = {
    "job_name": "str",
    "task_name": "str",
    "worker_name": "str",
    "inst_id": "str",
    "machine": "str",
    "gpu_name": "str",
    "cpu_usage": "float32",
    "gpu_wrk_util": "float32",
    "avg_mem": "float32",
    "max_mem": "float32",
    "avg_gpu_wrk_mem": "float32",
    "max_gpu_wrk_mem": "float32",
    "read": "float64",
    "write": "float64",
    "read_count": "float64",
    "write_count": "float64",
}

DTYPES_INSTANCE = {
    "job_name": "str",
    "task_name": "str",
    "inst_name": "str",
    "worker_name": "str",
    "inst_id": "str",
    "status": "str",
    "start_time": "float64",
    "end_time": "float64",
    "machine": "str",
}

DTYPES_MACHINE_METRIC = {
    "worker_name": "str",
    "machine": "str",
    "start_time": "float64",
    "end_time": "float64",
    "machine_cpu_iowait": "float32",
    "machine_cpu_kernel": "float32",
    "machine_cpu_usr": "float32",
    "machine_gpu": "float32",
    "machine_load_1": "float32",
    "machine_net_receive": "float32",
    "machine_num_worker": "float32",
    "machine_cpu": "float32",
}

DTYPES_TASK = {
    "job_name": "str",
    "task_name": "str",
    "inst_num": "float32",
    "status": "str",
    "start_time": "float64",
    "end_time": "float64",
    "plan_cpu": "float32",
    "plan_mem": "float32",
    "plan_gpu": "float32",
    "gpu_type": "str",
}

DTYPES_JOB = {
    "job_name": "str",
    "inst_id": "str",
    "user": "str",
    "status": "str",
    "start_time": "float64",
    "end_time": "float64",
}

DTYPES_GROUP_TAG = {
    "inst_id": "str",
    "user": "str",
    "gpu_type_spec": "str",
    "group": "str",
    "workload": "str",
}

DTYPES_MACHINE_SPEC = {
    "machine": "str",
    "gpu_type": "str",
    "cap_cpu": "float32",
    "cap_mem": "float32",
    "cap_gpu": "float32",
}

TABLE_DTYPES = {
    "pai_job_table": DTYPES_JOB,
    "pai_task_table": DTYPES_TASK,
    "pai_instance_table": DTYPES_INSTANCE,
    "pai_sensor_table": DTYPES_SENSOR,
    "pai_group_tag_table": DTYPES_GROUP_TAG,
    "pai_machine_spec": DTYPES_MACHINE_SPEC,
    "pai_machine_metric": DTYPES_MACHINE_METRIC,
}

TABLE_COLS = {
    "pai_job_table": PAI_JOB_TABLE_COLS,
    "pai_task_table": PAI_TASK_TABLE_COLS,
    "pai_instance_table": PAI_INSTANCE_TABLE_COLS,
    "pai_sensor_table": PAI_SENSOR_TABLE_COLS,
    "pai_group_tag_table": PAI_GROUP_TAG_TABLE_COLS,
    "pai_machine_spec": PAI_MACHINE_SPEC_TABLE_COLS,
    "pai_machine_metric": PAI_MACHINE_METRIC_TABLE_COLS,
}
