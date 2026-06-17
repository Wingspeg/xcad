import os
import sys
import logging
from typing import Dict, Tuple, List, Set
import pandas as pd
import numpy as np
import torch
import dgl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import OUTPUT_ROOT, RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{OUTPUT_ROOT}/logs/build_dgl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

NODE_DIR = f"{OUTPUT_ROOT}/nodes"
EDGE_DIR = f"{OUTPUT_ROOT}/edges"
DGL_DIR = f"{OUTPUT_ROOT}/dgl"
LOG_DIR = f"{OUTPUT_ROOT}/logs"
os.makedirs(DGL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

NUM_TIME_WINDOWS = 69
R4_THRESHOLD = 0.05

NODE_TYPES = ["algorithm", "compute", "data"]
GPU_TYPES = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]

EDGE_TYPES = ["placement", "r1_suits", "r2_requires", "r3_drives", "r4_shifts"]

RELATION_SCHEMA = {
    "placement": ("algorithm", "compute"),
    "r1_suits": ("algorithm", "compute"),
    "r2_requires": ("algorithm", "compute"),
    "r3_drives": ("data", "algorithm"),
    "r4_shifts": ("compute", "compute"),
}


def load_algorithm_and_data_nodes() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """Load algorithm and data nodes, create ID mappings."""
    logger.info("Loading algorithm and data nodes...")
    
    algo_df = pd.read_parquet(f"{NODE_DIR}/algorithm_nodes.parquet")
    algo_ids = {pk: idx for idx, pk in enumerate(algo_df["job_name"].unique())}
    logger.info(f"  algorithm: {len(algo_df)} nodes, {len(algo_ids)} unique IDs")
    
    data_df = pd.read_parquet(f"{NODE_DIR}/data_nodes.parquet")
    data_ids = {pk: idx for idx, pk in enumerate(data_df["group"].unique())}
    logger.info(f"  data: {len(data_df)} nodes, {len(data_ids)} unique IDs")
    
    return algo_df, data_df, algo_ids, data_ids


def create_compute_nodes_with_gpu_type_pool() -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Create compute nodes that include individual machines AND a gpu_type pool.
    
    Compute node ID space structure:
    - 0 ~ (N-1): individual machine nodes (from compute_nodes.parquet)
    - N ~ (N+5): gpu_type pool nodes (CPU, T4, MISC, P100, V100, V100M32)
    
    This allows:
    - placement: Algorithm → machine_node (individual machine)
    - r1_suits: Algorithm → gpu_type_node (via gpu_type)
    - r2_requires: Algorithm → gpu_type_node (via gpu_type_spec)
    - r4_shifts: gpu_type_node → gpu_type_node
    """
    logger.info("Creating compute nodes with gpu_type pool...")
    
    compute_df = pd.read_parquet(f"{NODE_DIR}/compute_nodes.parquet")
    num_machines = len(compute_df)
    
    compute_ids = {pk: idx for idx, pk in enumerate(compute_df["machine"].unique())}
    logger.info(f"  Individual machines: {num_machines}, unique IDs: {len(compute_ids)}")
    
    gpu_type_pool_ids = {}
    for i, gpu_type in enumerate(GPU_TYPES):
        gpu_type_pool_ids[gpu_type] = num_machines + i
    
    logger.info(f"  GPU type pool: {len(GPU_TYPES)} types, IDs: {list(gpu_type_pool_ids.items())}")
    
    compute_df["node_id"] = compute_df["machine"].map(compute_ids)
    
    return compute_df, compute_ids, gpu_type_pool_ids


def load_edges_with_mappings(
    algo_ids: Dict[str, int],
    data_ids: Dict[str, int],
    compute_ids: Dict[str, int],
    gpu_type_pool_ids: Dict[str, int]
) -> Dict[str, pd.DataFrame]:
    """Load edges and map string keys to node IDs."""
    logger.info("Loading and mapping edges...")
    
    edges_by_type = {}
    
    placement_df = pd.read_parquet(f"{EDGE_DIR}/placement_edges.parquet")
    placement_df["src_id"] = placement_df["src"].map(algo_ids)
    placement_df["dst_id"] = placement_df["dst"].map(compute_ids)
    placement_df = placement_df.dropna(subset=["src_id", "dst_id"])
    placement_df["src_id"] = placement_df["src_id"].astype(int)
    placement_df["dst_id"] = placement_df["dst_id"].astype(int)
    edges_by_type["placement"] = placement_df
    logger.info(f"  placement: {len(placement_df):,} edges (Algorithm → Machine)")
    if len(placement_df) == 0:
        logger.warning("  WARNING: placement edges are 0 (src is worker_name, not job_name). "
                     "This edge type will have no edges in the graph.")
    
    r1_df = pd.read_parquet(f"{EDGE_DIR}/r1_suits_edges.parquet")
    r1_df["src_id"] = r1_df["src"].map(algo_ids)
    r1_df["dst_id"] = r1_df["dst"].map(gpu_type_pool_ids)
    r1_df = r1_df.dropna(subset=["src_id", "dst_id"])
    r1_df["src_id"] = r1_df["src_id"].astype(int)
    r1_df["dst_id"] = r1_df["dst_id"].astype(int)
    edges_by_type["r1_suits"] = r1_df
    logger.info(f"  r1_suits: {len(r1_df):,} edges (Algorithm → GPU_Type)")
    
    r2_df = pd.read_parquet(f"{EDGE_DIR}/r2_requires_edges.parquet")
    r2_df["src_id"] = r2_df["src"].map(algo_ids)
    r2_df["dst_id"] = r2_df["dst"].map(gpu_type_pool_ids)
    r2_df = r2_df.dropna(subset=["src_id", "dst_id"])
    r2_df["src_id"] = r2_df["src_id"].astype(int)
    r2_df["dst_id"] = r2_df["dst_id"].astype(int)
    edges_by_type["r2_requires"] = r2_df
    logger.info(f"  r2_requires: {len(r2_df):,} edges (Algorithm → GPU_Type)")
    
    r3_df = pd.read_parquet(f"{EDGE_DIR}/r3_drives_edges.parquet")
    src_mapped_before = len(r3_df)
    r3_df["src_id"] = r3_df["src"].map(data_ids)
    r3_df["dst_id"] = r3_df["dst"].map(algo_ids)
    r3_df = r3_df.dropna(subset=["src_id", "dst_id"])
    r3_df["src_id"] = r3_df["src_id"].astype(int)
    r3_df["dst_id"] = r3_df["dst_id"].astype(int)
    edges_by_type["r3_drives"] = r3_df
    src_dropped = src_mapped_before - len(r3_df)
    logger.info(f"  r3_drives: {len(r3_df):,} edges (Data → Algorithm)")
    if src_dropped > 0:
        logger.info(f"    NOTE: {src_dropped:,} edges dropped (group not in filtered data_nodes, "
                   f"MIN_GROUP_SIZE=5 filter). This is expected behavior.")
    
    r4_df = pd.read_parquet(f"{EDGE_DIR}/r4_shifts_edges.parquet")
    r4_df["src_id"] = r4_df["src"].map(gpu_type_pool_ids)
    r4_df["dst_id"] = r4_df["dst"].map(gpu_type_pool_ids)
    r4_df = r4_df.dropna(subset=["src_id", "dst_id"])
    r4_df["src_id"] = r4_df["src_id"].astype(int)
    r4_df["dst_id"] = r4_df["dst_id"].astype(int)
    edges_by_type["r4_shifts"] = r4_df
    logger.info(f"  r4_shifts: {len(r4_df):,} edges (GPU_Type → GPU_Type)")
    
    return edges_by_type


def preprocess_node_features(
    algo_df: pd.DataFrame,
    compute_df: pd.DataFrame,
    data_df: pd.DataFrame,
    num_machines: int
) -> Tuple[Dict[str, torch.Tensor], Dict, Dict]:
    """Preprocess node features: categorical → IDs, continuous → z-score normalization."""
    logger.info("Preprocessing node features...")
    
    node_feats = {}
    norm_params = {}
    cat_mappings = {}
    
    gpu_types_in_data = compute_df["machine_gpu_type"].dropna().unique()
    all_gpu_types = list(set(GPU_TYPES) | set(gpu_types_in_data))
    gpu_type_to_id = {gt: i for i, gt in enumerate(sorted(all_gpu_types))}
    cat_mappings["compute"] = {"machine_gpu_type": gpu_type_to_id}
    
    compute_feats = []
    compute_feat_names = []
    compute_norm_params = {}
    
    machine_gpu_type_id = compute_df["machine_gpu_type"].map(gpu_type_to_id).fillna(0).astype(int).values
    compute_feats.append(machine_gpu_type_id)
    compute_feat_names.append("machine_gpu_type")
    
    gpu_type_ordinal = compute_df["machine_gpu_type"].map(
        {gt: i for i, gt in enumerate(GPU_TYPES)}
    ).fillna(0).astype(int).values
    compute_feats.append(gpu_type_ordinal)
    compute_feat_names.append("gpu_type_ordinal")
    
    continuous_cols = [
        "machine_cap_cpu", "machine_cap_mem", "machine_cap_gpu",
        "avg_gpu_util", "max_gpu_mem", "avg_host_mem", "max_host_mem",
        "avg_cpu_usage", "avg_net_read", "avg_net_write",
        "avg_machine_load", "avg_num_worker", "avg_machine_cpu", "avg_machine_gpu"
    ]
    
    for col in continuous_cols:
        if col in compute_df.columns:
            valid = compute_df[col].dropna().values
            mean = float(np.mean(valid)) if len(valid) > 0 else 0.0
            std = float(np.std(valid)) if len(valid) > 0 else 1.0
            if std == 0:
                std = 1.0
            feat = (compute_df[col].fillna(mean) - mean) / std
            compute_feats.append(feat.values)
            compute_feat_names.append(col)
            compute_norm_params[col] = (mean, std)
    
    compute_tensor = torch.tensor(np.column_stack(compute_feats), dtype=torch.float32)
    node_feats["compute"] = compute_tensor
    norm_params["compute"] = compute_norm_params
    logger.info(f"  compute: {compute_tensor.shape[1]} features ({compute_feat_names})")
    
    all_workloads = sorted(algo_df["workload"].dropna().unique())
    workload_to_id = {w: i for i, w in enumerate(all_workloads)}
    cat_mappings["algorithm"] = {"workload": workload_to_id}
    
    algo_feats = []
    algo_feat_names = []
    algo_norm_params = {}
    
    workload_id = algo_df["workload"].map(workload_to_id).fillna(-1).astype(int).values
    algo_feats.append(workload_id)
    algo_feat_names.append("workload")
    
    algo_continuous = [
        "avg_plan_cpu", "std_plan_cpu", "min_plan_cpu", "max_plan_cpu",
        "avg_plan_mem", "std_plan_mem", "min_plan_mem", "max_plan_mem",
        "avg_plan_gpu", "std_plan_gpu", "min_plan_gpu", "max_plan_gpu",
        "instance_count"
    ]
    
    for col in algo_continuous:
        if col in algo_df.columns:
            valid = algo_df[col].dropna().values
            mean = float(np.mean(valid)) if len(valid) > 0 else 0.0
            std = float(np.std(valid)) if len(valid) > 0 else 1.0
            if std == 0:
                std = 1.0
            feat = (algo_df[col].fillna(mean) - mean) / std
            algo_feats.append(feat.values)
            algo_feat_names.append(col)
            algo_norm_params[col] = (mean, std)
    
    algo_tensor = torch.tensor(np.column_stack(algo_feats), dtype=torch.float32)
    node_feats["algorithm"] = algo_tensor
    norm_params["algorithm"] = algo_norm_params
    logger.info(f"  algorithm: {algo_tensor.shape[1]} features ({algo_feat_names})")
    
    data_feats = []
    data_feat_names = []
    data_norm_params = {}
    
    data_continuous = [
        "avg_read", "sum_read", "max_read", "avg_write", "sum_write", "max_write",
        "avg_read_count", "sum_read_count", "avg_write_count", "sum_write_count",
        "instance_count"
    ]
    
    for col in data_continuous:
        if col in data_df.columns:
            valid = data_df[col].dropna().values
            mean = float(np.mean(valid)) if len(valid) > 0 else 0.0
            std = float(np.std(valid)) if len(valid) > 0 else 1.0
            if std == 0:
                std = 1.0
            feat = (data_df[col].fillna(mean) - mean) / std
            data_feats.append(feat.values)
            data_feat_names.append(col)
            data_norm_params[col] = (mean, std)
    
    data_tensor = torch.tensor(np.column_stack(data_feats), dtype=torch.float32)
    node_feats["data"] = data_tensor
    norm_params["data"] = data_norm_params
    logger.info(f"  data: {data_tensor.shape[1]} features ({data_feat_names})")
    
    return node_feats, norm_params, cat_mappings


def build_temporal_graphs(
    edges_by_type: Dict[str, pd.DataFrame],
    node_counts: Dict[str, int]
) -> Tuple[List[dgl.DGLHeteroGraph], List[dict]]:
    """Build temporal graph sequence G^(1)...G^(69)."""
    logger.info(f"Building temporal graph sequence ({NUM_TIME_WINDOWS} windows)...")
    
    graph_seq = []
    snapshot_stats = []
    
    for tau in range(1, NUM_TIME_WINDOWS + 1):
        graph_data = {}
        edge_counts = {}
        
        for edge_type in EDGE_TYPES:
            df = edges_by_type[edge_type]
            src_type, dst_type = RELATION_SCHEMA[edge_type]
            
            srcs = df["src_id"].values
            dsts = df["dst_id"].values
            
            graph_data[(src_type, edge_type, dst_type)] = (
                torch.tensor(srcs, dtype=torch.int64),
                torch.tensor(dsts, dtype=torch.int64)
            )
            edge_counts[edge_type] = len(df)
        
        g = dgl.heterograph(graph_data)
        graph_seq.append(g)
        
        stats = {
            "tau": tau,
            "node_counts": {ntype: g.num_nodes(ntype) for ntype in NODE_TYPES},
            "edge_counts": edge_counts
        }
        snapshot_stats.append(stats)
        
        if tau % 10 == 0 or tau <= 3:
            logger.info(f"  τ={tau}: nodes={stats['node_counts']}, edges={edge_counts}")
    
    empty_snapshots = [s["tau"] for s in snapshot_stats if sum(s["edge_counts"].values()) == 0]
    if empty_snapshots:
        logger.warning(f"  Empty snapshots: {empty_snapshots}")
    else:
        logger.info("  No empty snapshots found.")
    
    return graph_seq, snapshot_stats


def main():
    logger.info("=" * 60)
    logger.info("Starting DGL graph construction")
    logger.info("=" * 60)
    
    algo_df, data_df, algo_ids, data_ids = load_algorithm_and_data_nodes()
    
    compute_df, compute_ids, gpu_type_pool_ids = create_compute_nodes_with_gpu_type_pool()
    num_machines = len(compute_df)
    num_compute_nodes = num_machines + len(GPU_TYPES)
    
    node_counts = {
        "algorithm": len(algo_df),
        "compute": num_compute_nodes,
        "data": len(data_df)
    }
    logger.info(f"Node counts: {node_counts}")
    
    edges_by_type = load_edges_with_mappings(algo_ids, data_ids, compute_ids, gpu_type_pool_ids)
    
    node_feats, norm_params, cat_mappings = preprocess_node_features(
        algo_df, compute_df, data_df, num_machines
    )
    
    for ntype, tensor in node_feats.items():
        logger.info(f"  {ntype} features shape: {tensor.shape}")
    
    graph_seq, snapshot_stats = build_temporal_graphs(edges_by_type, node_counts)
    
    graph_path = f"{DGL_DIR}/graph_seq.bin"
    dgl.save_graphs(graph_path, graph_seq)
    logger.info(f"Saved graph sequence to {graph_path}")
    
    id_mappings = {
        "algorithm": algo_ids,
        "compute": compute_ids,
        "gpu_type_pool": gpu_type_pool_ids,
        "data": data_ids
    }
    
    compute_node_schema = {
        "compute_machine": {
            "id_range": [0, num_machines - 1],
            "count": num_machines,
            "primary_key": "machine",
            "description": "Individual machine nodes from machine_spec table"
        },
        "compute_gpu_type": {
            "id_range": [num_machines, num_machines + len(GPU_TYPES) - 1],
            "count": len(GPU_TYPES),
            "primary_key": "gpu_type_name",
            "gpu_types": GPU_TYPES,
            "description": "GPU type pool nodes for r1_suits/r2_requires/r4_shifts edges"
        }
    }
    
    meta = {
        "num_time_windows": NUM_TIME_WINDOWS,
        "node_counts": node_counts,
        "node_types": NODE_TYPES,
        "edge_types": EDGE_TYPES,
        "feature_dims": {ntype: tensor.shape[1] for ntype, tensor in node_feats.items()},
        "norm_params": norm_params,
        "cat_mappings": cat_mappings,
        "id_mappings": id_mappings,
        "snapshot_stats": snapshot_stats,
        "num_machines": num_machines,
        "gpu_types": GPU_TYPES,
        "compute_node_schema": compute_node_schema
    }
    
    meta_path = f"{DGL_DIR}/graph_meta.pt"
    torch.save(meta, meta_path)
    logger.info(f"Saved graph metadata to {meta_path}")
    
    node_feats_path = f"{DGL_DIR}/node_features.pt"
    torch.save(node_feats, node_feats_path)
    logger.info(f"Saved node features to {node_feats_path}")
    
    logger.info("=" * 60)
    logger.info("Graph construction complete")
    logger.info("=" * 60)
    
    return graph_seq, node_feats, meta


if __name__ == "__main__":
    main()