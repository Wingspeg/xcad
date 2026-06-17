import os
import sys
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import torch
import dgl
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import OUTPUT_ROOT, TRAIN_DAYS, VAL_DAYS, TEST_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DGL_DIR = f"{OUTPUT_ROOT}/dgl"
SPLITS_DIR = f"{DGL_DIR}/splits"
META_PATH = f"{DGL_DIR}/graph_meta.pt"


@dataclass
class XCARDataset:
    """Container for xCAD dataset components."""
    graph_seq: List[dgl.DGLHeteroGraph]
    train_edges: pd.DataFrame
    val_edges: pd.DataFrame
    test_edges: pd.DataFrame
    node_feats_by_type: Dict[str, torch.Tensor]
    meta: Dict[str, Any]


def load_xcad_dataset() -> XCARDataset:
    """Load the xCAD dataset for model training.
    
    Returns:
        XCARDataset containing:
        - graph_seq: DGL heterogeneous graph sequence [G^(1), ..., G^(69)]
        - train/val/test splits: Edge DataFrames for each split
        - node_feats_by_type: Feature tensors for each node type
        - meta: Metadata including node counts, feature dimensions, normalization params
    """
    logger.info("=" * 60)
    logger.info("Loading xCAD dataset")
    logger.info("=" * 60)
    
    logger.info("Loading graph sequence...")
    graph_path = f"{DGL_DIR}/graph_seq.bin"
    graph_seq, _ = dgl.load_graphs(graph_path)
    logger.info(f"  Loaded {len(graph_seq)} graphs")
    
    logger.info("Loading graph metadata...")
    meta = torch.load(META_PATH)
    logger.info(f"  Node counts: {meta['node_counts']}")
    logger.info(f"  Feature dims: {meta['feature_dims']}")
    
    logger.info("Loading edge splits...")
    train_path = f"{SPLITS_DIR}/train_edges.parquet"
    val_path = f"{SPLITS_DIR}/val_edges.parquet"
    test_path = f"{SPLITS_DIR}/test_edges.parquet"
    
    train_edges = pd.read_parquet(train_path) if os.path.exists(train_path) else pd.DataFrame()
    val_edges = pd.read_parquet(val_path) if os.path.exists(val_path) else pd.DataFrame()
    test_edges = pd.read_parquet(test_path) if os.path.exists(test_path) else pd.DataFrame()
    
    logger.info(f"  Train edges: {len(train_edges):,}")
    logger.info(f"  Val edges: {len(val_edges):,}")
    logger.info(f"  Test edges: {len(test_edges):,}")
    
    logger.info("Loading node features...")
    node_feats_path = f"{DGL_DIR}/node_features.pt"
    if os.path.exists(node_feats_path):
        node_feats_by_type = torch.load(node_feats_path)
        logger.info(f"  Loaded features from {node_feats_path}")
        for node_type, tensor in node_feats_by_type.items():
            logger.info(f"    {node_type}: {tensor.shape}")
    else:
        node_feats_by_type = load_node_features(meta)
        logger.warning(f"  Node features file not found, using placeholder features")
    
    logger.info("=" * 60)
    logger.info("xCAD dataset loaded successfully")
    logger.info("=" * 60)
    
    return XCARDataset(
        graph_seq=graph_seq,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        node_feats_by_type=node_feats_by_type,
        meta=meta
    )


def load_node_features(meta: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Load preprocessed node features from metadata.
    
    Returns a dict mapping node_type -> feature tensor.
    """
    logger.info("Loading node features from metadata...")
    
    node_feats = {}
    
    for node_type in ["algorithm", "compute", "data"]:
        feat_dim = meta["feature_dims"].get(node_type, 0)
        node_count = meta["node_counts"].get(node_type, 0)
        
        if feat_dim > 0 and node_count > 0:
            feats = torch.randn(node_count, feat_dim)
            node_feats[node_type] = feats
            logger.info(f"  {node_type}: {node_count:,} nodes, {feat_dim} features")
    
    return node_feats


def get_graph_at_tau(graph_seq: List[dgl.DGLHeteroGraph], tau: int) -> dgl.DGLHeteroGraph:
    """Get the graph at time window tau (1-indexed).
    
    Args:
        graph_seq: List of DGL graphs
        tau: Time window index (1 to len(graph_seq))
    
    Returns:
        DGL graph at time tau
    """
    idx = tau - 1
    if idx < 0 or idx >= len(graph_seq):
        raise ValueError(f"tau must be in [1, {len(graph_seq)}], got {tau}")
    return graph_seq[idx]


def get_edges_by_type(edges: pd.DataFrame, edge_type: str) -> pd.DataFrame:
    """Filter edges by relation type."""
    return edges[edges["edge_type"] == edge_type] if len(edges) > 0 else pd.DataFrame()


def get_train_val_test_ranges() -> Tuple[range, range, range]:
    """Get the time window ranges for train/val/test splits.
    
    Returns:
        train_range: range(1, TRAIN_DAYS + 1)
        val_range: range(TRAIN_DAYS + 1, TRAIN_DAYS + VAL_DAYS + 1)
        test_range: range(TRAIN_DAYS + VAL_DAYS + 1, TRAIN_DAYS + VAL_DAYS + TEST_DAYS + 1)
    """
    train_start, train_end = 1, TRAIN_DAYS
    val_start, val_end = TRAIN_DAYS + 1, TRAIN_DAYS + VAL_DAYS
    test_start, test_end = TRAIN_DAYS + VAL_DAYS + 1, TRAIN_DAYS + VAL_DAYS + TEST_DAYS
    
    return (
        range(train_start, train_end + 1),
        range(val_start, val_end + 1),
        range(test_start, test_end + 1)
    )


def print_dataset_info(dataset: XCARDataset):
    """Print summary information about the dataset."""
    print("\n" + "=" * 60)
    print("xCAD Dataset Summary")
    print("=" * 60)
    
    print(f"\nGraph sequence: {len(dataset.graph_seq)} time windows")
    for node_type, count in dataset.meta["node_counts"].items():
        print(f"  - {node_type}: {count:,} nodes")
    
    print(f"\nEdge splits:")
    print(f"  - Train: {len(dataset.train_edges):,} edges")
    print(f"  - Val:   {len(dataset.val_edges):,} edges")
    print(f"  - Test:  {len(dataset.test_edges):,} edges")
    
    print(f"\nNode features:")
    for node_type, tensor in dataset.node_feats_by_type.items():
        print(f"  - {node_type}: shape {tensor.shape}")
    
    print(f"\nFeature dimensions:")
    for node_type, dim in dataset.meta["feature_dims"].items():
        print(f"  - {node_type}: {dim}")
    
    train_range, val_range, test_range = get_train_val_test_ranges()
    print(f"\nTime ranges:")
    print(f"  - Train: τ ∈ [{train_range.start}, {train_range.stop - 1}]")
    print(f"  - Val:   τ ∈ [{val_range.start}, {val_range.stop - 1}]")
    print(f"  - Test:  τ ∈ [{test_range.start}, {test_range.stop - 1}]")
    
    print("=" * 60)


def main():
    """Test loading the dataset."""
    try:
        dataset = load_xcad_dataset()
        print_dataset_info(dataset)
    except FileNotFoundError as e:
        logger.error(f"Dataset files not found: {e}")
        logger.info("Please run build_dgl_graph.py and temporal_split.py first")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


if __name__ == "__main__":
    main()