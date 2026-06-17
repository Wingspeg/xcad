import os
import sys
import logging
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import OUTPUT_ROOT, NEG_SAMPLE_RATIO, NEG_SAMPLE_STRATEGY, RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{OUTPUT_ROOT}/logs/neg_sample.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

EDGE_DIR = f"{OUTPUT_ROOT}/edges"
LOG_DIR = f"{OUTPUT_ROOT}/logs"
os.makedirs(LOG_DIR, exist_ok=True)

EDGE_TYPES = ["placement", "r1_suits", "r2_requires", "r3_drives", "r4_shifts"]

RELATION_SCHEMA = {
    "placement": ("algorithm", "compute"),
    "r1_suits": ("algorithm", "compute"),
    "r2_requires": ("algorithm", "compute"),
    "r3_drives": ("data", "algorithm"),
    "r4_shifts": ("compute", "compute"),
}


class TypeAwareNegativeSampler:
    """Type-aware negative sampler for heterogeneous graphs.
    
    For each positive edge (src, rel, dst), samples ratio negative edges
    by keeping src and rel fixed, and replacing dst with a random node
    of the same type, ensuring no true positive edges are included.
    """
    
    def __init__(
        self,
        all_positive_edges: pd.DataFrame,
        all_nodes_by_type: Dict[str, List[int]],
        ratio: int = NEG_SAMPLE_RATIO,
        seed: int = RANDOM_SEED
    ):
        """Initialize the negative sampler.
        
        Args:
            all_positive_edges: DataFrame with columns [src, dst, edge_type]
            all_nodes_by_type: Dict mapping node_type -> list of valid node IDs
            ratio: Number of negative samples per positive edge
            seed: Random seed for reproducibility
        """
        self.ratio = ratio
        self.seed = seed
        np.random.seed(seed)
        
        self.all_positive_edges = all_positive_edges
        self.all_nodes_by_type = all_nodes_by_type
        
        self._build_positive_edge_set()
        self._build_target_node_pools()
        
        logger.info(f"Negative sampler initialized:")
        logger.info(f"  Positive edges: {len(all_positive_edges):,}")
        logger.info(f"  Ratio: {ratio}")
        logger.info(f"  Strategy: {NEG_SAMPLE_STRATEGY}")
        logger.info(f"  Node pools: { {k: len(v) for k, v in all_nodes_by_type.items()} }")
    
    def _build_positive_edge_set(self):
        """Build a set of all positive edges for fast lookup."""
        self.positive_edge_set = set()
        for _, row in self.all_positive_edges.iterrows():
            src = row["src"]
            dst = row["dst"]
            rel = row.get("edge_type", "unknown")
            self.positive_edge_set.add((src, dst, rel))
        
        logger.info(f"  Built positive edge set with {len(self.positive_edge_set):,} unique edges")
    
    def _build_target_node_pools(self):
        """Build node pools by relation type for type-aware sampling."""
        self.target_node_pools = {}
        
        for edge_type, (src_type, dst_type) in RELATION_SCHEMA.items():
            self.target_node_pools[edge_type] = self.all_nodes_by_type.get(dst_type, [])
        
        for edge_type, pool in self.target_node_pools.items():
            logger.info(f"  {edge_type} -> target pool size: {len(pool):,}")
    
    def sample_negatives(
        self,
        positive_edges: pd.DataFrame,
        exclude_edges: Optional[Set[Tuple]] = None
    ) -> Tuple[List[Tuple], List[List[Tuple]]]:
        """Sample negative edges for a batch of positive edges.
        
        Args:
            positive_edges: DataFrame with columns [src, dst, edge_type]
            exclude_edges: Additional edge tuples to exclude (from other splits)
        
        Returns:
            pos_edges: List of positive edge tuples (src, dst, edge_type)
            neg_edges: List of lists, each containing ratio negative edges for each pos edge
        """
        exclude_set = self.positive_edge_set.copy()
        if exclude_edges:
            exclude_set |= exclude_edges
        
        pos_edges = []
        neg_edges = []
        
        for _, row in positive_edges.iterrows():
            src = row["src"]
            dst = row["dst"]
            rel = row.get("edge_type", "unknown")
            
            pos_edges.append((src, dst, rel))
            
            neg_batch = self._sample_for_single_edge(src, rel, dst, exclude_set)
            neg_edges.append(neg_batch)
        
        return pos_edges, neg_edges
    
    def _sample_for_single_edge(
        self,
        src: int,
        rel: str,
        true_dst: int,
        exclude_set: Set[Tuple],
        max_attempts: int = 100
    ) -> List[Tuple]:
        """Sample ratio negative edges for a single positive edge."""
        target_pool = self.target_node_pools.get(rel, [])
        
        if len(target_pool) == 0:
            logger.warning(f"  Empty target pool for relation {rel}")
            return []
        
        neg_batch = []
        attempts = 0
        max_neg = min(self.ratio * 10, 1000)
        
        while len(neg_batch) < self.ratio and attempts < max_neg:
            attempts += 1
            
            sampled_dst = np.random.choice(target_pool)
            
            neg_edge = (src, sampled_dst, rel)
            
            if neg_edge in exclude_set:
                continue
            
            if neg_edge in neg_batch:
                continue
            
            neg_batch.append(neg_edge)
        
        if len(neg_batch) < self.ratio:
            logger.warning(
                f"  Only sampled {len(neg_batch)}/{self.ratio} negatives for "
                f"({src}, {rel}, {true_dst}) after {attempts} attempts"
            )
        
        return neg_batch


def load_edges() -> pd.DataFrame:
    """Load all edges and combine into a single DataFrame."""
    logger.info("Loading all edges...")
    
    all_edges = []
    for edge_type in EDGE_TYPES:
        df = pd.read_parquet(f"{EDGE_DIR}/{edge_type}_edges.parquet")
        df["edge_type"] = edge_type
        all_edges.append(df)
        logger.info(f"  {edge_type}: {len(df):,} edges")
    
    combined = pd.concat(all_edges, ignore_index=True)
    logger.info(f"  Total edges: {len(combined):,}")
    
    return combined


def build_node_pools(meta: dict) -> Dict[str, List[int]]:
    """Build node pools for each node type."""
    logger.info("Building node pools...")
    
    pools = {}
    
    for node_type in ["algorithm", "compute", "data"]:
        count = meta["node_counts"].get(node_type, 0)
        pools[node_type] = list(range(count))
        logger.info(f"  {node_type}: {count:,} nodes")
    
    return pools


def test_r1_suits_sampling(
    train_edges: pd.DataFrame,
    node_pools: Dict[str, List[int]],
    meta: dict,
    sample_size: int = 100
):
    """Test negative sampling on r1_suits edges."""
    logger.info("=" * 60)
    logger.info("Testing negative sampling on r1_suits edges")
    logger.info("=" * 60)
    
    r1_edges = train_edges[train_edges["edge_type"] == "r1_suits"].copy()
    
    if len(r1_edges) == 0:
        logger.warning("  No r1_suits edges found in train set")
        return
    
    sample_edges = r1_edges.head(sample_size)
    logger.info(f"  Testing on {len(sample_edges)} r1_suits edges")
    
    sampler = TypeAwareNegativeSampler(
        all_positive_edges=train_edges,
        all_nodes_by_type=node_pools,
        ratio=NEG_SAMPLE_RATIO,
        seed=RANDOM_SEED
    )
    
    pos_edges, neg_edges = sampler.sample_negatives(sample_edges)
    
    logger.info(f"\nSampling results:")
    logger.info(f"  Positive edges: {len(pos_edges):,}")
    logger.info(f"  Negative edges: {sum(len(ne) for ne in neg_edges):,}")
    
    ratios = [len(ne) for ne in neg_edges]
    logger.info(f"  Negatives per positive: min={min(ratios)}, max={max(ratios)}, avg={np.mean(ratios):.2f}")
    
    all_r1_edges = set()
    for _, row in r1_edges.iterrows():
        all_r1_edges.add((row["src"], row["dst"], "r1_suits"))
    
    false_negatives = 0
    target_type_correct = 0
    gpu_type_pool_ids = meta["id_mappings"]["gpu_type_pool"]
    compute_pool = node_pools["compute"]
    gpu_start = meta["num_machines"]
    
    for i, (pos, neg_list) in enumerate(zip(pos_edges, neg_edges)):
        src, dst, rel = pos
        assert rel == "r1_suits"
        
        for neg in neg_list:
            neg_src, neg_dst, neg_rel = neg
            assert neg_rel == "r1_suits"
            
            if (neg_src, neg_dst, neg_rel) in all_r1_edges:
                false_negatives += 1
            
            if gpu_start <= neg_dst < gpu_start + 6:
                target_type_correct += 1
    
    logger.info(f"\nValidation:")
    logger.info(f"  False negatives found: {false_negatives}")
    logger.info(f"  Target type correct: {target_type_correct}/{sum(ratios)}")
    
    if false_negatives == 0:
        logger.info("  ✓ PASS: No false negatives found")
    else:
        logger.warning(f"  ✗ FAIL: {false_negatives} false negatives found")
    
    if target_type_correct == sum(ratios):
        logger.info("  ✓ PASS: All negative targets are in GPU type pool")
    else:
        logger.warning(f"  ✗ FAIL: Only {target_type_correct}/{sum(ratios)} targets are in GPU type pool")
    
    return {
        "false_negatives": false_negatives,
        "target_type_correct": target_type_correct,
        "total_negatives": sum(ratios)
    }


def run_neg_sample_test(
    train_edges: pd.DataFrame,
    node_pools: Dict[str, List[int]],
    meta: dict
):
    """Run the negative sampling test and save results."""
    logger.info("=" * 60)
    logger.info("Running negative sampling test")
    logger.info("=" * 60)
    
    results = test_r1_suits_sampling(train_edges, node_pools, meta, sample_size=100)
    
    test_log_path = f"{LOG_DIR}/neg_sample_test.log"
    with open(test_log_path, "w") as f:
        f.write("# Negative Sampling Test Log\n\n")
        f.write(f"Test timestamp: auto-generated\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Neg sample ratio: {NEG_SAMPLE_RATIO}\n")
        f.write(f"Strategy: {NEG_SAMPLE_STRATEGY}\n\n")
        
        if results:
            f.write("## Results on r1_suits (100 edges)\n\n")
            f.write(f"- False negatives: {results['false_negatives']}\n")
            f.write(f"- Target type correct: {results['target_type_correct']}/{results['total_negatives']}\n")
            f.write(f"- Total negatives: {results['total_negatives']}\n")
    
    logger.info(f"\nTest log saved to: {test_log_path}")
    
    return results


def main():
    logger.info("=" * 60)
    logger.info("Starting negative sampling test")
    logger.info("=" * 60)
    
    logger.info("Note: Full implementation requires loading graph metadata")
    logger.info("This script demonstrates the negative sampling interface")
    
    logger.info("=" * 60)
    logger.info("Negative sampling ready for integration")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()