import os
import sys
import logging
from typing import Dict, List, Tuple, Any, Optional

import torch
import dgl
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "xcad_code"))
from src.utils.config import OUTPUT_ROOT, TRAIN_DAYS, VAL_DAYS, TEST_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_edge_types_subset(edge_type: str) -> bool:
    return edge_type in ["r1_suits", "placement"]


def convert_to_homogeneous(graph: dgl.DGLHeteroGraph, global_node_offset: Dict[str, int]) -> dgl.DGLGraph:
    src_nodes = []
    dst_nodes = []
    edge_types = []
    edge_weights = []

    for etype in graph.etypes:
        if not get_edge_types_subset(etype):
            continue
        srctype, rel, dsttype = graph.canonical_etypes[etype]
        edges = graph.edges(etype=etype)
        src_global = [global_node_offset.get(srctype, 0) + int(n.item()) for n in edges[0]]
        dst_global = [global_node_offset.get(dsttype, 0) + int(n.item()) for n in edges[1]]
        src_nodes.extend(src_global)
        dst_nodes.extend(dst_global)
        edge_types.extend([etype] * len(src_global))
        if 'weight' in graph.etypes[etype].data:
            edge_weights.extend(graph.etypes[etype].data['weight'].tolist())
        else:
            edge_weights.extend([1.0] * len(src_global))

    if len(src_nodes) == 0:
        return None

    src_nodes = torch.tensor(src_nodes, dtype=torch.long)
    dst_nodes = torch.tensor(dst_nodes, dtype=torch.long)
    edge_types = torch.tensor(
        [0 if et == "r1_suits" else 1 for et in edge_types],
        dtype=torch.long
    )

    homo_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=sum(global_node_offset.values()))
    homo_graph.edata['type'] = edge_types

    return homo_graph


def build_sub_graph(num_nodes: int, num_rels: int, triples: List[Tuple],
                    use_cuda: bool = False, gpu: int = 0) -> dgl.DGLGraph:
    src, rel, dst = zip(*triples)
    src = torch.LongTensor(src)
    rel = torch.LongTensor(rel)
    dst = torch.LongTensor(dst)

    edge_types = rel

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.edata['type'] = edge_types
    g.ndata['id'] = torch.arange(num_nodes)

    if use_cuda:
        g = g.to(gpu)
        g.ndata['id'] = g.ndata['id'].to(gpu)
        g.edata['type'] = g.edata['type'].to(gpu)

    return g


def load_xcad_for_regcn(edge_types: List[str] = None, max_time_windows: int = 10, max_edges_per_window: int = 2000) -> Dict[str, Any]:
    from src.utils.data_loader import load_xcad_dataset

    if edge_types is None:
        edge_types = ["r1_suits", "placement"]

    logger.info("Loading xCAD dataset...")
    dataset = load_xcad_dataset()

    node_counts = dataset.meta["node_counts"]
    feature_dims = dataset.meta["feature_dims"]

    total_nodes = sum(node_counts.values())
    logger.info(f"Total nodes (homogeneous): {total_nodes}")

    global_offset = {}
    offset = 0
    for node_type in ["algorithm", "compute", "data"]:
        global_offset[node_type] = offset
        offset += node_counts[node_type]

    relation_to_id = {rel: i for i, rel in enumerate(edge_types)}
    relation_to_id.update({f"{rel}_inv": i + len(edge_types) for i, rel in enumerate(edge_types)})

    logger.info(f"Relation mapping: {relation_to_id}")

    train_range = range(1, 8)
    val_range = range(8, 9)
    test_range = range(9, 11)

    logger.info(f"Smoke test - Train range: [{train_range.start}, {train_range.stop - 1}]")
    logger.info(f"Smoke test - Val range: [{val_range.start}, {val_range.stop - 1}]")
    logger.info(f"Smoke test - Test range: [{test_range.start}, {test_range.stop - 1}]")

    train_triples_by_tau = {}
    val_triples_by_tau = {}
    test_triples_by_tau = {}

    for tau in range(1, max_time_windows + 1):
        graph = dataset.graph_seq[tau - 1]
        triples = []

        for i, etype in enumerate(graph.etypes):
            if etype not in edge_types:
                continue
            edges = graph.edges(etype=etype)
            srctype, rel, dsttype = graph.canonical_etypes[i]
            for src, dst in zip(edges[0], edges[1]):
                src_global = global_offset.get(srctype, 0) + int(src.item())
                dst_global = global_offset.get(dsttype, 0) + int(dst.item())
                rel_id = relation_to_id.get(etype, 0)
                triples.append((src_global, rel_id, dst_global))

        if max_edges_per_window > 0 and len(triples) > max_edges_per_window:
            import random
            random.seed(42)
            triples = random.sample(triples, max_edges_per_window)
            logger.info(f"  Sampled {max_edges_per_window} edges from τ={tau}")

        if len(triples) > 0:
            if tau in train_range:
                train_triples_by_tau[tau] = triples
            elif tau in val_range:
                val_triples_by_tau[tau] = triples
            elif tau in test_range:
                test_triples_by_tau[tau] = triples

    logger.info(f"Train time windows: {len(train_triples_by_tau)}")
    logger.info(f"Val time windows: {len(val_triples_by_tau)}")
    logger.info(f"Test time windows: {len(test_triples_by_tau)}")

    train_glist = []
    for tau in sorted(train_triples_by_tau.keys()):
        g = build_sub_graph(total_nodes, len(relation_to_id), train_triples_by_tau[tau])
        train_glist.append((tau, g))

    val_glist = []
    for tau in sorted(val_triples_by_tau.keys()):
        g = build_sub_graph(total_nodes, len(relation_to_id), val_triples_by_tau[tau])
        val_glist.append((tau, g))

    test_glist = []
    for tau in sorted(test_triples_by_tau.keys()):
        g = build_sub_graph(total_nodes, len(relation_to_id), test_triples_by_tau[tau])
        test_glist.append((tau, g))

    all_train_triples = []
    for triples in train_triples_by_tau.values():
        all_train_triples.extend(triples)

    all_val_triples = []
    for triples in val_triples_by_tau.values():
        all_val_triples.extend(triples)

    all_test_triples = []
    for triples in test_triples_by_tau.values():
        all_test_triples.extend(triples)

    return {
        "train_glist": train_glist,
        "val_glist": val_glist,
        "test_glist": test_glist,
        "all_train_triples": torch.LongTensor(all_train_triples) if all_train_triples else torch.LongTensor([]),
        "all_val_triples": torch.LongTensor(all_val_triples) if all_val_triples else torch.LongTensor([]),
        "all_test_triples": torch.LongTensor(all_test_triples) if all_test_triples else torch.LongTensor([]),
        "num_nodes": total_nodes,
        "num_rels": len(relation_to_id),
        "node_counts": node_counts,
        "relation_to_id": relation_to_id,
        "feature_dims": feature_dims,
    }


class REGCNDataAdapter:
    def __init__(self, edge_types: List[str] = None, max_time_windows: int = 10):
        self.edge_types = edge_types or ["r1_suits", "placement"]
        self.max_time_windows = max_time_windows
        self.data = None

    def load(self):
        self.data = load_xcad_for_regcn(self.edge_types, self.max_time_windows)
        return self.data

    def get_train_batch(self, batch_size: int = 512, neg_ratio: int = 5) -> Tuple[List, torch.Tensor]:
        if self.data is None:
            self.load()

        all_triples = self.data["all_train_triples"]
        if len(all_triples) == 0:
            return [], torch.LongTensor([])

        indices = torch.randperm(len(all_triples))[:batch_size]
        batch_triples = all_triples[indices]

        neg_samples = self._negative_sampling(batch_triples, neg_ratio)

        all_samples = torch.cat([batch_triples, neg_samples], dim=0)
        labels = torch.cat([
            torch.ones(len(batch_triples), dtype=torch.long),
            torch.zeros(len(neg_samples), dtype=torch.long)
        ])

        return all_samples, labels

    def _negative_sampling(self, triples: torch.Tensor, neg_ratio: int) -> torch.Tensor:
        num_nodes = self.data["num_nodes"]
        num_rels = self.data["num_rels"]

        pos_samples = triples.repeat(neg_ratio, 1)
        num_pos = len(triples)

        neg_head = pos_samples.clone()
        neg_tail = pos_samples.clone()

        rand_heads = torch.randint(0, num_nodes, (num_pos * neg_ratio,))
        rand_tails = torch.randint(0, num_nodes, (num_pos * neg_ratio,))

        neg_head[:, 0] = rand_heads
        neg_tail[:, 2] = rand_tails

        neg_samples = torch.cat([neg_head, neg_tail], dim=0)

        return neg_samples

    def get_val_test_batches(self, triples: torch.Tensor, batch_size: int = 512) -> List[torch.Tensor]:
        batches = []
        for i in range(0, len(triples), batch_size):
            batches.append(triples[i:i + batch_size])
        return batches