import os
import sys
import time
import logging
import tracemalloc
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "..", "xcad_code"))

from src.utils.config import OUTPUT_ROOT
from src.models.regcn import REGCNModel
from baselines.regcn_training.regcn_adapter import REGCNDataAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(OUTPUT_ROOT) / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 设备检测 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if device.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    logger.warning("CUDA not available — falling back to CPU. To use GPU:")
    logger.warning("  1) On server: check `nvidia-smi`, install matching CUDA + PyTorch")
    logger.warning("  2) Verify: python -c 'import torch; print(torch.cuda.is_available())'")


def get_total_rank(scores, answers):
    scores = scores.cpu()
    answers = answers.cpu()

    ranks = []
    for i, ans in enumerate(answers):
        score = scores[i]
        ans_set = set(ans.tolist()) if isinstance(ans, torch.Tensor) else set([ans])
        scores_i = score.tolist()
        true_score = scores_i[ans] if isinstance(ans, int) else scores_i[ans[0]]

        rank = 1
        for j, s in enumerate(scores_i):
            if j not in ans_set and s > true_score:
                rank += 1
        ranks.append(rank)

    ranks = torch.tensor(ranks)
    mrr = torch.mean(1.0 / ranks.float())

    hits_at_1 = torch.mean((ranks <= 1).float())
    hits_at_3 = torch.mean((ranks <= 3).float())
    hits_at_10 = torch.mean((ranks <= 10).float())

    return mrr.item(), hits_at_1.item(), hits_at_3.item(), hits_at_10.item(), ranks


def evaluate_model(model, glist, triples, num_nodes, num_rels, max_eval_samples=500):
    model.eval()
    with torch.no_grad():
        if isinstance(glist[0], tuple):
            g_list = [g for _, g in glist]
        else:
            g_list = glist
        history_embs, r_emb = model.forward(g_list)
        final_emb = history_embs[-1]
        final_emb = F.normalize(final_emb, p=2, dim=1)

        inverse_triples = triples[:, [2, 1, 0]].clone()
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
        all_triples = torch.cat([triples, inverse_triples])

        scores = model.predict(final_emb, r_emb, all_triples)

        all_answers = {}
        for i in range(len(all_triples)):
            s, r, o = all_triples[i].tolist()
            if s not in all_answers:
                all_answers[s] = set()
            all_answers[s].add(o)

        mrr_list, hits1_list, hits3_list, hits10_list = [], [], [], []
        batch_size = 100

        eval_samples = min(max_eval_samples, len(all_triples))
        indices = torch.randperm(len(all_triples), device=device)[:eval_samples]

        for idx in indices:
            i = idx.item()
            s = all_triples[i, 0].item()
            score = scores[i]
            ans_set = all_answers.get(s, set())

            true_score = score[list(ans_set)[0]] if ans_set else score[all_triples[i, 2].item()]

            rank = 1
            for j in range(len(score)):
                if j not in ans_set and score[j] > true_score:
                    rank += 1

            mrr_list.append(1.0 / rank)
            hits1_list.append(1.0 if rank <= 1 else 0.0)
            hits3_list.append(1.0 if rank <= 3 else 0.0)
            hits10_list.append(1.0 if rank <= 10 else 0.0)

        mrr = np.mean(mrr_list) if mrr_list else 0.0
        hits1 = np.mean(hits1_list) if hits1_list else 0.0
        hits3 = np.mean(hits3_list) if hits3_list else 0.0
        hits10 = np.mean(hits10_list) if hits10_list else 0.0

    return mrr, hits1, hits3, hits10


def main():
    logger.info("=" * 60)
    logger.info("RE-GCN Smoke Test on xCAD Dataset")
    logger.info("=" * 60)

    tracemalloc.start()
    start_time = time.time()

    EDGE_TYPES = ["r1_suits", "placement"]
    MAX_TIME_WINDOWS = 10
    TRAIN_WINDOWS = [1, 7]
    VAL_WINDOWS = [8, 8]
    TEST_WINDOWS = [9, 10]

    HIDDEN_DIM = 64
    N_LAYERS = 1
    N_BASES = -1
    DROPOUT = 0.1
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 5
    BATCH_SIZE = 128
    NEG_RATIO = 5
    EARLY_STOP_PATIENCE = 2

    logger.info(f"Configuration:")
    logger.info(f"  Edge types: {EDGE_TYPES}")
    logger.info(f"  Max time windows: {MAX_TIME_WINDOWS}")
    logger.info(f"  Train windows: {TRAIN_WINDOWS}")
    logger.info(f"  Val windows: {VAL_WINDOWS}")
    logger.info(f"  Test windows: {TEST_WINDOWS}")
    logger.info(f"  Hidden dim: {HIDDEN_DIM}")
    logger.info(f"  Max epochs: {MAX_EPOCHS}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Neg ratio: {NEG_RATIO}")

    logger.info("\nLoading data...")
    adapter = REGCNDataAdapter(edge_types=EDGE_TYPES, max_time_windows=MAX_TIME_WINDOWS)
    data = adapter.load()

    num_nodes = data["num_nodes"]
    num_rels = data["num_rels"]
    logger.info(f"  Num nodes: {num_nodes}")
    logger.info(f"  Num relations: {num_rels}")
    logger.info(f"  Train triples: {len(data['all_train_triples'])}")
    logger.info(f"  Val triples: {len(data['all_val_triples'])}")
    logger.info(f"  Test triples: {len(data['all_test_triples'])}")

    model = REGCNModel(
        num_nodes=num_nodes,
        num_rels=num_rels,
        h_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        num_bases=N_BASES,
        dropout=DROPOUT
    )
    # === 模型搬 device(参数 / buffer / 子模块递归生效)===
    model = model.to(device)
    logger.info(f"Model moved to: {device}")

    # === 节点特征搬 device ===
    # 实际节点特征是 REGCNModel.dynamic_emb(Parameter),已随 model.to(device) 上 GPU。
    # 此处为防御性显式检查,若未来引入外部 node_features,也可在此统一搬 device。
    if hasattr(model, "dynamic_emb"):
        logger.info(
            f"dynamic_emb on: {model.dynamic_emb.device}, "
            f"shape={tuple(model.dynamic_emb.shape)}"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # === DGL 图对象与三元组统一搬 device ===
    # DGL 的 g.to(device) 会同时移动 ndata / edata
    train_glist = [(tau, g.to(device)) for tau, g in data["train_glist"]]
    val_glist = [(tau, g.to(device)) for tau, g in data["val_glist"]]
    test_glist = [(tau, g.to(device)) for tau, g in data["test_glist"]]

    all_train_triples = data["all_train_triples"].to(device)
    all_val_triples = data["all_val_triples"].to(device)
    all_test_triples = data["all_test_triples"].to(device)
    logger.info(
        f"Data moved to: train/val/test triples on {device}, "
        f"{len(train_glist)} train graphs / {len(val_glist)} val / {len(test_glist)} test"
    )

    logger.info("\nStarting training...")
    epoch_losses = []
    epoch_times = []
    best_val_mrr = 0
    patience_counter = 0
    training_complete = False

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        model.train()

        total_loss = 0
        num_batches = 0

        indices = torch.randperm(len(all_train_triples), device=device)
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            batch_triples = all_train_triples[batch_idx]

            neg_head = batch_triples.repeat(NEG_RATIO, 1)
            neg_tail = batch_triples.clone().repeat(NEG_RATIO, 1)

            # 负采样直接在 device 上生成,避免 CPU→GPU 来回搬运
            rand_heads = torch.randint(0, num_nodes, (len(batch_triples) * NEG_RATIO,), device=device)
            rand_tails = torch.randint(0, num_nodes, (len(batch_triples) * NEG_RATIO,), device=device)

            neg_head[:, 0] = rand_heads
            neg_tail[:, 2] = rand_tails

            neg_samples = torch.cat([neg_head, neg_tail], dim=0)

            pos_labels = torch.ones(len(batch_triples), dtype=torch.long, device=device)
            neg_labels = torch.zeros(len(neg_samples), dtype=torch.long, device=device)
            all_labels = torch.cat([pos_labels, neg_labels])

            all_samples = torch.cat([batch_triples, neg_samples], dim=0)

            loss, _, _ = model.get_loss(train_glist, all_samples, all_samples)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start

        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)

        logger.info(f"Epoch {epoch + 1}/{MAX_EPOCHS} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

        # === 显存监控 ===
        if device.type == "cuda":
            alloc_mb = torch.cuda.memory_allocated() / 1024**2
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            reserved_mb = torch.cuda.memory_reserved() / 1024**2
            logger.info(
                f"  GPU memory: allocated={alloc_mb:.2f} MB, "
                f"peak={peak_mb:.2f} MB, reserved={reserved_mb:.2f} MB"
            )

        if len(all_val_triples) > 0:
            val_mrr, val_h1, val_h3, val_h10 = evaluate_model(
                model, val_glist, all_val_triples, num_nodes, num_rels
            )
            logger.info(f"  Val - MRR: {val_mrr:.4f}, Hits@1: {val_h1:.4f}, Hits@3: {val_h3:.4f}, Hits@10: {val_h10:.4f}")

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    training_complete = True
                    break
        else:
            training_complete = True

    if not training_complete:
        training_complete = True

    logger.info("\nEvaluating on test set...")
    if len(all_test_triples) > 0:
        test_mrr, test_h1, test_h3, test_h10 = evaluate_model(
            model, test_glist, all_test_triples, num_nodes, num_rels
        )
        logger.info(f"  Test - MRR: {test_mrr:.4f}, Hits@1: {test_h1:.4f}, Hits@3: {test_h3:.4f}, Hits@10: {test_h10:.4f}")
    else:
        test_mrr, test_h1, test_h3, test_h10 = 0, 0, 0, 0
        logger.warning("No test triples found!")

    total_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    val_mrr_final, val_h1_final, val_h3_final, val_h10_final = 0, 0, 0, 0
    if len(all_val_triples) > 0:
        val_mrr_final, val_h1_final, val_h3_final, val_h10_final = evaluate_model(
            model, val_glist, all_val_triples, num_nodes, num_rels
        )

    final_loss = f"{epoch_losses[-1]:.4f}" if epoch_losses else "N/A"

    report = f"""# RE-GCN Stage 3.1 Smoke Test Report

## Configuration
- Edge types: {EDGE_TYPES}
- Max time windows: {MAX_TIME_WINDOWS}
- Train windows: {TRAIN_WINDOWS}
- Val windows: {VAL_WINDOWS}
- Test windows: {TEST_WINDOWS}
- Hidden dimension: {HIDDEN_DIM}
- Number of layers: {N_LAYERS}
- Dropout: {DROPOUT}
- Learning rate: {LEARNING_RATE}
- Batch size: {BATCH_SIZE}
- Negative sampling ratio: {NEG_RATIO}
- Max epochs: {MAX_EPOCHS}

## Training Summary
- Training completed: {training_complete}
- Total epochs run: {len(epoch_losses)}
- Total training time: {total_time:.2f}s
- Final training loss: {final_loss}

## Training Loss Curve
| Epoch | Loss | Time (s) |
|-------|------|----------|
"""
    for i, (loss, t) in enumerate(zip(epoch_losses, epoch_times)):
        report += f"| {i+1} | {loss:.4f} | {t:.2f} |\n"

    report += f"""
## Validation Metrics
- MRR: {val_mrr_final:.4f}
- Hits@1: {val_h1_final:.4f}
- Hits@3: {val_h3_final:.4f}
- Hits@10: {val_h10_final:.4f}

## Test Metrics
- MRR: {test_mrr:.4f}
- Hits@1: {test_h1:.4f}
- Hits@3: {test_h3:.4f}
- Hits@10: {test_h10:.4f}

## Runtime Metadata
- Total nodes: {num_nodes}
- Total relations: {num_rels}
- Train triples: {len(all_train_triples)}
- Val triples: {len(all_val_triples)}
- Test triples: {len(all_test_triples)}

## Memory Usage
- Peak memory: {peak / 1024 / 1024:.2f} MB
- Current memory: {current / 1024 / 1024:.2f} MB

## Status
- Training loop completed: {'YES' if training_complete else 'NO'}
- Output non-NaN metrics: {'YES' if test_mrr > 0 and not np.isnan(test_mrr) else 'NO'}
- Memory stable: {'YES' if peak < 14 * 1024 * 1024 * 1024 else 'WARNING: High memory usage'}
"""

    report_path = OUTPUT_DIR / "stage3.1_smoke.md"
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"\nReport saved to: {report_path}")
    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Test MRR: {test_mrr:.4f}")
    logger.info(f"Test Hits@10: {test_h10:.4f}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

    return {
        "training_complete": training_complete,
        "test_mrr": test_mrr,
        "test_hits10": test_h10,
        "peak_memory_mb": peak / 1024 / 1024
    }


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["training_complete"] else 1)