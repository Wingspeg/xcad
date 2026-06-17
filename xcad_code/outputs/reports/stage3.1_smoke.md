# RE-GCN Stage 3.1 Smoke Test Report

## Configuration
- Edge types: ['r1_suits', 'placement']
- Max time windows: 10
- Train windows: [1, 7]
- Val windows: [8, 8]
- Test windows: [9, 10]
- Hidden dimension: 64
- Number of layers: 1
- Dropout: 0.1
- Learning rate: 0.001
- Batch size: 128
- Negative sampling ratio: 5
- Max epochs: 5

## Training Summary
- Training completed: True
- Total epochs run: 5
- Total training time: 2455.34s
- Final training loss: 9.1584

## Training Loss Curve
| Epoch | Loss | Time (s) |
|-------|------|----------|
| 1 | 10.3338 | 102.54 |
| 2 | 9.6119 | 101.80 |
| 3 | 9.3748 | 102.72 |
| 4 | 9.2544 | 116.84 |
| 5 | 9.1584 | 112.49 |

## Validation Metrics
- MRR: 0.0127
- Hits@1: 0.0040
- Hits@3: 0.0160
- Hits@10: 0.0280

## Test Metrics
- MRR: 0.0651
- Hits@1: 0.0460
- Hits@3: 0.0660
- Hits@10: 0.1020

## Runtime Metadata
- Total nodes: 124891
- Total relations: 4
- Train triples: 14000
- Val triples: 2000
- Test triples: 4000

## Memory Usage
- Peak memory: 228.20 MB
- Current memory: 64.18 MB

## Status
- Training loop completed: YES
- Output non-NaN metrics: YES
- Memory stable: YES
