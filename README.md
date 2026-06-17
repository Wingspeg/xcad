# xCAD

**xCAD**：面向三要素调度的时序知识图谱表征——诊断、基准与方法。

本仓库提供论文 *xCAD: Temporal Knowledge Graph Representation for Tri-Element Scheduling* 的官方实现，包括：从 Alibaba GPU 集群 trace 重构三要素调度时序图谱的数据流水线、基于 RE-GCN 骨架的两个新机制（异质关系基消息传递、算法兼容性头），以及全部诊断与评测脚本。

---

## 简介

时序知识图谱（TKG）外推推理的方法与基准几乎全部面向通用事件图谱（ICEWS、GDELT），其依赖历史重复模式的归纳偏置在其他结构的图谱上是否成立，缺乏检验。xCAD 从真实 GPU 集群生产 trace 重构出首个**三要素（算力 Compute、算法 Algorithm、数据 Data）调度时序图谱**，并发现：通用 TKG 方法在核心关系"算法→GPU 适配"上**退化为频率先验**（恒预测高频众数 GPU，per-GPU macro 仅 0.20）。本文提出两个针对性机制，将该关系的 per-GPU macro 提升至 0.52，逼近随机森林上界 0.59。

**核心结果（r1\_suits, per-GPU macro Hits@1）**

| 方法 | per-GPU macro |
|---|---|
| RE-GCN / CEN / RETIA | 0.20（退化为频率先验）|
| 随机森林上界（仅用算法画像）| 0.59 |
| **xCAD（ours）** | **0.51** |

---

## 数据集：来源与重构

xCAD 基准重构自公开数据集 **Alibaba Cluster-Trace-GPU-v2020**（来源：<https://github.com/alibaba/clusterdata>，路径 `cluster-trace-gpu-v2020`）。本仓库**不直接分发重构后的图数据**，而是提供完整的重构脚本，便于从原始 trace 复现，同时尊重原数据集的许可与分发条款。

### 重构步骤

1. 从 Alibaba clusterdata 仓库下载 `cluster-trace-gpu-v2020` 原始 trace，置于 `data/raw/`。
2. 运行重构脚本，将 trace 映射为三要素时序异构图：

   ```bash
   python tools/export_to_regcn_format.py --raw-dir data/raw --out-dir data/xcad
   ```

### 图的构成（重构后）

- **节点（124,891 实体）**：算力 Compute（GPU 代次 + 机器）、算法 Algorithm（job 级，每个含 13 维资源画像）、数据 Data（数据组）。
- **关系（4 种，含逆关系共 8）**：placement、r1\_suits（算法→GPU 适配，核心关系）、r2\_requires、r3\_drives。
- **时序切分**：train / valid / test = 585,335 / 87,553 / 90,785 条边，对应 49 / 6 / 6 个时间快照。

重构遵循一条纪律：所有关系边的存在与权重均来自真实共现/运行统计，不引入人造规则；唯一的作者先验是 schema 设计（字段到节点/关系类型的映射）。重构细节见论文 §3 与脚本注释。

---

## 环境

- Python 3.11
- PyTorch 2.x + CUDA（在单块 RTX 3090, 24GB 上训练）
- DGL

```bash
# 推荐使用 uv 或 venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 复现实验

所有实验默认历史长度 `h=6`，报告三随机种子（42 / 123 / 2024）。

### 基线（RE-GCN 骨架）

```bash
python main.py -d xcad --gpu 0 \
  --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2 \
  --n-bases 100 --train-history-len 6 --dropout 0.4 \
  --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4 \
  --weight-decay 1e-4 --lr 0.001 --entity-prediction \
  --n-epochs 30 --evaluate-every 5 --seed 42
```

### 点1：异质关系基消息传递

在基线命令后追加：

```bash
  --hetero --n-msg-basis 2
```

### 点2：算法兼容性头

在基线命令后追加：

```bash
  --compat --compat-lambda 5.0 --compat-aux-weight 0.5
```

### xCAD 完整方法（点1 + 点2）

```bash
python main.py -d xcad --gpu 0 \
  --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2 \
  --n-bases 100 --train-history-len 6 --dropout 0.4 \
  --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4 \
  --weight-decay 1e-4 --lr 0.001 --entity-prediction \
  --n-epochs 30 --evaluate-every 5 --seed 42 \
  --hetero --n-msg-basis 2 --compat --compat-lambda 5.0 --compat-aux-weight 0.5
```

测试集评测会输出总体 filter MRR/Hits@k、按关系的 per-relation MRR，以及核心关系 r1\_suits 的 per-GPU Hits@1 与 macro。

### 诊断与上界探测

```bash
# 频率先验诊断（加载 checkpoint，输出预测熵 / 混淆 / per-GPU Hits@1）
python diagnose_r1.py --checkpoint <path-to-checkpoint>

# 随机森林上界探测（仅用算法画像；--split-by-algo 为按算法严格切分，预测新算法）
python diagnose_sklearn.py
python diagnose_sklearn.py --split-by-algo
```

---

## 主要参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--train-history-len` | 历史快照长度 | 6 |
| `--hetero` | 启用异质关系基消息传递（点1）| 关闭 |
| `--n-msg-basis` | 基矩阵数量 $B$ | 2 |
| `--compat` | 启用算法兼容性头（点2）| 关闭 |
| `--compat-lambda` | 兼容性偏置融合权重 $\lambda$ | 5.0 |
| `--compat-aux-weight` | 可选 GPU 辅助分类损失权重 $\beta$ | 0.5 |
| `--seed` | 随机种子 | 42 |

关闭 `--hetero` 与 `--compat` 时，模型与原始 RE-GCN 逐比特一致。

---

## 引用

如果本工作对你有帮助，请引用（占位，正式发表后更新）：

```bibtex
@article{xcad,
  title   = {xCAD: Temporal Knowledge Graph Representation for Tri-Element Scheduling},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

---

## 致谢

本工作的骨架与评测协议基于 RE-GCN（Li et al., SIGIR 2021）。数据集重构自 Alibaba Cluster-Trace-GPU-v2020。

## 许可

代码以 MIT 许可发布。重构所依赖的原始 trace 数据请遵循 Alibaba clusterdata 的原始许可条款。
