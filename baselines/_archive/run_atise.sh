#!/bin/bash
# run_atise.sh
# 在服务器上运行 ATISE (TeRo / ATiSE) 基线实验
# 使用 xCAD 转换数据
#
# 用法: bash baselines/run_atise.sh [tero|atise] [dim] [lr] [epochs]
# 示例: bash baselines/run_atise.sh tero 64 0.1 200

set -e

MODEL=${1:-tero}
DIM=${2:-64}
LR=${3:-0.1}
EPOCHS=${4:-200}

# run_atise.sh 在 baselines/ 下，Main.py 在 baselines/ATISE/ 下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATISE_DIR="${SCRIPT_DIR}/ATISE"

cd "$ATISE_DIR"

# 激活虚拟环境
source "${SCRIPT_DIR}/.venv/bin/activate"

echo "=========================================="
echo "  ATISE Baseline Experiment"
echo "  Model: ${MODEL}"
echo "  Dim:   ${DIM}"
echo "  LR:    ${LR}"
echo "  Epochs:${EPOCHS}"
echo "  Data:  xcad"
echo "=========================================="

python3 -u Main.py \
    --model     "${MODEL}" \
    --dataset   "xcad" \
    --dim       "${DIM}" \
    --lr        "${LR}" \
    --max_epoch "${EPOCHS}" \
    --gamma     1.0 \
    --loss      logloss \
    --eta       5 \
    --timedisc  0 \
    --cuda      True \
    --gran      1 \
    --batch     512 \

echo "Done. Results saved in baselines/ATISE/xcad/"
