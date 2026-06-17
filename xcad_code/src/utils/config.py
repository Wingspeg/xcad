"""
xCAD Configuration

Central configuration for data paths and output directories.
All paths are derived from this file's location, so the code works on any machine
without modification as long as the directory structure is preserved.
"""

from pathlib import Path

# === 路径推算(基于本文件位置)===
# 本文件:        <xcad>/xcad_code/src/config.py
# xcad_code 根:  <xcad>/xcad_code/
# xcad 项目根:   <xcad>/
_HERE = Path(__file__).resolve().parent              # src/
_CODE_ROOT = _HERE.parent                            # xcad_code/
_XCAD_ROOT = _CODE_ROOT.parent                       # xcad/

# === 主要路径 ===
DATA_ROOT = str(_XCAD_ROOT / "datasets" / "alibaba-cluster-trace-gpu-v2020")
OUTPUT_ROOT = str(_CODE_ROOT / "outputs")

# === 数据处理参数 ===
MIN_GROUP_SIZE = 5

# === 阶段二:数据切分与负采样参数 ===
FRAMEWORK = "dgl"
TRAIN_DAYS = 60
VAL_DAYS = 5
TEST_DAYS = 4
NEG_SAMPLE_RATIO = 5
NEG_SAMPLE_STRATEGY = "type_aware"
RANDOM_SEED = 42


# === 启动时打印路径(便于调试) ===
if __name__ == "__main__":
    print(f"XCAD_ROOT:   {_XCAD_ROOT}")
    print(f"CODE_ROOT:   {_CODE_ROOT}")
    print(f"DATA_ROOT:   {DATA_ROOT}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
