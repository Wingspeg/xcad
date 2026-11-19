"""
xCAD Configuration

Central configuration for data paths and output directories.
All paths are derived from this file's location, so the code works on any machine
without modification as long as the directory structure is preserved.
"""

from pathlib import Path

# === 路径推算(基于本文件位置)===
# 本文件实际位置:  <xcad>/xcad_code/src/utils/config.py
#   _HERE        = <xcad>/xcad_code/src/utils
#   _CODE_ROOT   = <xcad>/xcad_code
#   _XCAD_ROOT   = <xcad>
# 注: 顶部注释与旧推算假设本文件在 src/config.py(少算一层), 导致
#     OUTPUT_ROOT 解析成 <xcad>/xcad_code/src/outputs 而 FileNotFoundError
#     (2026-11-12 跑 src/analysis/relation_drift.py 时暴露)。
_HERE = Path(__file__).resolve().parent              # src/utils/
_CODE_ROOT = _HERE.parent.parent                     # xcad_code/
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
