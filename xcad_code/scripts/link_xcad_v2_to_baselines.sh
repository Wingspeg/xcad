#!/bin/bash
# link_xcad_v2_to_baselines.sh
#
# 把 xcad_code/xcad_model/data/xcad_v2 里三个基线实际读取的那 7 个文件复制进
# baselines/{CEN,RE-GCN,RETIA}/data/xcad_v2/，复制后逐文件核对校验和。
#
# 为什么只复制 7 个、而不是把 13 个全复制一遍:
#   ours 的数据目录里还带 edge_weights_{train,valid,test}.npy 与
#   node_feat_{algo,compute,data}.npy。它们由 src/data/export_edge_weights.py 产出,
#   只被 ours 的边权与静态画像通路读取; 三个基线的 src/main.py 只读
#   train/valid/test.txt、entity2id.txt、relation2id.txt、stat.txt、node_features.npy。
#   多放不会改变任何基线读数, 但会让同一份派生数据在仓库里存四遍。
#
# 编号口径(全局四段, 见 G-4):
#   algo [0,102609] / data [102610,122987] / compute [122988,124884] / gpu [124885,124890]
#   旧编号 data/xcad 里 GPU 落在 [0,5], 与 algo/data/compute 段互相撞号,
#   所以基线不能拿旧编号数据跑——那会得到与 ours 不同尺子的读数。
#
# 门禁(任一 FAIL 则退出码非 0):
#   G-1 源目录 7 个文件齐全且非空
#   G-2 每个目标目录复制后与源文件校验和相同
#   G-3 entity2id.txt 行数 == 124891(三要素图实体数)
#   G-4 train.txt 的 tail 最大 id == 124890, 且落在 GPU 段 [124885,124890] 的边 > 0
#       (这两条同时成立才说明这是全局四段编号; 旧编号数据下 max tail 只有 5)
#   G-5 relation2id.txt 恰好 4 行, r1_suits 的 id 为 1
#
# 用法: bash xcad_code/scripts/link_xcad_v2_to_baselines.sh
#   幂等; 目标已是同一批文件时只会重跑校验。

set -u

# 仓库根由脚本自身位置推导, 不写死任何机器上的绝对路径。
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

SRC="$REPO/xcad_code/xcad_model/data/xcad_v2"
BASELINES="CEN RE-GCN RETIA"
FILES="train.txt valid.txt test.txt entity2id.txt relation2id.txt stat.txt node_features.npy"

fail() { echo "GATES FAIL: $*"; exit 1; }

sum() {  # 跨 macOS / Linux 的校验和
    if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1
    else md5 -q "$1"
    fi
}

echo "[staging] SRC=$SRC"

# --- G-1 源目录齐全 -----------------------------------------------------------
[ -d "$SRC" ] || fail "缺少源数据目录 $SRC"
for f in $FILES; do
    [ -s "$SRC/$f" ] || fail "G-1 源文件缺失或为空: $SRC/$f"
done
echo "G-1 PASS 7/7 源文件齐全"

# --- G-2 复制 + 校验和一致 ----------------------------------------------------
for b in $BASELINES; do
    dst="$REPO/baselines/$b/data/xcad_v2"
    [ -d "$REPO/baselines/$b" ] || fail "G-2 基线目录不存在: baselines/$b"
    mkdir -p "$dst" || fail "G-2 无法创建 $dst"
    for f in $FILES; do
        [ "$(sum "$SRC/$f")" = "$(sum "$dst/$f" 2>/dev/null || echo x)" ] \
            || cp "$SRC/$f" "$dst/$f" || fail "G-2 复制失败: $dst/$f"
        [ "$(sum "$SRC/$f")" = "$(sum "$dst/$f")" ] \
            || fail "G-2 校验和不一致: $b/$f"
    done
    echo "G-2 PASS $b: 7/7 校验和与源一致"
done

# --- G-3 / G-4 / G-5 编号与 schema 不变量 -------------------------------------
n_ents=$(wc -l < "$SRC/entity2id.txt" | tr -d ' ')
[ "$n_ents" = "124891" ] || fail "G-3 entity2id.txt 行数 = $n_ents, 期望 124891"
echo "G-3 PASS entity 数 = $n_ents"

max_tail=$(awk '{if(NR==1||$3>m)m=$3} END{print m}' "$SRC/train.txt")
[ "$max_tail" = "124890" ] \
    || fail "G-4 train.txt 的 max tail = $max_tail, 期望 124890(全局四段编号的 GPU 段上界)"
gpu_edges=$(awk '$3>=124885 && $3<=124890' "$SRC/train.txt" | wc -l | tr -d ' ')
[ "$gpu_edges" -gt 0 ] || fail "G-4 train.txt 里没有 tail 落在 GPU 段的边"
echo "G-4 PASS 全局四段编号 (max tail=$max_tail, GPU 段训练边=$gpu_edges)"

n_rels=$(wc -l < "$SRC/relation2id.txt" | tr -d ' ')
r1_id=$(awk '$1=="r1_suits"{print $2}' "$SRC/relation2id.txt")
{ [ "$n_rels" = "4" ] && [ "$r1_id" = "1" ]; } \
    || fail "G-5 relation2id.txt: 行数=$n_rels(期望 4), r1_suits id=$r1_id(期望 1)"
echo "G-5 PASS 4 类关系, r1_suits = $r1_id"

echo "GATES PASS 三份基线数据已就位; 运行方式见 xcad_code/scripts/README.md"
