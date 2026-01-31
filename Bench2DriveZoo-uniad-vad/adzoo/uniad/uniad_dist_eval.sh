#!/usr/bin/env bash

T=$(date +%m%d%H%M)

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1        # 配置文件
CKPT=$2       # checkpoint（可选）
GPUS=$3       # GPU 数（可选）
# -------------------------------------------------- #

# 如果只传了两个参数，比如：
#   ./uniad_dist_eval.sh cfg.py 1
# 那我们认为：CFG=$1, GPUS=$2, CKPT 留空
if [ -z "$GPUS" ]; then
    GPUS=$CKPT
    CKPT=""
fi

# 如果还是没设 GPUS，就默认 1
if [ -z "$GPUS" ]; then
    GPUS=1
fi

# 安全地计算 GPUS_PER_NODE（不使用三元运算）
if [ "$GPUS" -lt 8 ]; then
    GPUS_PER_NODE=$GPUS
else
    GPUS_PER_NODE=8
fi

MASTER_PORT=${MASTER_PORT:-12145}
WORK_DIR=$(echo "${CFG%.*}" | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d "${WORK_DIR}logs" ]; then
    mkdir -p "${WORK_DIR}logs"
fi

PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "$(dirname "$0")/test.py" \
    "$CFG" \
    ${CKPT:+$CKPT} \
    --launcher pytorch ${@:4} \
    --eval bbox \
    --show-dir "${WORK_DIR}" \
    2>&1 | tee "${WORK_DIR}logs/eval.$T"
