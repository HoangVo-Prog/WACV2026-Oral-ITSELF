#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="RSTPReid"
LOSS_NAMES="tal+cid"
RUN_TIME="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="ITSELF"
NOHUP_LOG_DIR="logs/${DATASET_NAME}/${RUN_TIME}_${RUN_NAME}_${LOSS_NAMES}"
NOHUP_LOG_PATH="${NOHUP_LOG_DIR}/${RUN_TIME}.log"

mkdir -p "$NOHUP_LOG_DIR"

nohup env CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
--name PPL \
--output_dir 'ITSELF' \
--dataset_name "$DATASET_NAME" \
--loss_names "$LOSS_NAMES" \
--num_epoch 60 \
--only_global \
--nohup \
--run_time "$RUN_TIME" \
--nohup_log_dir logs \
>> "$NOHUP_LOG_PATH" 2>&1 &

PID=$!
disown "$PID" 2>/dev/null || true

echo "PID: ${PID}"
echo "Log file: ${NOHUP_LOG_PATH}"

# --return_all \
# --topk_type 'custom' \
# --modify_k
