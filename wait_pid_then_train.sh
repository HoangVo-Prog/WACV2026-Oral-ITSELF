#!/usr/bin/env bash
set -euo pipefail

# Replace this with the PID that must finish before training starts.
STATIC_PID="187050"
CHECK_INTERVAL_SECONDS=30
LOG_DIR="logs"

# Keep the script alive when launched as: nohup bash wait_pid_then_train.sh &
trap '' HUP

if [[ ! "$STATIC_PID" =~ ^[0-9]+$ ]]; then
  echo "Please edit STATIC_PID in $0 before running."
  exit 1
fi

pid_exists() {
  if [[ -d /proc ]]; then
    [[ -d "/proc/$1" ]]
  else
    ps -p "$1" >/dev/null 2>&1
  fi
}

if ! pid_exists "$STATIC_PID"; then
  echo "PID ${STATIC_PID} is not visible from this shell. Refusing to start training."
  echo "Check with: ps -p ${STATIC_PID} -o pid,ppid,user,stat,cmd"
  exit 1
fi

echo "Waiting for PID ${STATIC_PID} to finish..."
while pid_exists "$STATIC_PID"; do
  date "+%Y-%m-%d %H:%M:%S still waiting for PID ${STATIC_PID}..."
  sleep "$CHECK_INTERVAL_SECONDS"
done

echo "PID ${STATIC_PID} has finished. Starting training..."

DATASET_NAME="RSTPReid"
mkdir -p "$LOG_DIR"
RUN_LOG="${LOG_DIR}/PPL_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log"

nohup env CUDA_VISIBLE_DEVICES=0 \
  python3 train.py \
  --name PPL \
  --output_dir 'ITSELF' \
  --dataset_name "$DATASET_NAME" \
  --loss_names 'tal+cid' \
  --num_epoch 15 \
  --only_global \
  --batch_size 256 \
  --prototype \
  --use_loss_id \
  --use_loss_rank \
  --nohup \
  > "$RUN_LOG" 2>&1 &

TRAIN_PID=$!
echo "Training started with PID ${TRAIN_PID}. Log: ${RUN_LOG}"
# --return_all \
# --topk_type 'custom' \
# --modify_k
