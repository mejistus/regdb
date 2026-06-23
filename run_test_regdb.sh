set -euo pipefail

DATA_DIR=${DATA_DIR:-/mnt/datasets/RegDB}
LOGS_DIR=${LOGS_DIR:-logs}
BATCH_SIZE=${BATCH_SIZE:-64}
GPU=${CUDA_VISIBLE_DEVICES:-0}

CUDA_VISIBLE_DEVICES=$GPU python test_regdb.py -b "$BATCH_SIZE" -a agw -d regdb_rgb \
--iters 100 --momentum 0.1 --eps 0.6 --num-instances 16 \
--data-dir "$DATA_DIR" --logs-dir "$LOGS_DIR"
