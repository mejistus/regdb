set -euo pipefail

DATA_DIR=${DATA_DIR:-/mnt/datasets/RegDB}
LOGS_DIR=${LOGS_DIR:-logs}
BATCH_SIZE=${BATCH_SIZE:-64}
STAGE2_BATCH_SIZE=${STAGE2_BATCH_SIZE:-32}
GPU=${CUDA_VISIBLE_DEVICES:-0}
STAGE=${STAGE:-all}
TRIALS=${TRIALS:-"1 2 3 4 5 6 7 8 9 10"}
STAGE1_LOG_NAME=${STAGE1_LOG_NAME:-regdb_s1}
STAGE2_LOG_NAME=${STAGE2_LOG_NAME:-regdb_s2}
STAGE1_LOGS_DIR=${STAGE1_LOGS_DIR:-}
AMP=${AMP:-0}
MDUE_SAMPLES=${MDUE_SAMPLES:-1}
USE_CGCF=${USE_CGCF:-0}

EXTRA_ARGS=()
if [ "$AMP" = "1" ]; then
  EXTRA_ARGS+=(--amp)
fi
if [ "$MDUE_SAMPLES" -gt 1 ]; then
  EXTRA_ARGS+=(--mdue-samples "$MDUE_SAMPLES")
fi
if [ "$USE_CGCF" = "1" ]; then
  EXTRA_ARGS+=(--use-cgcf)
fi
if [ -n "$STAGE1_LOGS_DIR" ]; then
  EXTRA_ARGS+=(--stage1-logs-dir "$STAGE1_LOGS_DIR")
fi

for trial in $TRIALS
do
CUDA_VISIBLE_DEVICES=$GPU \
python train_regdb.py -b "$BATCH_SIZE" -a agw -d regdb_rgb -mb CMhybrid --iters 100 \
--momentum 0.1 --eps 0.3 --num-instances 16 --trial "$trial" \
--stage "$STAGE" --stage2-batch-size "$STAGE2_BATCH_SIZE" \
--stage1-log-name "$STAGE1_LOG_NAME" --stage2-log-name "$STAGE2_LOG_NAME" \
--data-dir "$DATA_DIR" --logs-dir "$LOGS_DIR" "${EXTRA_ARGS[@]}"
done
echo 'Done'
