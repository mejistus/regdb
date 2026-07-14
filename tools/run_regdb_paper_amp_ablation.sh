#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/agent/regdb}"
PYTHON="${PYTHON:-/mnt/conda/envs/regdb/bin/python}"
DATA_DIR="${DATA_DIR:-data/RegDB}"
LOGS_DIR="${LOGS_DIR:-logs}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

TRIALS="${TRIALS:-1 2 3}"
EPOCHS="${EPOCHS:-50}"
ITERS="${ITERS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
NUM_INSTANCES="${NUM_INSTANCES:-16}"
MOMENTUM="${MOMENTUM:-0.1}"
EPS="${EPS:-0.3}"
STAGE1_LOG_NAME="${STAGE1_LOG_NAME:-regdb_s1}"

GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-2000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-20}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/datasets/regdb}"
LOG_FILE="${LOG_FILE:-logs/paper_amp_ablation.log}"

cd "$ROOT"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "PAPER_AMP_ABLATION_START:$(date -Is)"
echo "ROOT:$ROOT GPU:$GPU TRIALS:$TRIALS EPOCHS:$EPOCHS ITERS:$ITERS"
if [[ "$DATA_DIR" != /* ]]; then
  printf '\033[31mWARNING: 您使用的是相对路径 --data-dir "%s"；请从仓库根目录运行，或确认数据已放在该相对路径下。\033[0m\n' "$DATA_DIR"
fi

free_gb() {
  df -BG "$ROOT" | awk 'NR==2 {gsub("G","",$4); print $4}'
}

ensure_space() {
  local free
  free="$(free_gb)"
  echo "SPACE_FREE_GB:$free"
  if [ "$free" -ge "$MIN_FREE_GB" ]; then
    return 0
  fi

  local spill="$ARCHIVE_DIR/spill_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$spill"
  echo "SPACE_LOW:moving non-best checkpoint.pth.tar files to $spill"
  while IFS= read -r -d '' file; do
    local rel="${file#./}"
    mkdir -p "$spill/$(dirname "$rel")"
    mv "$file" "$spill/$rel"
    echo "MOVED:$rel"
  done < <(find ./logs -type f -name checkpoint.pth.tar -print0 2>/dev/null || true)
  echo "SPACE_FREE_GB_AFTER:$(free_gb)"
}

gpu_ready() {
  local used util
  used="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  util="$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  echo "GPU_STATUS:$(date -Is):used=${used}MiB util=${util}%"
  [ "$used" -le "$GPU_MAX_USED_MB" ] && [ "$util" -le "$GPU_MAX_UTIL" ]
}

wait_for_gpu() {
  while ! gpu_ready; do
    sleep "$POLL_SECONDS"
  done
}

trial_complete() {
  local folder="$1"
  local trial="$2"
  local log="$LOGS_DIR/$folder/$trial/${trial}log.txt"
  [ -f "$log" ] && grep -q "Total running time:" "$log"
}

run_config() {
  local folder="$1"
  local label="$2"
  local mdue_samples="$3"
  local dropout="$4"
  local use_cgcf="$5"

  echo "CONFIG_START:$label folder=$folder mdue_samples=$mdue_samples dropout=$dropout cgcf=$use_cgcf"
  for trial in $TRIALS; do
    if trial_complete "$folder" "$trial"; then
      echo "SKIP_COMPLETE:$folder trial=$trial"
      continue
    fi

    wait_for_gpu
    ensure_space

    args=(
      train_regdb.py
      -b "$BATCH_SIZE"
      -a agw
      -d regdb_rgb
      -mb CMhybrid
      --epochs "$EPOCHS"
      --iters "$ITERS"
      --momentum "$MOMENTUM"
      --eps "$EPS"
      --num-instances "$NUM_INSTANCES"
      --trial "$trial"
      --stage stage2
      --stage2-batch-size "$STAGE2_BATCH_SIZE"
      --stage1-log-name "$STAGE1_LOG_NAME"
      --stage2-log-name "$folder"
      --data-dir "$DATA_DIR"
      --logs-dir "$LOGS_DIR"
      --amp
      --dropout "$dropout"
    )
    if [ "$mdue_samples" -gt 1 ]; then
      args+=(--mdue-samples "$mdue_samples")
    fi
    if [ "$use_cgcf" = "1" ]; then
      args+=(--use-cgcf)
    fi

    echo "TRIAL_START:$label trial=$trial $(date -Is)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "${args[@]}"
    echo "TRIAL_DONE:$label trial=$trial $(date -Is)"
  done
  echo "CONFIG_DONE:$label $(date -Is)"
}

ensure_space
wait_for_gpu

run_config "regdb_s2_amp_baseline" "PCLHD AMP baseline" 1 0.0 0
run_config "regdb_s2_amp_mdue" "PCLHD + MDUE AMP" 3 0.10 0
run_config "regdb_s2_amp_mdue_cgcf" "PCLHD + MDUE + CGCF AMP" 3 0.10 1

echo "PAPER_AMP_ABLATION_DONE:$(date -Is)"
