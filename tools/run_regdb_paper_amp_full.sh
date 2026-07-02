#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/agent/regdb}"
PYTHON="${PYTHON:-/mnt/conda/envs/regdb/bin/python}"
DATA_DIR="${DATA_DIR:-/mnt/datasets/RegDB}"
LOGS_DIR="${LOGS_DIR:-logs}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

TRIALS="${TRIALS:-1 2 3 4 5 6 7 8 9 10}"
EPOCHS="${EPOCHS:-50}"
ITERS="${ITERS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
NUM_INSTANCES="${NUM_INSTANCES:-16}"
MOMENTUM="${MOMENTUM:-0.1}"
EPS="${EPS:-0.3}"

GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-2000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-20}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/datasets/regdb}"
LOG_FILE="${LOG_FILE:-logs/paper_amp10_ablation.log}"

cd "$ROOT"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "PAPER_AMP_FULL_ABLATION_START:$(date -Is)"
echo "ROOT:$ROOT GPU:$GPU TRIALS:$TRIALS EPOCHS:$EPOCHS ITERS:$ITERS"

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

run_complete() {
  local folder="$1"
  local trial="$2"
  local log="$LOGS_DIR/$folder/$trial/${trial}log.txt"
  [ -f "$log" ] && grep -q "Total running time:" "$log"
}

run_train() {
  local stage="$1"
  local label="$2"
  local trial="$3"
  local stage1_log="$4"
  local stage2_log="$5"
  local mdue_samples="$6"
  local dropout="$7"
  local use_cgcf="$8"

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
    --stage "$stage"
    --stage2-batch-size "$STAGE2_BATCH_SIZE"
    --stage1-log-name "$stage1_log"
    --stage2-log-name "$stage2_log"
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

  wait_for_gpu
  ensure_space
  echo "TRIAL_START:$label stage=$stage trial=$trial $(date -Is)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "${args[@]}"
  echo "TRIAL_DONE:$label stage=$stage trial=$trial $(date -Is)"
}

run_full_config() {
  local stage1_folder="$1"
  local stage2_folder="$2"
  local label="$3"
  local mdue_samples="$4"
  local dropout="$5"
  local use_cgcf="$6"

  echo "CONFIG_START:$label stage1=$stage1_folder stage2=$stage2_folder mdue_samples=$mdue_samples dropout=$dropout cgcf=$use_cgcf"
  for trial in $TRIALS; do
    if run_complete "$stage2_folder" "$trial"; then
      echo "SKIP_COMPLETE:$stage2_folder trial=$trial"
      continue
    fi
    if run_complete "$stage1_folder" "$trial"; then
      run_train "stage2" "$label" "$trial" "$stage1_folder" "$stage2_folder" "$mdue_samples" "$dropout" "$use_cgcf"
    else
      run_train "all" "$label" "$trial" "$stage1_folder" "$stage2_folder" "$mdue_samples" "$dropout" "$use_cgcf"
    fi
  done
  echo "CONFIG_DONE:$label $(date -Is)"
}

run_stage2_config() {
  local stage1_folder="$1"
  local stage2_folder="$2"
  local label="$3"
  local mdue_samples="$4"
  local dropout="$5"
  local use_cgcf="$6"

  echo "CONFIG_START:$label stage1=$stage1_folder stage2=$stage2_folder mdue_samples=$mdue_samples dropout=$dropout cgcf=$use_cgcf"
  for trial in $TRIALS; do
    if run_complete "$stage2_folder" "$trial"; then
      echo "SKIP_COMPLETE:$stage2_folder trial=$trial"
      continue
    fi
    if ! run_complete "$stage1_folder" "$trial"; then
      echo "MISSING_STAGE1:$stage1_folder trial=$trial"
      exit 2
    fi
    run_train "stage2" "$label" "$trial" "$stage1_folder" "$stage2_folder" "$mdue_samples" "$dropout" "$use_cgcf"
  done
  echo "CONFIG_DONE:$label $(date -Is)"
}

ensure_space

run_full_config "regdb_s1_amp10_baseline" "regdb_s2_amp10_baseline" "PCLHD AMP baseline" 1 0.0 0
run_full_config "regdb_s1_amp10_mdue" "regdb_s2_amp10_mdue" "PCLHD + MDUE AMP ablation" 3 0.10 0
run_stage2_config "regdb_s1_amp10_mdue" "regdb_s2_amp10_mdue_cgcf" "PCLHD + MDUE + CGCF AMP ours" 3 0.10 1

echo "PAPER_AMP_FULL_ABLATION_DONE:$(date -Is)"
