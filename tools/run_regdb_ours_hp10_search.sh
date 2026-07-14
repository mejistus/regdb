#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/agent/regdb}"
PYTHON="${PYTHON:-/mnt/conda/envs/regdb/bin/python}"
DATA_DIR="${DATA_DIR:-data/RegDB}"
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

# Strict search: each candidate trains its own MDUE stage1 and CGCF stage2.
# The already completed paper setting S=3,p=0.10 is summarized, not rerun.
CANDIDATES="${CANDIDATES:-s3_p005:3:0.05 s2_p005:2:0.05 s4_p005:4:0.05 s2_p010:2:0.10}"

GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-2000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-20}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/datasets/regdb}"
LOG_FILE="${LOG_FILE:-logs/regdb_ours_hp10_search.log}"

cd "$ROOT"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "HP10_SEARCH_START:$(date -Is)"
echo "ROOT:$ROOT GPU:$GPU TRIALS:$TRIALS CANDIDATES:$CANDIDATES"
echo "METRIC:10-trial mean best Rank-1, tie-break mean best mAP"
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

  local spill="$ARCHIVE_DIR/spill_hp10_$(date +%Y%m%d_%H%M%S)"
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
    --mdue-samples "$mdue_samples"
  )
  if [ "$use_cgcf" = "1" ]; then
    args+=(--use-cgcf)
  fi

  wait_for_gpu
  ensure_space
  echo "TRIAL_START:$label stage=$stage trial=$trial mdue_samples=$mdue_samples dropout=$dropout cgcf=$use_cgcf $(date -Is)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "${args[@]}"
  echo "TRIAL_DONE:$label stage=$stage trial=$trial $(date -Is)"
}

for spec in $CANDIDATES; do
  IFS=: read -r tag mdue_samples dropout <<<"$spec"
  stage1_folder="regdb_s1_hp10_${tag}"
  stage2_folder="regdb_s2_hp10_${tag}_cgcf"
  label="PCLHD+MDUE+CGCF hp10 ${tag}"
  echo "CONFIG_START:$label stage1=$stage1_folder stage2=$stage2_folder mdue_samples=$mdue_samples dropout=$dropout cgcf=1 $(date -Is)"

  for trial in $TRIALS; do
    if run_complete "$stage1_folder" "$trial"; then
      echo "SKIP_COMPLETE:$stage1_folder trial=$trial"
    else
      run_train "stage1" "$label" "$trial" "$stage1_folder" "$stage2_folder" "$mdue_samples" "$dropout" 0
    fi

    if run_complete "$stage2_folder" "$trial"; then
      echo "SKIP_COMPLETE:$stage2_folder trial=$trial"
    else
      run_train "stage2" "$label" "$trial" "$stage1_folder" "$stage2_folder" "$mdue_samples" "$dropout" 1
    fi

    "$PYTHON" tools/summarize_regdb_ours_hp10_search.py --trials 1-10 || true
  done

  echo "CONFIG_DONE:$label $(date -Is)"
  "$PYTHON" tools/summarize_regdb_ours_hp10_search.py --trials 1-10 || true
done

echo "HP10_SEARCH_DONE:$(date -Is)"
"$PYTHON" tools/summarize_regdb_ours_hp10_search.py --trials 1-10 || true
