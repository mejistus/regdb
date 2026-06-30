#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/agent/regdb}"
PYTHON="${PYTHON:-/mnt/conda/envs/regdb/bin/python}"
TRAIN_LOG="${TRAIN_LOG:-logs/paper_amp_full_ablation.log}"
PUBLISH_LOG="${PUBLISH_LOG:-logs/paper_amp_full_publish.log}"
STATUS_LOG="${STATUS_LOG:-logs/paper_amp_full_status.log}"
TRIALS="${TRIALS:-1-3}"
POLL_SECONDS="${POLL_SECONDS:-600}"

cd "$ROOT" || exit 1
mkdir -p "$(dirname "$STATUS_LOG")"
exec > >(tee -a "$STATUS_LOG") 2>&1

echo "PAPER_AMP_FULL_STATUS_WATCH_START:$(date -Is)"
echo "ROOT:$ROOT TRAIN_LOG:$TRAIN_LOG PUBLISH_LOG:$PUBLISH_LOG TRIALS:$TRIALS POLL_SECONDS:$POLL_SECONDS"

while true; do
  echo "STATUS_TICK:$(date -Is)"
  git log -1 --oneline 2>/dev/null || true
  "$PYTHON" tools/summarize_paper_amp_full.py --trials "$TRIALS" 2>&1 || true
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 || true
  df -h /agent/regdb /datasets/regdb /mnt/datasets 2>/dev/null || true
  rg -n "TRIAL_START|TRIAL_DONE|CONFIG_START|CONFIG_DONE|PAPER_AMP_FULL_ABLATION_DONE|Traceback|RuntimeError|CUDA out of memory|nan|inf" "$TRAIN_LOG" 2>/dev/null | tail -80 || true
  rg -n "PAPER_FULL_PUBLISH|PAPER_AMP_FULL_DONE_DETECTED|INLINE_SCRIPT_BLOCKS|SyntaxError|PUSHED_COMMIT|PUSH_FAILED|WAITING|DONE" "$PUBLISH_LOG" 2>/dev/null | tail -40 || true

  if grep -q "PAPER_FULL_PUBLISH_DONE" "$PUBLISH_LOG" 2>/dev/null; then
    echo "PAPER_AMP_FULL_STATUS_WATCH_DONE:$(date -Is)"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
