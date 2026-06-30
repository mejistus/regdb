#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/agent/regdb}"
OLD_LOG="${OLD_LOG:-logs/regdb_s2_amp_mdue/1/1log.txt}"
POLL_SECONDS="${POLL_SECONDS:-120}"

cd "$ROOT"

echo "SWITCH_TO_PAPER_AMP_FULL_START:$(date -Is)"
echo "WAIT_OLD_TRIAL_LOG:$OLD_LOG"

while [ ! -f "$OLD_LOG" ] || ! grep -q "Total running time:" "$OLD_LOG"; do
  echo "WAITING_OLD_MDUE_TRIAL1:$(date -Is)"
  sleep "$POLL_SECONDS"
done

echo "OLD_MDUE_TRIAL1_COMPLETE:$(date -Is)"

pkill -f "tools/wait_and_publish_paper_amp.sh" 2>/dev/null || true
pkill -f "tools/run_regdb_paper_amp_ablation.sh" 2>/dev/null || true
pkill -f "train_regdb.py .*--stage2-log-name regdb_s2_amp_mdue" 2>/dev/null || true
pkill -f "train_regdb.py .*--stage2-log-name regdb_s2_amp_mdue_cgcf" 2>/dev/null || true
sleep 5

echo "STARTING_FULL_STAGE_AMP:$(date -Is)"
TRIALS="${TRIALS:-1 2 3}" LOG_FILE="${LOG_FILE:-logs/paper_amp_full_ablation.log}" tools/run_regdb_paper_amp_full.sh
