#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/agent/regdb}"
PYTHON="${PYTHON:-/mnt/conda/envs/regdb/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-600}"
LOG_FILE="${LOG_FILE:-logs/postprocess_publish.log}"
WAIT_SESSIONS="${WAIT_SESSIONS:-regdb_train regdb_follow regdb_resume}"

cd "$ROOT"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "POSTPROCESS_START:$(date -Is)"

has_wait_session() {
  local session
  for session in $WAIT_SESSIONS; do
    if tmux has-session -t "$session" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

while has_wait_session; do
  echo "WAITING:$(date -Is)"
  sleep "$POLL_SECONDS"
done

echo "TRAINING_SESSIONS_DONE:$(date -Is)"

stage2_status="missing"
trials2_status="missing"
resume_status="missing"
if [ -f logs/full_train_stage2_trial1.log ]; then
  stage2_status="$(grep -E 'TRAIN_EXIT:' logs/full_train_stage2_trial1.log | tail -1 | cut -d: -f2- || true)"
fi
if [ -f logs/full_train_trials2_10.log ]; then
  trials2_status="$(grep -E 'TRAIN_EXIT:' logs/full_train_trials2_10.log | tail -1 | cut -d: -f2- || true)"
fi
if [ -f logs/full_train_resume_trials8_10_batch32.log ]; then
  resume_status="$(grep -E 'TRAIN_EXIT:' logs/full_train_resume_trials8_10_batch32.log | tail -1 | cut -d: -f2- || true)"
fi

echo "STAGE2_TRIAL1_EXIT:${stage2_status}"
echo "TRIALS2_10_EXIT:${trials2_status}"
echo "RESUME_TRIALS8_10_EXIT:${resume_status}"

if [ "$stage2_status" != "0" ] || { [ "$trials2_status" != "0" ] && [ "$resume_status" != "0" ]; }; then
  "$PYTHON" tools/build_regdb_stats.py --trials 1-10 || true
  echo "POSTPROCESS_ABORT:training did not finish cleanly"
  exit 1
fi

"$PYTHON" tools/build_regdb_stats.py --trials 1-10

"$PYTHON" - <<'PY'
from pathlib import Path
import re

html = Path("htmls/stats.html").read_text(encoding="utf-8")
scripts = [
    match.group(1)
    for match in re.finditer(r"<script(?![^>]*application/json)[^>]*>(.*?)</script>", html, re.S)
]
Path("/tmp/regdb_stats_inline.js").write_text("\n".join(scripts), encoding="utf-8")
print(f"INLINE_SCRIPT_BLOCKS:{len(scripts)}")
PY

if command -v node >/dev/null 2>&1; then
  node --check /tmp/regdb_stats_inline.js
else
  echo "NODE_CHECK_SKIPPED:node not found"
fi

git status --short
git add .gitignore \
  clustercontrast/datasets/regdb_ir.py \
  clustercontrast/datasets/regdb_rgb.py \
  clustercontrast/evaluation_metrics/ranking.py \
  clustercontrast/utils/faiss_rerank.py \
  prepare_regdb.py \
  run_test_regdb.sh \
  run_train_regdb.sh \
  test_regdb.py \
  train_regdb.py \
  tools/build_regdb_stats.py \
  tools/wait_and_publish_regdb.sh \
  htmls/stats.html

if git diff --cached --quiet; then
  echo "COMMIT_SKIPPED:no staged changes"
else
  git commit -m "Reproduce RegDB training and report metrics"
fi

git push
echo "PUSHED_COMMIT:$(git rev-parse HEAD)"
echo "POSTPROCESS_DONE:$(date -Is)"
