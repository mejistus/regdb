#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/agent/regdb}"
PYTHON="${PYTHON:-/mnt/conda/envs/regdb/bin/python}"
TRAIN_LOG="${TRAIN_LOG:-logs/paper_amp_ablation.log}"
PUBLISH_LOG="${PUBLISH_LOG:-logs/paper_amp_publish.log}"
POLL_SECONDS="${POLL_SECONDS:-600}"
TRIALS="${TRIALS:-1-10}"
AUTO_GIT="${AUTO_GIT:-1}"

cd "$ROOT"
mkdir -p "$(dirname "$PUBLISH_LOG")"
exec > >(tee -a "$PUBLISH_LOG") 2>&1

echo "PAPER_PUBLISH_WATCH_START:$(date -Is)"
echo "ROOT:$ROOT TRAIN_LOG:$TRAIN_LOG TRIALS:$TRIALS"

while ! grep -q "PAPER_AMP_ABLATION_DONE" "$TRAIN_LOG" 2>/dev/null; do
  echo "WAITING_PAPER_AMP:$(date -Is)"
  sleep "$POLL_SECONDS"
done

echo "PAPER_AMP_DONE_DETECTED:$(date -Is)"
"$PYTHON" tools/build_regdb_stats.py --trials "$TRIALS"
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

if [ "$AUTO_GIT" = "1" ]; then
  git add htmls/stats.html
  if git diff --cached --quiet; then
    echo "COMMIT_SKIPPED:no report changes"
  else
    git commit -m "Update paper AMP ablation report"
    if ! git push; then
      echo "PUSH_FAILED:first attempt, retrying with proxy if available"
      if [ -f /etc/profile.d/clash.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/clash.sh
      fi
      proxy_on 2>/dev/null || true
      git push
    fi
    echo "PUSHED_COMMIT:$(git rev-parse HEAD)"
  fi
fi

echo "PAPER_PUBLISH_DONE:$(date -Is)"
