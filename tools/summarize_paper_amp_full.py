#!/usr/bin/env python3
"""Summarize full-stage paper-parameter AMP ablation logs."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any


CONFIGS = [
    ("amp10_baseline", "regdb_s1_amp10_baseline", "regdb_s2_amp10_baseline"),
    ("amp10_mdue", "regdb_s1_amp10_mdue", "regdb_s2_amp10_mdue"),
    ("amp10_mdue_cgcf", "regdb_s1_amp10_mdue", "regdb_s2_amp10_mdue_cgcf"),
]
STAGE2_ONLY_CONFIGS = {"amp10_mdue_cgcf"}
DEFAULT_EPOCHS = 50
TRAIN_LOG = Path("logs/paper_amp10_ablation.log")

EPOCH_RE = re.compile(
    r"Finished epoch\s+(?P<epoch>\d+)\s+"
    r"model R1:\s*(?P<rank1>[0-9.]+)%\s+"
    r"model mAP:\s*(?P<map>[0-9.]+)%\s+"
    r"best R1:\s*(?P<best_rank1>[0-9.]+)%\s+"
    r"best mAP:\s*(?P<best_map>[0-9.]+)%"
    r"\(best_epoch:(?P<best_epoch>\d+)\)"
)
START_RE = re.compile(r"PAPER_AMP_FULL_ABLATION_START:(?P<timestamp>\S+)")


def parse_trials(raw: str) -> list[int]:
    trials: list[int] = []
    for part in raw.replace(",", " ").split():
        if "-" in part:
            start, end = part.split("-", 1)
            trials.extend(range(int(start), int(end) + 1))
        else:
            trials.append(int(part))
    return sorted(dict.fromkeys(trials))


def parse_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "pending", "done": False, "log_path": str(path)}

    text = path.read_text(errors="replace")
    matches = list(EPOCH_RE.finditer(text))
    if not matches:
        return {"status": "started", "done": False, "log_path": str(path)}

    last = matches[-1].groupdict()
    row: dict[str, Any] = {
        "status": "complete" if "Total running time:" in text else "running",
        "done": "Total running time:" in text,
        "log_path": str(path),
    }
    for key in ("epoch", "best_epoch"):
        row[key] = int(last[key])
    for key in ("rank1", "map", "best_rank1", "best_map"):
        row[key] = float(last[key])
    return row


def collect(root: Path, trials: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config, stage1_folder, stage2_folder in CONFIGS:
        for trial in trials:
            stage2_path = root / "logs" / stage2_folder / str(trial) / f"{trial}log.txt"
            row = parse_log(stage2_path)
            row.update(
                {
                    "config": config,
                    "folder": stage2_folder,
                    "stage1_folder": stage1_folder,
                    "stage2_folder": stage2_folder,
                    "trial": trial,
                }
            )
            if config not in STAGE2_ONLY_CONFIGS and not row.get("done") and "epoch" not in row:
                stage1_path = root / "logs" / stage1_folder / str(trial) / f"{trial}log.txt"
                row["stage1"] = parse_log(stage1_path)
            rows.append(row)
    return rows


def epoch_progress(log_row: dict[str, Any], epochs: int) -> int:
    if log_row.get("done"):
        return epochs
    if "epoch" not in log_row:
        return 0
    return max(0, min(int(log_row["epoch"]) + 1, epochs))


def row_progress(row: dict[str, Any], epochs: int = DEFAULT_EPOCHS) -> tuple[int, int]:
    if row["config"] in STAGE2_ONLY_CONFIGS:
        return epoch_progress(row, epochs), epochs
    if "epoch" in row:
        return epochs + epoch_progress(row, epochs), epochs * 2
    return epoch_progress(row.get("stage1", {}), epochs), epochs * 2


def active_phase(row: dict[str, Any], epochs: int = DEFAULT_EPOCHS) -> str:
    if row["config"] in STAGE2_ONLY_CONFIGS:
        return "stage2"
    if "epoch" in row:
        return "stage2"
    if epoch_progress(row.get("stage1", {}), epochs) >= epochs:
        return "stage2-pending"
    return "stage1"


def progress_counts(rows: list[dict[str, Any]]) -> tuple[int, int, list[str]]:
    done = 0
    total = 0
    active: list[str] = []
    for row in rows:
        row_done, row_total = row_progress(row)
        done += row_done
        total += row_total
        if 0 < row_done < row_total:
            active.append(f"{row['config']}/trial-{row['trial']}/{active_phase(row)}")
    return done, total, active


def parse_start_time(path: Path) -> datetime | None:
    if not path.exists():
        return None
    for line in path.read_text(errors="replace").splitlines():
        match = START_RE.search(line)
        if not match:
            continue
        try:
            return datetime.fromisoformat(match.group("timestamp"))
        except ValueError:
            return None
    return None


def format_duration(delta: timedelta) -> str:
    seconds = max(0, int(delta.total_seconds()))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h{minutes:02d}m"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def print_progress(rows: list[dict[str, Any]], root: Path) -> None:
    done, total, active = progress_counts(rows)
    percent = (100.0 * done / total) if total else 0.0
    active_text = ", ".join(active) if active else "none"
    print(f"overall_progress={done}/{total} epochs ({percent:.1f}%) active={active_text}")

    start = parse_start_time(root / TRAIN_LOG)
    if not start or done <= 0:
        return
    now = datetime.now(start.tzinfo)
    elapsed = now - start
    seconds_per_epoch = elapsed.total_seconds() / done
    remaining = timedelta(seconds=seconds_per_epoch * max(total - done, 0))
    finish = now + remaining
    print(
        "rough_eta="
        f"elapsed={format_duration(elapsed)} "
        f"avg_epoch={format_duration(timedelta(seconds=seconds_per_epoch))} "
        f"remaining={format_duration(remaining)} "
        f"finish={finish.isoformat(timespec='seconds')}"
    )


def print_table(rows: list[dict[str, Any]], root: Path) -> None:
    for row in rows:
        if "epoch" in row:
            print(
                "{config:16s} trial={trial}: stage2 epoch={epoch:>2d} "
                "current={rank1:5.1f}/{map:5.1f} "
                "best={best_rank1:5.1f}/{best_map:5.1f}@{best_epoch} "
                "done={done}".format(**row)
            )
        elif "epoch" in row.get("stage1", {}):
            stage1 = row["stage1"]
            print(
                f"{row['config']:16s} trial={row['trial']}: "
                f"stage1 epoch={stage1['epoch']:>2d} "
                f"current={stage1['rank1']:5.1f}/{stage1['map']:5.1f} "
                f"best={stage1['best_rank1']:5.1f}/{stage1['best_map']:5.1f}@{stage1['best_epoch']} "
                f"stage2={row['status']}"
            )
        elif row.get("stage1", {}).get("status") == "started":
            print(f"{row['config']:16s} trial={row['trial']}: stage1 started stage2={row['status']}")
        else:
            print(f"{row['config']:16s} trial={row['trial']}: {row['status']}")

    print("")
    for config, _, _ in CONFIGS:
        complete = [row for row in rows if row["config"] == config and row.get("done")]
        if not complete:
            print(f"{config:16s} complete=0")
            continue
        print(
            f"{config:16s} complete={len(complete)} "
            f"mean_best={mean(row['best_rank1'] for row in complete):.2f}/"
            f"{mean(row['best_map'] for row in complete):.2f}"
        )
    print_progress(rows, root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--trials", default="1-10")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = collect(args.root, parse_trials(args.trials))
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print_table(rows, args.root)


if __name__ == "__main__":
    main()
