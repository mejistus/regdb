#!/usr/bin/env python3
"""Summarize paper-parameter AMP ablation logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


CONFIGS = [
    ("baseline", "regdb_s2_amp_baseline"),
    ("mdue", "regdb_s2_amp_mdue"),
    ("mdue_cgcf", "regdb_s2_amp_mdue_cgcf"),
]

EPOCH_RE = re.compile(
    r"Finished epoch\s+(?P<epoch>\d+)\s+"
    r"model R1:\s*(?P<rank1>[0-9.]+)%\s+"
    r"model mAP:\s*(?P<map>[0-9.]+)%\s+"
    r"best R1:\s*(?P<best_rank1>[0-9.]+)%\s+"
    r"best mAP:\s*(?P<best_map>[0-9.]+)%"
    r"\(best_epoch:(?P<best_epoch>\d+)\)"
)


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
    for config, folder in CONFIGS:
        for trial in trials:
            path = root / "logs" / folder / str(trial) / f"{trial}log.txt"
            row = parse_log(path)
            row.update({"config": config, "folder": folder, "trial": trial})
            rows.append(row)
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if "epoch" in row:
            print(
                "{config:10s} trial={trial}: epoch={epoch:>2d} "
                "current={rank1:5.1f}/{map:5.1f} "
                "best={best_rank1:5.1f}/{best_map:5.1f}@{best_epoch} "
                "done={done}".format(**row)
            )
        else:
            print(f"{row['config']:10s} trial={row['trial']}: {row['status']}")

    print("")
    for config, _ in CONFIGS:
        complete = [row for row in rows if row["config"] == config and row.get("done")]
        if not complete:
            print(f"{config:10s} complete=0")
            continue
        print(
            f"{config:10s} complete={len(complete)} "
            f"mean_best={mean(row['best_rank1'] for row in complete):.2f}/"
            f"{mean(row['best_map'] for row in complete):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--trials", default="1-3")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = collect(args.root, parse_trials(args.trials))
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
