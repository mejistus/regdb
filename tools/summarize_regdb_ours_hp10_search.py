#!/usr/bin/env python3
"""Summarize RegDB 10-trial hyperparameter search for PCLHD+MDUE+CGCF."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


EPOCH_RE = re.compile(
    r"\* Finished epoch\s+(?P<epoch>\d+)\s+"
    r"model R1:\s+(?P<model_r1>[\d.]+)%\s+"
    r"model mAP:\s+(?P<model_map>[\d.]+)%\s+"
    r"best R1:\s+(?P<best_r1>[\d.]+)%\s+"
    r"best mAP:\s+(?P<best_map>[\d.]+)%"
    r"\(best_epoch:(?P<best_epoch>\d+)\)"
)


DEFAULT_CANDIDATES: list[dict[str, Any]] = [
    {
        "tag": "s3_p010",
        "samples": 3,
        "dropout": 0.10,
        "stage1": "regdb_s1_amp10_mdue",
        "stage2": "regdb_s2_amp10_mdue_cgcf",
        "source": "completed_paper_setting",
    },
    {
        "tag": "s3_p005",
        "samples": 3,
        "dropout": 0.05,
        "stage1": "regdb_s1_hp10_s3_p005",
        "stage2": "regdb_s2_hp10_s3_p005_cgcf",
        "source": "strict_search",
    },
    {
        "tag": "s2_p005",
        "samples": 2,
        "dropout": 0.05,
        "stage1": "regdb_s1_hp10_s2_p005",
        "stage2": "regdb_s2_hp10_s2_p005_cgcf",
        "source": "strict_search",
    },
    {
        "tag": "s4_p005",
        "samples": 4,
        "dropout": 0.05,
        "stage1": "regdb_s1_hp10_s4_p005",
        "stage2": "regdb_s2_hp10_s4_p005_cgcf",
        "source": "strict_search",
    },
    {
        "tag": "s2_p010",
        "samples": 2,
        "dropout": 0.10,
        "stage1": "regdb_s1_hp10_s2_p010",
        "stage2": "regdb_s2_hp10_s2_p010_cgcf",
        "source": "strict_search",
    },
]


def parse_trials(value: str) -> list[int]:
    trials: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            trials.extend(range(int(start), int(end) + 1))
        else:
            trials.append(int(chunk))
    return sorted(dict.fromkeys(trials))


def parse_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "complete": False, "epoch": None}
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = list(EPOCH_RE.finditer(text))
    result: dict[str, Any] = {
        "exists": True,
        "complete": "Total running time:" in text,
        "epoch": None,
        "best_rank1": None,
        "best_map": None,
        "best_epoch": None,
        "current_rank1": None,
        "current_map": None,
        "log_path": str(path),
    }
    if matches:
        last = matches[-1].groupdict()
        result.update(
            {
                "epoch": int(last["epoch"]),
                "best_rank1": float(last["best_r1"]),
                "best_map": float(last["best_map"]),
                "best_epoch": int(last["best_epoch"]),
                "current_rank1": float(last["model_r1"]),
                "current_map": float(last["model_map"]),
            }
        )
    return result


def summarize_candidate(root: Path, logs_dir: Path, candidate: dict[str, Any], trials: list[int]) -> dict[str, Any]:
    rows = []
    for trial in trials:
        stage2_log = root / logs_dir / candidate["stage2"] / str(trial) / f"{trial}log.txt"
        stage1_log = root / logs_dir / candidate["stage1"] / str(trial) / f"{trial}log.txt"
        stage2 = parse_log(stage2_log)
        stage1 = parse_log(stage1_log)
        rows.append({"trial": trial, "stage1": stage1, "stage2": stage2})

    complete = [row for row in rows if row["stage2"].get("complete") and row["stage2"].get("best_rank1") is not None]
    rank1_values = [row["stage2"]["best_rank1"] for row in complete]
    map_values = [row["stage2"]["best_map"] for row in complete]
    return {
        **candidate,
        "expected_trials": len(trials),
        "complete_trials": len(complete),
        "mean_best_rank1": mean(rank1_values) if rank1_values else None,
        "mean_best_map": mean(map_values) if map_values else None,
        "max_best_rank1": max(rank1_values) if rank1_values else None,
        "max_best_map": max(map_values) if map_values else None,
        "trials": rows,
    }


def fmt(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:6.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--trials", default="1-10")
    parser.add_argument("--json-out", type=Path, default=Path("logs/regdb_ours_hp10_search_summary.json"))
    args = parser.parse_args()

    root = args.root.resolve()
    trials = parse_trials(args.trials)
    summaries = [summarize_candidate(root, args.logs, candidate, trials) for candidate in DEFAULT_CANDIDATES]
    summaries_sorted = sorted(
        summaries,
        key=lambda row: (
            row["complete_trials"] == row["expected_trials"],
            row["mean_best_rank1"] if row["mean_best_rank1"] is not None else -1,
            row["mean_best_map"] if row["mean_best_map"] is not None else -1,
        ),
        reverse=True,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summaries_sorted, indent=2), encoding="utf-8")

    print("RegDB PCLHD+MDUE+CGCF HP10 search")
    print("metric=10-trial mean best Rank-1; tie_break=mean best mAP")
    print(f"json={args.json_out}")
    print("rank tag      S    p     done  mean_R1 mean_mAP max_R1  source")
    for index, row in enumerate(summaries_sorted, 1):
        print(
            f"{index:>4} {row['tag']:<8} {row['samples']:<4} {row['dropout']:<5.2f} "
            f"{row['complete_trials']:>2}/{row['expected_trials']:<2} "
            f"{fmt(row['mean_best_rank1'])} {fmt(row['mean_best_map'])} {fmt(row['max_best_rank1'])}  {row['source']}"
        )
        active = [
            t
            for t in row["trials"]
            if t["stage2"].get("exists") and not t["stage2"].get("complete")
        ]
        if active:
            latest = active[-1]
            s2 = latest["stage2"]
            print(
                f"     active trial={latest['trial']} epoch={s2.get('epoch')} "
                f"current={fmt(s2.get('current_rank1')).strip()}/{fmt(s2.get('current_map')).strip()} "
                f"best={fmt(s2.get('best_rank1')).strip()}/{fmt(s2.get('best_map')).strip()}@{s2.get('best_epoch')}"
            )


if __name__ == "__main__":
    main()
