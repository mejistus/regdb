#!/usr/bin/env python3
"""Build a self-contained RegDB training metrics report."""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime
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
FC_RE = re.compile(
    r"FC:\s+Rank-1:\s+(?P<rank1>[\d.]+)%\s*\|\s*"
    r"Rank-5:\s+(?P<rank5>[\d.]+)%\s*\|\s*"
    r"Rank-10:\s+(?P<rank10>[\d.]+)%\s*\|\s*"
    r"Rank-20:\s+(?P<rank20>[\d.]+)%\s*\|\s*"
    r"mAP:\s+(?P<map>[\d.]+)%\s*\|\s*"
    r"mINP:\s+(?P<minp>[\d.]+)%"
)
ASSOC_RE = re.compile(r"associate rate\s+(?P<rate>[\d.eE+-]+)")
ARG_RE = re.compile(r"(?P<key>\w+)=(?P<value>'[^']*'|\"[^\"]*\"|[^,\)]+)")
TEST_AVG_RE = re.compile(
    r"FC:\s+Rank-1:\s+(?P<rank1>[\d.]+)%.*?"
    r"mAP:\s+(?P<map>[\d.]+)%.*?"
    r"mINP:\s+(?P<minp>[\d.]+)%"
)


COMPARISON_ROWS: list[dict[str, Any]] = [
    {"type": "SVI-ReID", "method": "DDAG", "sysu_all_rank1": 54.75, "sysu_all_map": 53.02, "sysu_indoor_rank1": 61.02, "sysu_indoor_map": 67.98, "regdb_v2t_rank1": 69.34, "regdb_v2t_map": 63.46, "regdb_t2v_rank1": 68.06, "regdb_t2v_map": 61.80, "source": "paper"},
    {"type": "SVI-ReID", "method": "AGW", "sysu_all_rank1": 47.50, "sysu_all_map": 47.65, "sysu_indoor_rank1": 54.17, "sysu_indoor_map": 62.97, "regdb_v2t_rank1": 70.05, "regdb_v2t_map": 66.37, "regdb_t2v_rank1": 70.49, "regdb_t2v_map": 65.90, "source": "paper"},
    {"type": "SVI-ReID", "method": "NFS", "sysu_all_rank1": 56.91, "sysu_all_map": 55.45, "sysu_indoor_rank1": 62.79, "sysu_indoor_map": 69.79, "regdb_v2t_rank1": 80.54, "regdb_v2t_map": 72.10, "regdb_t2v_rank1": 77.95, "regdb_t2v_map": 69.79, "source": "paper"},
    {"type": "SVI-ReID", "method": "LbA", "sysu_all_rank1": 55.41, "sysu_all_map": 54.14, "sysu_indoor_rank1": 58.46, "sysu_indoor_map": 66.33, "regdb_v2t_rank1": 74.17, "regdb_v2t_map": 67.64, "regdb_t2v_rank1": 72.43, "regdb_t2v_map": 65.46, "source": "paper"},
    {"type": "SVI-ReID", "method": "CAJ", "sysu_all_rank1": 69.88, "sysu_all_map": 66.89, "sysu_indoor_rank1": 76.26, "sysu_indoor_map": 80.37, "regdb_v2t_rank1": 85.03, "regdb_v2t_map": 79.14, "regdb_t2v_rank1": 84.75, "regdb_t2v_map": 77.82, "source": "paper"},
    {"type": "SVI-ReID", "method": "MPANet", "sysu_all_rank1": 70.58, "sysu_all_map": 68.24, "sysu_indoor_rank1": 76.74, "sysu_indoor_map": 80.95, "regdb_v2t_rank1": 83.70, "regdb_v2t_map": 80.90, "regdb_t2v_rank1": 82.80, "regdb_t2v_map": 80.70, "source": "paper"},
    {"type": "SVI-ReID", "method": "DART", "sysu_all_rank1": 68.72, "sysu_all_map": 66.29, "sysu_indoor_rank1": 72.52, "sysu_indoor_map": 78.17, "regdb_v2t_rank1": 83.60, "regdb_v2t_map": 75.70, "regdb_t2v_rank1": 81.97, "regdb_t2v_map": 73.78, "source": "paper"},
    {"type": "SVI-ReID", "method": "FMCNet", "sysu_all_rank1": 66.34, "sysu_all_map": 62.51, "sysu_indoor_rank1": 68.15, "sysu_indoor_map": 74.09, "regdb_v2t_rank1": 89.12, "regdb_v2t_map": 84.43, "regdb_t2v_rank1": 88.38, "regdb_t2v_map": 83.86, "source": "paper"},
    {"type": "SVI-ReID", "method": "MID", "sysu_all_rank1": 60.27, "sysu_all_map": 59.40, "sysu_indoor_rank1": 64.86, "sysu_indoor_map": 70.12, "regdb_v2t_rank1": 87.45, "regdb_v2t_map": 84.85, "regdb_t2v_rank1": 84.29, "regdb_t2v_map": 81.41, "source": "paper"},
    {"type": "SVI-ReID", "method": "LUPI", "sysu_all_rank1": 71.75, "sysu_all_map": 67.12, "sysu_indoor_rank1": 78.59, "sysu_indoor_map": 81.17, "regdb_v2t_rank1": 81.61, "regdb_v2t_map": 77.01, "regdb_t2v_rank1": 82.30, "regdb_t2v_map": 76.22, "source": "paper"},
    {"type": "SVI-ReID", "method": "DEEN", "sysu_all_rank1": 74.70, "sysu_all_map": 71.80, "sysu_indoor_rank1": 80.30, "sysu_indoor_map": 83.30, "regdb_v2t_rank1": 91.10, "regdb_v2t_map": 85.10, "regdb_t2v_rank1": 89.50, "regdb_t2v_map": 83.40, "source": "paper"},
    {"type": "SVI-ReID", "method": "SGIEL", "sysu_all_rank1": 77.12, "sysu_all_map": 72.33, "sysu_indoor_rank1": 82.23, "sysu_indoor_map": 84.92, "regdb_v2t_rank1": 88.40, "regdb_v2t_map": 81.89, "regdb_t2v_rank1": 86.60, "regdb_t2v_map": 79.60, "source": "paper"},
    {"type": "SSVI-ReID", "method": "PartMix", "sysu_all_rank1": 77.78, "sysu_all_map": 74.62, "sysu_indoor_rank1": 81.52, "sysu_indoor_map": 84.38, "regdb_v2t_rank1": 85.66, "regdb_v2t_map": 82.27, "regdb_t2v_rank1": 84.93, "regdb_t2v_map": 81.07, "source": "paper"},
    {"type": "SSVI-ReID", "method": "CAL", "sysu_all_rank1": 74.17, "sysu_all_map": 69.94, "sysu_indoor_rank1": 79.86, "sysu_indoor_map": 83.68, "regdb_v2t_rank1": 83.46, "regdb_v2t_map": 79.36, "regdb_t2v_rank1": 83.68, "regdb_t2v_map": 78.88, "source": "paper"},
    {"type": "SSVI-ReID", "method": "MUN", "sysu_all_rank1": 76.24, "sysu_all_map": 73.81, "sysu_indoor_rank1": 79.42, "sysu_indoor_map": 82.64, "regdb_v2t_rank1": 86.95, "regdb_v2t_map": 83.14, "regdb_t2v_rank1": 84.59, "regdb_t2v_map": 79.42, "source": "paper"},
    {"type": "SSVI-ReID", "method": "SAAI", "sysu_all_rank1": 75.90, "sysu_all_map": 77.03, "sysu_indoor_rank1": 83.20, "sysu_indoor_map": 88.01, "regdb_v2t_rank1": 89.85, "regdb_v2t_map": 83.10, "regdb_t2v_rank1": 88.01, "regdb_t2v_map": 79.43, "source": "paper"},
    {"type": "SSVI-ReID", "method": "FDNM", "sysu_all_rank1": 77.95, "sysu_all_map": 75.59, "sysu_indoor_rank1": 83.95, "sysu_indoor_map": 87.67, "regdb_v2t_rank1": 83.91, "regdb_v2t_map": 80.76, "regdb_t2v_rank1": 85.67, "regdb_t2v_map": 80.98, "source": "paper"},
    {"type": "SSVI-ReID", "method": "PMWGCN", "sysu_all_rank1": 78.96, "sysu_all_map": 75.81, "sysu_indoor_rank1": 86.85, "sysu_indoor_map": 89.61, "regdb_v2t_rank1": 89.34, "regdb_v2t_map": 84.16, "regdb_t2v_rank1": 87.52, "regdb_t2v_map": 81.39, "source": "paper"},
    {"type": "SSVI-ReID", "method": "LCNL", "sysu_all_rank1": 79.67, "sysu_all_map": 76.17, "sysu_indoor_rank1": 84.98, "sysu_indoor_map": 88.81, "regdb_v2t_rank1": 88.01, "regdb_v2t_map": 82.66, "regdb_t2v_rank1": 84.68, "regdb_t2v_map": 79.44, "source": "paper"},
    {"type": "SSVI-ReID", "method": "OTLA", "sysu_all_rank1": 78.11, "sysu_all_map": 75.73, "sysu_indoor_rank1": 81.63, "sysu_indoor_map": 85.11, "regdb_v2t_rank1": 88.54, "regdb_v2t_map": 83.61, "regdb_t2v_rank1": 85.48, "regdb_t2v_map": 80.02, "source": "paper"},
    {"type": "SSVI-ReID", "method": "TAA", "sysu_all_rank1": 78.97, "sysu_all_map": 75.63, "sysu_indoor_rank1": 83.39, "sysu_indoor_map": 87.58, "regdb_v2t_rank1": 88.19, "regdb_v2t_map": 82.54, "regdb_t2v_rank1": 87.65, "regdb_t2v_map": 81.96, "source": "paper"},
    {"type": "SSVI-ReID", "method": "DPIS", "sysu_all_rank1": 87.49, "sysu_all_map": 81.29, "sysu_indoor_rank1": 90.29, "sysu_indoor_map": 88.48, "regdb_v2t_rank1": 93.03, "regdb_v2t_map": 87.78, "regdb_t2v_rank1": 91.66, "regdb_t2v_map": 86.69, "source": "paper"},
    {"type": "USVI-ReID", "method": "H2H", "sysu_all_rank1": 30.15, "sysu_all_map": 29.40, "sysu_indoor_rank1": None, "sysu_indoor_map": None, "regdb_v2t_rank1": 23.81, "regdb_v2t_map": 18.87, "regdb_t2v_rank1": None, "regdb_t2v_map": None, "source": "paper"},
    {"type": "USVI-ReID", "method": "ADCA", "sysu_all_rank1": 45.51, "sysu_all_map": 42.73, "sysu_indoor_rank1": 50.60, "sysu_indoor_map": 59.11, "regdb_v2t_rank1": 67.20, "regdb_v2t_map": 64.05, "regdb_t2v_rank1": 68.48, "regdb_t2v_map": 63.81, "source": "paper"},
    {"type": "USVI-ReID", "method": "NGLR", "sysu_all_rank1": 56.15, "sysu_all_map": 55.10, "sysu_indoor_rank1": 66.12, "sysu_indoor_map": 72.56, "regdb_v2t_rank1": 85.65, "regdb_v2t_map": 78.88, "regdb_t2v_rank1": 82.17, "regdb_t2v_map": 76.01, "source": "paper"},
    {"type": "USVI-ReID", "method": "MBCCM", "sysu_all_rank1": 53.14, "sysu_all_map": 52.13, "sysu_indoor_rank1": 55.91, "sysu_indoor_map": 66.33, "regdb_v2t_rank1": 83.79, "regdb_v2t_map": 78.27, "regdb_t2v_rank1": 82.33, "regdb_t2v_map": 77.87, "source": "paper"},
    {"type": "USVI-ReID", "method": "CCLNet", "sysu_all_rank1": 54.03, "sysu_all_map": 50.19, "sysu_indoor_rank1": 56.68, "sysu_indoor_map": 65.53, "regdb_v2t_rank1": 69.94, "regdb_v2t_map": 65.53, "regdb_t2v_rank1": 70.17, "regdb_t2v_map": 65.13, "source": "paper"},
    {"type": "USVI-ReID", "method": "PGM", "sysu_all_rank1": 57.27, "sysu_all_map": 51.78, "sysu_indoor_rank1": 56.23, "sysu_indoor_map": 62.74, "regdb_v2t_rank1": 69.53, "regdb_v2t_map": 65.41, "regdb_t2v_rank1": 69.95, "regdb_t2v_map": 65.17, "source": "paper"},
    {"type": "USVI-ReID", "method": "GUR*", "sysu_all_rank1": 64.05, "sysu_all_map": 61.30, "sysu_indoor_rank1": 68.79, "sysu_indoor_map": 76.94, "regdb_v2t_rank1": 73.91, "regdb_v2t_map": 70.23, "regdb_t2v_rank1": 75.00, "regdb_t2v_map": 69.90, "source": "paper"},
    {"type": "USVI-ReID", "method": "MMM", "sysu_all_rank1": 68.91, "sysu_all_map": 65.46, "sysu_indoor_rank1": 75.37, "sysu_indoor_map": 79.85, "regdb_v2t_rank1": 84.50, "regdb_v2t_map": 80.82, "regdb_t2v_rank1": 84.72, "regdb_t2v_map": 79.94, "source": "paper"},
    {"type": "USVI-ReID", "method": "PCLHD", "sysu_all_rank1": 72.40, "sysu_all_map": 67.52, "sysu_indoor_rank1": 77.39, "sysu_indoor_map": 81.64, "regdb_v2t_rank1": 85.78, "regdb_v2t_map": 82.42, "regdb_t2v_rank1": 84.56, "regdb_t2v_map": 80.12, "source": "paper"},
    {"type": "USVI-ReID", "method": "PCLHD+MMM", "sysu_all_rank1": 75.52, "sysu_all_map": 69.18, "sysu_indoor_rank1": 79.29, "sysu_indoor_map": 81.96, "regdb_v2t_rank1": 87.73, "regdb_v2t_map": 82.31, "regdb_t2v_rank1": 86.18, "regdb_t2v_map": 80.60, "source": "paper"},
]


def parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw in {"True", "False"}:
        return raw == "True"
    if raw == "None":
        return None
    if (raw.startswith("'") and raw.endswith("'")) or (
        raw.startswith('"') and raw.endswith('"')
    ):
        return raw[1:-1]
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_args_line(line: str) -> dict[str, Any]:
    if "Args:Namespace(" not in line:
        return {}
    content = line.split("Args:Namespace(", 1)[1].rsplit(")", 1)[0]
    return {m.group("key"): parse_value(m.group("value")) for m in ARG_RE.finditer(content)}


def parse_log(path: Path, trial: int, stage: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    args: dict[str, Any] = {}
    epochs: list[dict[str, Any]] = []
    pending_fc: dict[str, float] | None = None
    pending_associate: float | None = None
    runtime: str | None = None
    errors: list[str] = []

    if not path.exists():
        return (
            {
                "trial": trial,
                "stage": stage,
                "log_path": str(path),
                "exists": False,
                "complete": False,
                "status": "missing",
                "errors": ["missing log file"],
            },
            [],
        )

    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("Args:Namespace("):
            args = parse_args_line(line)
            continue
        if "Traceback" in line or "OutOfMemory" in line or "RuntimeError:" in line:
            errors.append(line.strip())
        match = FC_RE.search(line)
        if match:
            pending_fc = {key: float(value) for key, value in match.groupdict().items()}
            continue
        match = ASSOC_RE.search(line)
        if match:
            pending_associate = float(match.group("rate"))
            continue
        match = EPOCH_RE.search(line)
        if match:
            row: dict[str, Any] = {
                "trial": trial,
                "stage": stage,
                "epoch": int(match.group("epoch")),
                "model_r1": float(match.group("model_r1")),
                "model_map": float(match.group("model_map")),
                "best_r1": float(match.group("best_r1")),
                "best_map": float(match.group("best_map")),
                "best_epoch": int(match.group("best_epoch")),
                "associate_rate": pending_associate,
                "log_path": str(path),
            }
            if pending_fc:
                row.update(pending_fc)
            epochs.append(row)
            pending_fc = None
            pending_associate = None
            continue
        if line.startswith("Total running time:"):
            runtime = line.split(":", 1)[1].strip()

    final_epoch = epochs[-1] if epochs else {}
    checkpoint_epoch_number = final_epoch.get("best_epoch")
    checkpoint_epoch = next(
        (row for row in epochs if row["epoch"] == checkpoint_epoch_number), final_epoch
    )
    best_rank1_epoch = max(
        epochs,
        key=lambda row: row_metric(row, "rank1", "model_r1") or -1.0,
        default={},
    )
    best_map_epoch = max(
        epochs,
        key=lambda row: row_metric(row, "map", "model_map") or -1.0,
        default={},
    )
    complete = runtime is not None and len(epochs) >= int(args.get("epochs", 50) or 50)
    status = "complete" if complete else ("running_or_incomplete" if epochs else "empty")
    checkpoint = path.parent / "model_best.pth.tar"

    summary = {
        "trial": trial,
        "stage": stage,
        "log_path": str(path),
        "exists": True,
        "status": status,
        "complete": complete,
        "epoch_count": len(epochs),
        "runtime": runtime,
        "batch_size": args.get("batch_size"),
        "stage2_batch_size": args.get("stage2_batch_size"),
        "seed": args.get("seed"),
        "data_dir": args.get("data_dir"),
        "logs_dir": args.get("logs_dir"),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "best_epoch": best_rank1_epoch.get("epoch"),
        "best_rank1": row_metric(best_rank1_epoch, "rank1", "model_r1"),
        "best_rank5": best_rank1_epoch.get("rank5"),
        "best_rank10": best_rank1_epoch.get("rank10"),
        "best_rank20": best_rank1_epoch.get("rank20"),
        "best_map": row_metric(best_map_epoch, "map", "model_map"),
        "best_map_epoch": best_map_epoch.get("epoch"),
        "best_minp": best_rank1_epoch.get("minp"),
        "checkpoint_epoch": checkpoint_epoch_number,
        "checkpoint_rank1": row_metric(checkpoint_epoch, "rank1", "model_r1"),
        "checkpoint_map": row_metric(checkpoint_epoch, "map", "model_map"),
        "final_epoch": final_epoch.get("epoch"),
        "final_rank1": final_epoch.get("rank1", final_epoch.get("model_r1")),
        "final_rank5": final_epoch.get("rank5"),
        "final_rank10": final_epoch.get("rank10"),
        "final_rank20": final_epoch.get("rank20"),
        "final_map": final_epoch.get("map", final_epoch.get("model_map")),
        "final_minp": final_epoch.get("minp"),
        "final_associate_rate": final_epoch.get("associate_rate"),
        "errors": errors[-5:],
    }
    return summary, epochs


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def row_metric(row: dict[str, Any], primary: str, fallback: str) -> float | None:
    value = row.get(primary)
    if isinstance(value, (int, float)):
        return float(value)
    value = row.get(fallback)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def parse_regdb_bidir_test(path: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    if not path.exists():
        return results

    pending_average = False
    directions = ["regdb_v2t", "regdb_t2v"]
    direction_index = 0
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("All Average:"):
            pending_average = True
            continue
        if not pending_average:
            continue
        match = TEST_AVG_RE.search(line)
        if not match:
            continue
        if direction_index < len(directions):
            results[directions[direction_index]] = {
                "rank1": float(match.group("rank1")),
                "map": float(match.group("map")),
                "minp": float(match.group("minp")),
            }
        direction_index += 1
        pending_average = False
    return results


def build_comparison_rows(log_root: Path) -> list[dict[str, Any]]:
    rows = [dict(row) for row in COMPARISON_ROWS]
    bidir = parse_regdb_bidir_test(log_root / "regdb_s2_test_bidir.log")
    v2t = bidir.get("regdb_v2t", {})
    t2v = bidir.get("regdb_t2v", {})
    rows.append(
        {
            "type": "USVI-ReID",
            "method": "PCLHD reproduced, stage2 avg",
            "sysu_all_rank1": None,
            "sysu_all_map": None,
            "sysu_indoor_rank1": None,
            "sysu_indoor_map": None,
            "regdb_v2t_rank1": v2t.get("rank1"),
            "regdb_v2t_map": v2t.get("map"),
            "regdb_t2v_rank1": t2v.get("rank1"),
            "regdb_t2v_map": t2v.get("map"),
            "source": "reproduced",
        }
    )
    return rows


def collect(root: Path, log_root: Path, trials: list[int]) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    missing_logs: list[str] = []
    incomplete_runs: list[str] = []

    for trial in trials:
        for stage, folder in (("stage1", "regdb_s1"), ("stage2", "regdb_s2")):
            log_path = log_root / folder / str(trial) / f"{trial}log.txt"
            summary, epochs = parse_log(log_path, trial, stage)
            summary_rows.append(summary)
            epoch_rows.extend(epochs)
            if not summary["exists"]:
                missing_logs.append(str(log_path))
            elif not summary["complete"]:
                incomplete_runs.append(f"{stage}/trial-{trial}")

    stage2_complete = [
        row for row in summary_rows if row["stage"] == "stage2" and row["complete"]
    ]
    stage2_best_r1 = finite_values(stage2_complete, "best_rank1")
    stage2_best_map = finite_values(stage2_complete, "best_map")
    expected_run_count = len(trials) * 2
    validation = {
        "expected_trials": trials,
        "expected_run_count": expected_run_count,
        "actual_run_count": len(summary_rows),
        "epoch_row_count": len(epoch_rows),
        "missing_logs": missing_logs,
        "incomplete_runs": incomplete_runs,
        "complete_run_count": sum(1 for row in summary_rows if row["complete"]),
        "complete_stage2_count": len(stage2_complete),
        "stage2_best_rank1_mean": mean(stage2_best_r1) if stage2_best_r1 else None,
        "stage2_best_map_mean": mean(stage2_best_map) if stage2_best_map else None,
        "stage2_best_rank1_max": max(stage2_best_r1) if stage2_best_r1 else None,
        "stage2_best_map_max": max(stage2_best_map) if stage2_best_map else None,
    }
    validation["ok"] = (
        validation["actual_run_count"] == expected_run_count
        and not missing_logs
        and not incomplete_runs
    )
    return {
        "manifest": {
            "script": "tools/build_regdb_stats.py",
            "repo_path": str(root),
            "log_root": str(log_root),
            "output": "htmls/stats.html",
            "metric_policy": "Rank and mAP values are parsed from FC evaluation lines; best Rank-1 and max mAP are computed independently. Checkpoint fields use the logged best epoch.",
            "split_policy": "RegDB trials 1-10. Training logs are visible-to-thermal; the comparison table also parses logs/regdb_s2_test_bidir.log when available.",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "comparison_source": "Literature rows are transcribed from arXiv:2402.19026v3 Table 1. The reproduced RegDB row is parsed from logs/regdb_s2_test_bidir.log.",
        },
        "summary_rows": summary_rows,
        "epoch_rows": epoch_rows,
        "comparison_rows": build_comparison_rows(log_root),
        "validation": validation,
    }


def build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    validation = payload["validation"]
    status_text = "complete" if validation["ok"] else "incomplete"
    generated_at = html.escape(payload["manifest"]["generated_at"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RegDB Training Stats</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d9dee8;
      --accent: #2563eb;
      --bad: #b42318;
      --ok: #067647;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }}
    header, main {{ max-width: 1480px; margin: 0 auto; padding: 20px 24px; }}
    header {{ padding-bottom: 8px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; font-weight: 760; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; font-weight: 700; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 16px 0; }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgb(16 24 40 / 5%);
    }}
    .card {{ padding: 14px 16px; }}
    .card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .card .value {{ margin-top: 4px; font-size: 24px; font-weight: 760; font-variant-numeric: tabular-nums; }}
    .card .note {{ margin-top: 2px; color: var(--muted); font-size: 12px; }}
    .panel {{ margin: 16px 0; padding: 14px; overflow: hidden; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin-bottom: 12px; }}
    label {{ display: grid; gap: 5px; color: var(--muted); font-size: 12px; font-weight: 650; }}
    select, input {{
      min-width: 160px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 7px 9px;
      font: inherit;
    }}
    select[multiple] {{ min-height: 74px; padding: 4px; }}
    input {{ min-width: 280px; }}
    button {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 8px 11px;
      font: inherit;
      cursor: pointer;
    }}
    button:hover {{ border-color: var(--accent); color: var(--accent); }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 6px; }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0; min-width: 1080px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 9px; vertical-align: middle; white-space: nowrap; }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f8fafc;
      color: #344054;
      font-size: 12px;
      text-align: left;
      font-weight: 720;
      cursor: pointer;
    }}
    th[data-key]::after {{ content: "  sort"; color: #98a2b3; font-weight: 520; }}
    th.active.asc::after {{ content: "  asc"; color: var(--accent); }}
    th.active.desc::after {{ content: "  desc"; color: var(--accent); }}
    tbody tr:hover td {{ background-color: #f8fbff; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 12px; }}
    .status {{ display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; font-weight: 700; }}
    .status.complete {{ color: var(--ok); background: #ecfdf3; }}
    .status.incomplete {{ color: var(--bad); background: #fef3f2; }}
    .muted {{ color: var(--muted); }}
    .explain-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 4px;
    }}
    .explain-block {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfe;
    }}
    .explain-block h3 {{
      margin: 0 0 8px;
      font-size: 14px;
      font-weight: 740;
      letter-spacing: 0;
    }}
    .explain-block p {{ margin: 0 0 8px; }}
    .explain-block p:last-child {{ margin-bottom: 0; }}
    .term-list {{
      display: grid;
      grid-template-columns: 150px 1fr;
      gap: 8px 12px;
      margin: 0;
    }}
    .term-list dt {{
      color: var(--ink);
      font-weight: 720;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
    }}
    .term-list dd {{
      margin: 0;
      color: var(--muted);
    }}
    .flow-steps {{
      display: grid;
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      gap: 10px;
      margin-top: 12px;
      padding: 0;
      list-style: none;
    }}
    .flow-steps li {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      padding: 10px;
    }}
    .flow-steps strong {{
      display: block;
      margin-bottom: 5px;
      color: var(--ink);
    }}
    .flow-steps span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .caption-title {{
      margin: 4px 0 0;
      color: var(--ink);
      font-weight: 720;
      text-align: center;
    }}
    .caption-en {{
      margin: 2px 0 12px;
      text-align: center;
      font-size: 13px;
    }}
    .source-note {{
      margin: 10px 0 0;
      font-size: 12px;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      color: #344054;
      background: #eef2f7;
      border-radius: 4px;
      padding: 1px 4px;
    }}
    .validation {{ display: grid; gap: 4px; color: var(--muted); }}
    .validation strong {{ color: var(--ink); }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .explain-grid {{ grid-template-columns: 1fr; }}
      .flow-steps {{ grid-template-columns: 1fr; }}
      .term-list {{ grid-template-columns: 1fr; }}
      input, select {{ min-width: 100%; }}
      label {{ flex: 1 1 180px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>RegDB Training Stats</h1>
    <p>Generated at {generated_at}. Report status: <strong>{status_text}</strong>.</p>
  </header>
  <main>
    <section class="cards" id="cards"></section>
    <section class="panel">
      <h2>项目与方法说明</h2>
      <div class="explain-grid">
        <div class="explain-block">
          <h3>这个项目做什么</h3>
          <p>本项目复现的是无监督可见光-红外行人重识别。输入是一组可见光图像和一组红外图像，训练时不使用人工身份标签，目标是学习同一个人在两种模态下都接近的特征表示。</p>
          <p>评测时用一个模态作为查询，另一个模态作为图库，按特征距离检索同一身份。当前 RegDB 脚本记录的是 visible-to-thermal 方向。</p>
        </div>
        <div class="explain-block">
          <h3>为什么有 10 个 trial</h3>
          <p>RegDB 官方提供 10 组划分。每个 <code>trial</code> 都是一套不同的训练/测试拆分，论文复现实验通常报告 10 个 trial 的平均结果，以降低单次划分带来的偶然性。</p>
          <p>因此完整复现不是只跑一次，而是 trial 1 到 10 都跑完。</p>
        </div>
        <div class="explain-block">
          <h3>为什么有 stage1 和 stage2</h3>
          <p><code>stage1</code> 是初始无监督聚类训练：分别对 RGB/IR 特征做聚类，把聚类编号当作伪标签训练模型，得到可用的初始 checkpoint。</p>
          <p><code>stage2</code> 从 stage1 checkpoint 继续训练，加入跨模态关联和二分图匹配，把 RGB 与 IR 中可能属于同一人的聚类对应起来，重点优化跨模态检索性能。</p>
        </div>
        <div class="explain-block">
          <h3>本次复现设置</h3>
          <p>环境使用 Python 3.12、PyTorch 2.6.0、单张 Tesla V100 16GB。由于 batch 64 在后续 trial 出现显存边缘 OOM，剩余 trial 已改为 batch 32 续跑；表格中的 <code>Batch</code> 列会记录每个 run 的实际 batch size。</p>
          <p>最终报告以完成的 stage2 结果为主；当前报告如果显示 incomplete，表示还有 trial 尚未全部完成。</p>
        </div>
      </div>
    </section>
    <section class="panel">
      <h2>算法流程</h2>
      <p>本项目把无监督跨模态 ReID 拆成“单模态伪标签初始化”和“跨模态关联优化”两段。训练过程中每个 epoch 会重新抽取特征、聚类、构造 memory，并用伪标签和跨模态匹配关系更新网络。</p>
      <ol class="flow-steps">
        <li><strong>1. 数据准备</strong><span>读取 RegDB/SYSU 的 RGB 与 IR 图像，按官方划分生成训练集、查询集和图库。</span></li>
        <li><strong>2. 特征初始化</strong><span>使用 ImageNet 预训练 backbone 提取两种模态的初始特征，并用 GeM/BNNeck 形成检索向量。</span></li>
        <li><strong>3. Stage1 聚类训练</strong><span>分别对 RGB 与 IR 特征做 DBSCAN 聚类，把聚类 ID 当作伪标签，用 DCL 和 ClusterMemory 训练初始模型。</span></li>
        <li><strong>4. Stage2 跨模态关联</strong><span>加载 stage1 checkpoint，构造 RGB、IR、All 三组 memory，通过跨模态相似度和二分图匹配建立 RGB-IR 原型对应关系。</span></li>
        <li><strong>5. 评估与汇总</strong><span>用最终 stage2 checkpoint 做 visible-to-thermal 与 thermal-to-visible 检索，汇总 Rank-1、mAP、mINP 和逐 trial 均值。</span></li>
      </ol>
    </section>
    <section class="panel">
      <h2>对比实验表格</h2>
      <div class="controls">
        <label>Types<select id="compareTypeFilter" multiple></select></label>
        <label>Search<input id="compareSearchInput" type="search" placeholder="method, type, source"></label>
        <button id="clearCompareFilters" type="button">Clear</button>
      </div>
      <p class="caption-title">表 5.2 不同方法在 SYSU-MM01 数据集上的定量对比</p>
      <p class="caption-en">Table 5.2 Quantitative experimental results of different methods on SYSU-MM01 dataset</p>
      <div class="table-wrap">
        <table id="sysuCompareTable">
          <thead>
            <tr>
              <th data-key="type" rowspan="2">类型</th>
              <th data-key="method" rowspan="2">方法</th>
              <th colspan="2">全搜索</th>
              <th colspan="2">室内搜索</th>
            </tr>
            <tr>
              <th data-key="sysu_all_rank1">Rank-1 (%)</th>
              <th data-key="sysu_all_map">mAP (%)</th>
              <th data-key="sysu_indoor_rank1">Rank-1 (%)</th>
              <th data-key="sysu_indoor_map">mAP (%)</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
      <p class="caption-title">表 5.3 不同方法在 RegDB 数据集上的定量对比</p>
      <p class="caption-en">Table 5.3 Quantitative experimental results of different methods on RegDB dataset</p>
      <div class="table-wrap">
        <table id="regdbCompareTable">
          <thead>
            <tr>
              <th data-key="type" rowspan="2">类型</th>
              <th data-key="method" rowspan="2">方法</th>
              <th colspan="2">可见光-红外</th>
              <th colspan="2">红外-可见光</th>
            </tr>
            <tr>
              <th data-key="regdb_v2t_rank1">Rank-1 (%)</th>
              <th data-key="regdb_v2t_map">mAP (%)</th>
              <th data-key="regdb_t2v_rank1">Rank-1 (%)</th>
              <th data-key="regdb_t2v_map">mAP (%)</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
      <p class="source-note muted">文献行来自 arXiv:2402.19026v3 Table 1；“PCLHD reproduced, stage2 avg”来自本次复现的 <code>logs/regdb_s2_test_bidir.log</code>。SYSU-MM01 未在本次任务中重新训练，因此复现行的 SYSU 字段留空。</p>
    </section>
    <section class="panel">
      <h2>术语解释</h2>
      <dl class="term-list">
        <dt>run</dt><dd>一次具体训练任务。完整 RegDB 复现包含 10 个 trial，每个 trial 有 stage1 和 stage2，所以一共 20 个 run。</dd>
        <dt>epoch</dt><dd>完整遍历一次当前训练采样流程后的评估点。本报告逐 epoch 记录 Rank 和 mAP，便于观察训练走势。</dd>
        <dt>checkpoint</dt><dd>训练过程中保存的最佳模型权重。本报告的 Ckpt 字段对应日志里记录的 best epoch。</dd>
        <dt>Rank-1 / R1</dt><dd>查询图像检索结果的第一名就是正确身份的比例。越高越好，是行人重识别最直观的指标。</dd>
        <dt>Rank-5/10/20</dt><dd>正确身份出现在前 5、10、20 个检索结果中的比例。它们衡量较宽松的召回能力。</dd>
        <dt>mAP</dt><dd>mean Average Precision，综合衡量排序列表中所有正确匹配的位置，越高说明整体排序越好。</dd>
        <dt>mINP</dt><dd>mean Inverse Negative Penalty，更关注最后一个正确匹配在排序中的位置，反映困难样本检索质量。</dd>
        <dt>Assoc</dt><dd>stage2 中 RGB/IR 聚类成功建立跨模态关联的比例。它用于观察跨模态匹配是否逐步稳定。</dd>
        <dt>Best R1</dt><dd>该 run 中 Rank-1 最高的 epoch，对应主要 checkpoint 选择依据。</dd>
        <dt>Best mAP</dt><dd>该 run 中 mAP 的最高值。本报告把 Best R1 和 Best mAP 分开计算，因为最高 Rank-1 和最高 mAP 不一定发生在同一个 epoch。</dd>
      </dl>
    </section>
    <section class="panel">
      <h2>Filters</h2>
      <div class="controls">
        <label>Trials<select id="trialFilter" multiple></select></label>
        <label>Stages<select id="stageFilter" multiple></select></label>
        <label>Search<input id="searchInput" type="search" placeholder="trial, stage, status, path"></label>
        <button id="clearFilters" type="button">Clear</button>
      </div>
      <p class="muted">Select multiple trials or stages with Shift/Cmd/Ctrl. Tables share the same filters.</p>
    </section>
    <section class="panel">
      <h2>Run Summary</h2>
      <div class="table-wrap">
        <table id="summaryTable">
          <thead>
            <tr>
              <th data-key="trial">Trial</th>
              <th data-key="stage">Stage</th>
              <th data-key="status">Status</th>
              <th data-key="epoch_count">Epochs</th>
              <th data-key="best_epoch">Best R1 epoch</th>
              <th data-key="best_rank1">Best R1</th>
              <th data-key="best_map_epoch">Max mAP epoch</th>
              <th data-key="best_map">Best mAP</th>
              <th data-key="best_minp">Best mINP</th>
              <th data-key="checkpoint_epoch">Ckpt epoch</th>
              <th data-key="checkpoint_rank1">Ckpt R1</th>
              <th data-key="checkpoint_map">Ckpt mAP</th>
              <th data-key="final_rank1">Final R1</th>
              <th data-key="final_map">Final mAP</th>
              <th data-key="final_associate_rate">Assoc</th>
              <th data-key="runtime">Runtime</th>
              <th data-key="batch_size">Batch</th>
              <th data-key="checkpoint_exists">Ckpt</th>
              <th data-key="log_path">Log</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </section>
    <section class="panel">
      <h2>Epoch Details</h2>
      <div class="table-wrap">
        <table id="epochTable">
          <thead>
            <tr>
              <th data-key="trial">Trial</th>
              <th data-key="stage">Stage</th>
              <th data-key="epoch">Epoch</th>
              <th data-key="rank1">R1</th>
              <th data-key="rank5">R5</th>
              <th data-key="rank10">R10</th>
              <th data-key="rank20">R20</th>
              <th data-key="map">mAP</th>
              <th data-key="minp">mINP</th>
              <th data-key="best_r1">Logged best R1</th>
              <th data-key="best_map">Logged best mAP</th>
              <th data-key="associate_rate">Assoc</th>
              <th data-key="log_path">Log</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </section>
    <section class="panel">
      <h2>Validation</h2>
      <div class="validation" id="validation"></div>
    </section>
  </main>
  <script id="report-data" type="application/json">{data_json}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('report-data').textContent);
    const state = {{
      summarySort: {{ key: 'trial', dir: 'asc' }},
      epochSort: {{ key: 'trial', dir: 'asc' }},
      sysuCompareSort: {{ key: 'type', dir: 'asc' }},
      regdbCompareSort: {{ key: 'type', dir: 'asc' }}
    }};

    const summaryColumns = ['trial', 'stage', 'status', 'epoch_count', 'best_epoch', 'best_rank1', 'best_map_epoch', 'best_map', 'best_minp', 'checkpoint_epoch', 'checkpoint_rank1', 'checkpoint_map', 'final_rank1', 'final_map', 'final_associate_rate', 'runtime', 'batch_size', 'checkpoint_exists', 'log_path'];
    const epochColumns = ['trial', 'stage', 'epoch', 'rank1', 'rank5', 'rank10', 'rank20', 'map', 'minp', 'best_r1', 'best_map', 'associate_rate', 'log_path'];
    const sysuCompareColumns = ['type', 'method', 'sysu_all_rank1', 'sysu_all_map', 'sysu_indoor_rank1', 'sysu_indoor_map'];
    const regdbCompareColumns = ['type', 'method', 'regdb_v2t_rank1', 'regdb_v2t_map', 'regdb_t2v_rank1', 'regdb_t2v_map'];
    const metricKeys = new Set(['best_rank1', 'best_rank5', 'best_rank10', 'best_rank20', 'best_map', 'best_minp', 'checkpoint_rank1', 'checkpoint_map', 'final_rank1', 'final_rank5', 'final_rank10', 'final_rank20', 'final_map', 'final_minp', 'rank1', 'rank5', 'rank10', 'rank20', 'map', 'minp', 'model_r1', 'model_map', 'best_r1', 'sysu_all_rank1', 'sysu_all_map', 'sysu_indoor_rank1', 'sysu_indoor_map', 'regdb_v2t_rank1', 'regdb_v2t_map', 'regdb_t2v_rank1', 'regdb_t2v_map']);

    function uniqueValues(rows, key) {{
      return Array.from(new Set(rows.map(row => row[key]).filter(value => value !== null && value !== undefined))).sort((a, b) => String(a).localeCompare(String(b), undefined, {{ numeric: true }}));
    }}

    function optionHtml(values) {{
      return values.map(value => `<option value="${{escapeHtml(String(value))}}">${{escapeHtml(String(value))}}</option>`).join('');
    }}

    function selectedValues(id) {{
      return Array.from(document.getElementById(id).selectedOptions).map(option => option.value);
    }}

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, char => ({{ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }}[char]));
    }}

    function fmtNumber(value, digits = 2) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '';
      return Number(value).toFixed(digits);
    }}

    function fmtPercent(value) {{
      const num = Number(value);
      if (value === null || value === undefined || Number.isNaN(num)) return '';
      return `${{num.toFixed(2)}}%`;
    }}

    function fmtRate(value) {{
      const num = Number(value);
      if (value === null || value === undefined || Number.isNaN(num)) return '';
      return `${{(num * 100).toFixed(2)}}%`;
    }}

    function metricColor(value, alreadyPercent = true) {{
      const num = Number(value);
      if (value === null || value === undefined || Number.isNaN(num)) return '';
      const t = Math.max(0, Math.min(1, alreadyPercent ? num / 100 : num));
      const hue = 8 + 132 * t;
      return `background:hsl(${{hue}} 78% 92%); color:#172033;`;
    }}

    function renderStatus(row) {{
      const cls = row.complete ? 'complete' : 'incomplete';
      return `<span class="status ${{cls}}">${{escapeHtml(row.status)}}</span>`;
    }}

    function cell(row, key) {{
      const value = row[key];
      if (key === 'status') return `<td>${{renderStatus(row)}}</td>`;
      if (key === 'checkpoint_exists') return `<td>${{value ? 'yes' : 'no'}}</td>`;
      if (key === 'final_associate_rate' || key === 'associate_rate') return `<td class="num" style="${{metricColor(value, false)}}">${{fmtRate(value)}}</td>`;
      if (metricKeys.has(key)) return `<td class="num" style="${{metricColor(value)}}">${{fmtPercent(value)}}</td>`;
      if (typeof value === 'number') return `<td class="num">${{fmtNumber(value, Number.isInteger(value) ? 0 : 2)}}</td>`;
      if (String(key).includes('path') || key === 'log_path') return `<td class="mono">${{escapeHtml(value ?? '')}}</td>`;
      return `<td>${{escapeHtml(value ?? '')}}</td>`;
    }}

    function compareCell(row, key) {{
      const value = row[key];
      if (metricKeys.has(key)) {{
        if (value === null || value === undefined || Number.isNaN(Number(value))) return '<td class="num muted">-</td>';
        return `<td class="num" style="${{metricColor(value)}}">${{fmtNumber(value)}}</td>`;
      }}
      return `<td>${{escapeHtml(value ?? '')}}</td>`;
    }}

    function rowMatches(row) {{
      const trialValues = selectedValues('trialFilter');
      const stageValues = selectedValues('stageFilter');
      if (trialValues.length && !trialValues.includes(String(row.trial))) return false;
      if (stageValues.length && !stageValues.includes(String(row.stage))) return false;
      const needle = document.getElementById('searchInput').value.trim().toLowerCase();
      if (!needle) return true;
      const haystack = Object.values(row).join(' ').toLowerCase();
      return haystack.includes(needle);
    }}

    function comparisonMatches(row) {{
      const typeValues = selectedValues('compareTypeFilter');
      if (typeValues.length && !typeValues.includes(String(row.type))) return false;
      const needle = document.getElementById('compareSearchInput').value.trim().toLowerCase();
      if (!needle) return true;
      const haystack = Object.values(row).join(' ').toLowerCase();
      return haystack.includes(needle);
    }}

    function sortRows(rows, sortState) {{
      return rows.slice().sort((a, b) => {{
        const av = a[sortState.key];
        const bv = b[sortState.key];
        const emptyA = av === null || av === undefined || av === '';
        const emptyB = bv === null || bv === undefined || bv === '';
        if (emptyA && emptyB) return 0;
        if (emptyA) return 1;
        if (emptyB) return -1;
        let result;
        if (typeof av === 'number' && typeof bv === 'number') {{
          result = av - bv;
        }} else {{
          result = String(av).localeCompare(String(bv), undefined, {{ numeric: true }});
        }}
        return sortState.dir === 'asc' ? result : -result;
      }});
    }}

    function updateHeaderState(tableId, sortState) {{
      document.querySelectorAll(`#${{tableId}} th[data-key]`).forEach(th => {{
        th.classList.toggle('active', th.dataset.key === sortState.key);
        th.classList.toggle('asc', th.dataset.key === sortState.key && sortState.dir === 'asc');
        th.classList.toggle('desc', th.dataset.key === sortState.key && sortState.dir === 'desc');
      }});
    }}

    function renderSummaryTable() {{
      const tbody = document.querySelector('#summaryTable tbody');
      const rows = sortRows(DATA.summary_rows.filter(rowMatches), state.summarySort);
      tbody.innerHTML = rows.map(row => `<tr>${{summaryColumns.map(key => cell(row, key)).join('')}}</tr>`).join('');
      updateHeaderState('summaryTable', state.summarySort);
    }}

    function renderEpochTable() {{
      const tbody = document.querySelector('#epochTable tbody');
      const rows = sortRows(DATA.epoch_rows.filter(rowMatches), state.epochSort);
      tbody.innerHTML = rows.map(row => `<tr>${{epochColumns.map(key => cell(row, key)).join('')}}</tr>`).join('');
      updateHeaderState('epochTable', state.epochSort);
    }}

    function renderComparisonTable(tableId, columns, sortState) {{
      const tbody = document.querySelector(`#${{tableId}} tbody`);
      const rows = sortRows(DATA.comparison_rows.filter(comparisonMatches), sortState);
      tbody.innerHTML = rows.map(row => `<tr>${{columns.map(key => compareCell(row, key)).join('')}}</tr>`).join('');
      updateHeaderState(tableId, sortState);
    }}

    function renderComparisonTables() {{
      renderComparisonTable('sysuCompareTable', sysuCompareColumns, state.sysuCompareSort);
      renderComparisonTable('regdbCompareTable', regdbCompareColumns, state.regdbCompareSort);
    }}

    function renderCards() {{
      const validation = DATA.validation;
      const completeRuns = validation.complete_run_count;
      const expectedRuns = validation.expected_run_count;
      const maxR1 = validation.stage2_best_rank1_max;
      const meanR1 = validation.stage2_best_rank1_mean;
      const meanMap = validation.stage2_best_map_mean;
      const cards = [
        ['Runs', `${{completeRuns}} / ${{expectedRuns}}`, validation.ok ? 'all expected logs complete' : 'missing or incomplete logs remain'],
        ['Stage2 Best R1', maxR1 === null ? '' : fmtPercent(maxR1), 'max over completed stage2 trials'],
        ['Stage2 Mean R1', meanR1 === null ? '' : fmtPercent(meanR1), `${{validation.complete_stage2_count}} completed stage2 trials`],
        ['Stage2 Mean mAP', meanMap === null ? '' : fmtPercent(meanMap), 'mean max mAP over completed stage2 trials']
      ];
      document.getElementById('cards').innerHTML = cards.map(([label, value, note]) => `<article class="card"><div class="label">${{escapeHtml(label)}}</div><div class="value">${{escapeHtml(value || 'n/a')}}</div><div class="note">${{escapeHtml(note)}}</div></article>`).join('');
    }}

    function renderValidation() {{
      const validation = DATA.validation;
      const missing = validation.missing_logs.length ? validation.missing_logs.join(', ') : 'none';
      const incomplete = validation.incomplete_runs.length ? validation.incomplete_runs.join(', ') : 'none';
      const manifest = DATA.manifest;
      document.getElementById('validation').innerHTML = [
        `<div><strong>Status:</strong> ${{validation.ok ? 'complete' : 'incomplete'}}</div>`,
        `<div><strong>Summary rows:</strong> ${{validation.actual_run_count}} / ${{validation.expected_run_count}}</div>`,
        `<div><strong>Epoch rows:</strong> ${{validation.epoch_row_count}}</div>`,
        `<div><strong>Missing logs:</strong> <span class="mono">${{escapeHtml(missing)}}</span></div>`,
        `<div><strong>Incomplete runs:</strong> <span class="mono">${{escapeHtml(incomplete)}}</span></div>`,
        `<div><strong>Metric policy:</strong> ${{escapeHtml(manifest.metric_policy)}}</div>`,
        `<div><strong>Repo:</strong> <span class="mono">${{escapeHtml(manifest.repo_path)}}</span></div>`
      ].join('');
    }}

    function renderAll() {{
      renderCards();
      renderComparisonTables();
      renderSummaryTable();
      renderEpochTable();
      renderValidation();
    }}

    function setSort(target, key) {{
      const sortState = {{
        summary: state.summarySort,
        epoch: state.epochSort,
        sysuCompare: state.sysuCompareSort,
        regdbCompare: state.regdbCompareSort
      }}[target];
      if (sortState.key === key) {{
        sortState.dir = sortState.dir === 'asc' ? 'desc' : 'asc';
      }} else {{
        sortState.key = key;
        sortState.dir = 'asc';
      }}
      renderAll();
    }}

    function init() {{
      document.getElementById('trialFilter').innerHTML = optionHtml(uniqueValues(DATA.summary_rows, 'trial'));
      document.getElementById('stageFilter').innerHTML = optionHtml(uniqueValues(DATA.summary_rows, 'stage'));
      document.getElementById('compareTypeFilter').innerHTML = optionHtml(uniqueValues(DATA.comparison_rows, 'type'));
      document.getElementById('trialFilter').addEventListener('change', renderAll);
      document.getElementById('stageFilter').addEventListener('change', renderAll);
      document.getElementById('searchInput').addEventListener('input', renderAll);
      document.getElementById('compareTypeFilter').addEventListener('change', renderAll);
      document.getElementById('compareSearchInput').addEventListener('input', renderAll);
      document.getElementById('clearFilters').addEventListener('click', () => {{
        document.getElementById('trialFilter').selectedIndex = -1;
        document.getElementById('stageFilter').selectedIndex = -1;
        document.getElementById('searchInput').value = '';
        renderAll();
      }});
      document.getElementById('clearCompareFilters').addEventListener('click', () => {{
        document.getElementById('compareTypeFilter').selectedIndex = -1;
        document.getElementById('compareSearchInput').value = '';
        renderAll();
      }});
      document.querySelectorAll('#summaryTable th[data-key]').forEach(th => th.addEventListener('click', () => setSort('summary', th.dataset.key)));
      document.querySelectorAll('#epochTable th[data-key]').forEach(th => th.addEventListener('click', () => setSort('epoch', th.dataset.key)));
      document.querySelectorAll('#sysuCompareTable th[data-key]').forEach(th => th.addEventListener('click', () => setSort('sysuCompare', th.dataset.key)));
      document.querySelectorAll('#regdbCompareTable th[data-key]').forEach(th => th.addEventListener('click', () => setSort('regdbCompare', th.dataset.key)));
      renderAll();
    }}

    init();
  </script>
</body>
</html>
"""


def parse_trials(value: str) -> list[int]:
    trials: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            trials.extend(range(int(start), int(end) + 1))
        else:
            trials.append(int(part))
    return sorted(dict.fromkeys(trials))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--logs", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--trials", default="1-10")
    args = parser.parse_args()

    root = args.root.resolve()
    log_root = (args.logs or root / "logs").resolve()
    output = (args.output or root / "htmls" / "stats.html").resolve()
    payload = collect(root, log_root, parse_trials(args.trials))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(payload), encoding="utf-8")
    print(f"wrote {output}")
    print(
        "complete_runs={complete}/{expected} epoch_rows={epochs} ok={ok}".format(
            complete=payload["validation"]["complete_run_count"],
            expected=payload["validation"]["expected_run_count"],
            epochs=payload["validation"]["epoch_row_count"],
            ok=payload["validation"]["ok"],
        )
    )


if __name__ == "__main__":
    main()
