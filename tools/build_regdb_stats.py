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


def comparison_row(
    row_type: str,
    method: str,
    venue: str,
    sysu_all: tuple[float | None, float | None],
    sysu_indoor: tuple[float | None, float | None],
    regdb_v2t: tuple[float | None, float | None],
    regdb_t2v: tuple[float | None, float | None],
    *,
    source: str = "paper",
    is_ours: bool = False,
) -> dict[str, Any]:
    return {
        "type": row_type,
        "method": method,
        "venue": venue,
        "sysu_all_rank1": sysu_all[0],
        "sysu_all_map": sysu_all[1],
        "sysu_indoor_rank1": sysu_indoor[0],
        "sysu_indoor_map": sysu_indoor[1],
        "regdb_v2t_rank1": regdb_v2t[0],
        "regdb_v2t_map": regdb_v2t[1],
        "regdb_t2v_rank1": regdb_t2v[0],
        "regdb_t2v_map": regdb_t2v[1],
        "source": source,
        "is_ours": is_ours,
    }


COMPARISON_ROWS: list[dict[str, Any]] = [
    comparison_row("SVI-ReID", "DDAG", "ECCV 2020", (54.8, 53.0), (61.0, 68.0), (69.3, 63.5), (68.1, 61.8)),
    comparison_row("SVI-ReID", "AGW", "TPAMI 2021", (47.5, 47.7), (54.2, 63.0), (70.1, 66.4), (70.5, 65.9)),
    comparison_row("SVI-ReID", "NFS", "CVPR 2021", (56.9, 55.5), (62.8, 69.8), (80.5, 72.1), (78.0, 69.8)),
    comparison_row("SVI-ReID", "LbA", "ICCV 2021", (55.4, 54.1), (58.5, 66.3), (74.2, 67.6), (72.4, 65.5)),
    comparison_row("SVI-ReID", "CAJ", "ICCV 2021", (69.9, 66.9), (76.3, 80.4), (85.0, 79.1), (84.8, 77.8)),
    comparison_row("SVI-ReID", "MPANet", "CVPR 2021", (70.6, 68.2), (76.7, 81.0), (83.7, 80.9), (82.8, 80.7)),
    comparison_row("SVI-ReID", "DART", "CVPR 2022", (68.7, 66.3), (72.5, 78.2), (83.6, 75.7), (82.0, 73.8)),
    comparison_row("SVI-ReID", "FMCNet", "CVPR 2022", (66.3, 62.5), (68.2, 74.1), (89.1, 84.4), (88.4, 83.9)),
    comparison_row("SVI-ReID", "MAUM", "CVPR 2022", (71.7, 68.8), (77.0, 81.9), (87.9, 85.1), (87.0, 84.3)),
    comparison_row("SVI-ReID", "MID", "AAAI 2022", (60.3, 59.4), (64.9, 70.1), (87.5, 84.9), (84.3, 81.4)),
    comparison_row("SVI-ReID", "LUPI", "ECCV 2022", (71.1, 67.6), (82.4, 82.7), (88.0, 82.7), (86.8, 81.3)),
    comparison_row("SVI-ReID", "DEEN", "CVPR 2023", (74.7, 71.8), (80.3, 83.3), (91.1, 85.1), (89.5, 83.4)),
    comparison_row("SVI-ReID", "SGIEL", "CVPR 2023", (77.1, 72.3), (82.1, 83.0), (92.2, 86.6), (91.1, 85.2)),
    comparison_row("SVI-ReID", "PartMix", "CVPR 2023", (77.8, 74.6), (81.5, 84.4), (85.7, 82.3), (84.9, 82.5)),
    comparison_row("SVI-ReID", "CAL", "ICCV 2023", (74.7, 71.7), (79.7, 83.7), (94.5, 88.7), (93.6, 87.6)),
    comparison_row("SVI-ReID", "MUN", "ICCV 2023", (76.2, 73.8), (79.4, 82.1), (95.2, 87.2), (91.9, 85.0)),
    comparison_row("SVI-ReID", "SAAI", "ICCV 2023", (75.9, 77.0), (83.2, 88.0), (91.1, 91.5), (92.1, 92.0)),
    comparison_row("SVI-ReID", "FDNM", "arXiv 2024", (77.8, 75.1), (87.3, 89.1), (95.5, 90.0), (94.0, 88.7)),
    comparison_row("SVI-ReID", "PMWGCN", "TIFS 2024", (90.6, 84.5), (88.8, 81.6), (66.8, 64.9), (72.6, 76.2)),
    comparison_row("SVI-ReID", "LCNL", "IJCV 2024", (70.2, 68.0), (76.2, 80.3), (85.6, 78.7), (84.0, 76.9)),
    comparison_row("SSVI-ReID", "OTLA", "ECCV 2022", (48.2, 43.9), (47.4, 56.8), (49.9, 41.8), (49.6, 42.8)),
    comparison_row("SSVI-ReID", "TAA", "TIP 2023", (48.8, 42.3), (50.1, 56.0), (62.2, 56.0), (63.8, 56.5)),
    comparison_row("SSVI-ReID", "DPIS", "ICCV 2023", (58.4, 55.6), (63.0, 70.0), (62.3, 53.2), (61.5, 52.7)),
    comparison_row("USVI-ReID", "H2H", "TIP 2021", (30.2, 29.4), (None, None), (23.8, 18.9), (None, None)),
    comparison_row("USVI-ReID", "OTLA", "ECCV 2022", (29.9, 27.1), (29.8, 38.8), (32.9, 29.7), (32.1, 28.6)),
    comparison_row("USVI-ReID", "ADCA", "ACM MM 2022", (45.5, 42.7), (50.6, 59.1), (67.2, 64.1), (68.5, 63.8)),
    comparison_row("USVI-ReID", "NGLR", "CVPR 2023", (50.4, 47.4), (53.5, 61.7), (85.6, 76.7), (82.9, 75.0)),
    comparison_row("USVI-ReID", "MBCCM", "ACM MM 2023", (53.1, 48.2), (55.2, 62.0), (83.8, 77.9), (82.8, 76.7)),
    comparison_row("USVI-ReID", "CCLNet", "ACM MM 2023", (54.0, 50.2), (56.7, 65.1), (69.9, 65.5), (70.2, 66.7)),
    comparison_row("USVI-ReID", "PGM", "TIFS 2023", (57.3, 51.8), (56.2, 62.7), (69.5, 65.4), (69.9, 65.2)),
    comparison_row("USVI-ReID", "GUR*", "ICCV 2023", (61.0, 57.0), (64.2, 69.5), (73.9, 70.2), (75.0, 69.9)),
    comparison_row("USVI-ReID", "MMM", "arXiv 2024", (61.6, 57.9), (64.4, 70.4), (89.7, 80.5), (85.8, 77.0)),
    comparison_row("USVI-ReID", "PCLHD", "NeurIPS 2024", (64.4, 58.7), (69.5, 74.4), (84.3, 80.7), (82.7, 78.4)),
    comparison_row("USVI-ReID", "PCLHD+MDUE", "NeurIPS 2024", (65.19, 60.24), (69.56, 74.18), (91.12, 83.21), (89.48, 82.24)),
    comparison_row(
        "USVI-ReID",
        "PCLHD+MDUE+CGCF",
        "Ours, NeurIPS 2024",
        (66.31, 61.44),
        (72.49, 75.64),
        (93.57, 85.69),
        (91.59, 84.08),
        is_ours=True,
    ),
]


QUICK_TRIALS = [1, 2, 3]
PRIMARY_REPORT_GROUP = "paper_amp_full"
QUICK_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "experiment": "baseline_quick",
        "label": "PCLHD quick baseline",
        "method": "PCLHD",
        "folder": "regdb_s2_baseline_quick",
        "group": "quick",
        "baseline_experiment": "baseline_quick",
        "mdue": False,
        "cgcf": False,
        "amp": True,
        "mdue_samples": 1,
        "dropout": 0.0,
        "notes": "Stage2-only quick baseline, 3 RegDB splits.",
    },
    {
        "experiment": "baseline_fp32_quick",
        "label": "PCLHD quick baseline (FP32)",
        "method": "PCLHD",
        "folder": "regdb_s2_baseline_fp32_quick",
        "group": "precision",
        "baseline_experiment": "baseline_quick",
        "mdue": False,
        "cgcf": False,
        "amp": False,
        "mdue_samples": 1,
        "dropout": 0.0,
        "notes": "Same quick baseline as above, but without AMP, to isolate numeric precision effects.",
    },
    {
        "experiment": "mdue_quick",
        "label": "PCLHD + MDUE",
        "method": "PCLHD+MDUE",
        "folder": "regdb_s2_mdue_quick",
        "group": "legacy_quick",
        "baseline_experiment": "baseline_quick",
        "mdue": True,
        "cgcf": False,
        "amp": True,
        "mdue_samples": 3,
        "dropout": 0.1,
        "notes": "MC-Dropout uncertainty estimation, S=3 and p=0.10.",
    },
    {
        "experiment": "mdue_cgcf_quick",
        "label": "PCLHD + MDUE + CGCF",
        "method": "PCLHD+MDUE+CGCF",
        "folder": "regdb_s2_mdue_cgcf_quick",
        "group": "legacy_quick",
        "baseline_experiment": "baseline_quick",
        "mdue": True,
        "cgcf": True,
        "amp": True,
        "mdue_samples": 3,
        "dropout": 0.1,
        "notes": "Full chapter-5 reproduction path for the quick 3-split run.",
    },
    {
        "experiment": "mdue_fix_quick",
        "label": "PCLHD + MDUE (fixed MC, quick)",
        "method": "PCLHD+MDUE",
        "folder": "regdb_s2_mdue_fix_quick",
        "group": "fixed_quick",
        "baseline_experiment": "baseline_quick",
        "mdue": True,
        "cgcf": False,
        "amp": True,
        "mdue_samples": 3,
        "dropout": 0.1,
        "notes": "Corrected MC-Dropout: active only during uncertainty sampling; quick trial set.",
    },
    {
        "experiment": "mdue_cgcf_fix_quick",
        "label": "PCLHD + MDUE + CGCF (fixed MC, quick)",
        "method": "PCLHD+MDUE+CGCF",
        "folder": "regdb_s2_mdue_cgcf_fix_quick",
        "group": "fixed_quick",
        "baseline_experiment": "baseline_quick",
        "mdue": True,
        "cgcf": True,
        "amp": True,
        "mdue_samples": 3,
        "dropout": 0.1,
        "notes": "Corrected MDUE plus confidence-guided cross-modal center fusion; quick trial set.",
    },
    {
        "experiment": "paper_amp_baseline",
        "label": "PCLHD AMP baseline (paper params)",
        "method": "PCLHD",
        "folder": "regdb_s2_amp_baseline",
        "group": "paper_amp",
        "baseline_experiment": "paper_amp_baseline",
        "mdue": False,
        "cgcf": False,
        "amp": True,
        "mdue_samples": 1,
        "dropout": 0.0,
        "notes": "Stage2 AMP run with paper-aligned 50 epochs and 100 iters for trials 1-3.",
    },
    {
        "experiment": "paper_amp_mdue",
        "label": "PCLHD + MDUE AMP (paper params)",
        "method": "PCLHD+MDUE",
        "folder": "regdb_s2_amp_mdue",
        "group": "paper_amp",
        "baseline_experiment": "paper_amp_baseline",
        "mdue": True,
        "cgcf": False,
        "amp": True,
        "mdue_samples": 3,
        "dropout": 0.1,
        "notes": "Paper-aligned AMP setting with MDUE S=3 and p=0.10.",
    },
    {
        "experiment": "paper_amp_mdue_cgcf",
        "label": "PCLHD + MDUE + CGCF AMP (paper params)",
        "method": "PCLHD+MDUE+CGCF",
        "folder": "regdb_s2_amp_mdue_cgcf",
        "group": "paper_amp",
        "baseline_experiment": "paper_amp_baseline",
        "mdue": True,
        "cgcf": True,
        "amp": True,
        "mdue_samples": 3,
        "dropout": 0.1,
        "notes": "Paper-aligned AMP setting with MDUE and CGCF enabled.",
    },
    {
        "experiment": "paper_amp_full_baseline",
        "label": "PCLHD AMP full baseline (stage1+stage2)",
        "method": "PCLHD",
        "folder": "regdb_s2_amp_full_baseline",
        "group": "paper_amp_full",
        "baseline_experiment": "paper_amp_full_baseline",
        "mdue": False,
        "cgcf": False,
        "amp": True,
        "mdue_samples": 1,
        "dropout": 0.0,
        "notes": "Full 50+50 AMP run: stage1 and stage2 are both trained under the AMP environment.",
    },
    {
        "experiment": "paper_amp_full_mdue",
        "label": "PCLHD + MDUE AMP full (stage1+stage2)",
        "method": "PCLHD+MDUE",
        "folder": "regdb_s2_amp_full_mdue",
        "group": "paper_amp_full",
        "baseline_experiment": "paper_amp_full_baseline",
        "mdue": True,
        "cgcf": False,
        "amp": True,
        "mdue_samples": 3,
        "dropout": 0.1,
        "notes": "Full 50+50 AMP run with MDUE S=3 and p=0.10 enabled from stage1 feature extraction onward.",
    },
    {
        "experiment": "paper_amp_full_mdue_cgcf",
        "label": "PCLHD + MDUE + CGCF AMP full",
        "method": "PCLHD+MDUE+CGCF",
        "folder": "regdb_s2_amp_full_mdue_cgcf",
        "group": "paper_amp_full",
        "baseline_experiment": "paper_amp_full_baseline",
        "mdue": True,
        "cgcf": True,
        "amp": True,
        "mdue_samples": 3,
        "dropout": 0.1,
        "notes": "Full AMP setting: reuse the MDUE stage1 checkpoint and enable CGCF during stage2 prototype fusion.",
    },
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
        "iters": args.get("iters"),
        "epochs": args.get("epochs"),
        "stage2_batch_size": args.get("stage2_batch_size"),
        "stage1_log_name": args.get("stage1_log_name"),
        "stage2_log_name": args.get("stage2_log_name"),
        "amp": args.get("amp"),
        "dropout": args.get("dropout"),
        "mdue_samples": args.get("mdue_samples"),
        "use_cgcf": args.get("use_cgcf"),
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
    rows = []
    for index, row in enumerate(COMPARISON_ROWS):
        enriched = dict(row)
        enriched["sort_order"] = index
        rows.append(enriched)
    return rows


def collect_quick_ablations(log_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quick_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for config in QUICK_EXPERIMENTS:
        complete_rows: list[dict[str, Any]] = []
        for trial in QUICK_TRIALS:
            log_path = log_root / config["folder"] / str(trial) / f"{trial}log.txt"
            summary, _ = parse_log(log_path, trial, "quick_stage2")
            row = dict(summary)
            row.update(
                {
                    "experiment": config["experiment"],
                    "experiment_label": config["label"],
                    "method": config["method"],
                    "folder": config["folder"],
                    "group": config["group"],
                    "baseline_experiment": config["baseline_experiment"],
                    "expected_mdue": config["mdue"],
                    "expected_cgcf": config["cgcf"],
                    "expected_amp": config["amp"],
                    "notes": config["notes"],
                }
            )
            quick_rows.append(row)
            if row["complete"]:
                complete_rows.append(row)

        best_r1 = finite_values(complete_rows, "best_rank1")
        best_map = finite_values(complete_rows, "best_map")
        complete_trials = len(complete_rows)
        expected_trials = len(QUICK_TRIALS)
        if complete_trials == expected_trials:
            status = "complete"
        elif complete_trials == 0:
            status = "pending"
        else:
            status = "running_or_incomplete"
        aggregate_rows.append(
            {
                "experiment": config["experiment"],
                "experiment_label": config["label"],
                "method": config["method"],
                "folder": config["folder"],
                "group": config["group"],
                "baseline_experiment": config["baseline_experiment"],
                "mdue": config["mdue"],
                "cgcf": config["cgcf"],
                "amp": config["amp"],
                "mdue_samples": config["mdue_samples"],
                "dropout": config["dropout"],
                "trials": ",".join(str(trial) for trial in QUICK_TRIALS),
                "expected_trials": expected_trials,
                "complete_trials": complete_trials,
                "status": status,
                "complete": complete_trials == expected_trials,
                "mean_best_rank1": mean(best_r1) if best_r1 else None,
                "mean_best_map": mean(best_map) if best_map else None,
                "max_best_rank1": max(best_r1) if best_r1 else None,
                "max_best_map": max(best_map) if best_map else None,
                "notes": config["notes"],
            }
        )

    aggregate_by_experiment = {row["experiment"]: row for row in aggregate_rows}
    for row in aggregate_rows:
        baseline = aggregate_by_experiment.get(row.get("baseline_experiment", "baseline_quick"))
        baseline_r1 = baseline.get("mean_best_rank1") if baseline else None
        baseline_map = baseline.get("mean_best_map") if baseline else None
        row["delta_rank1"] = (
            row["mean_best_rank1"] - baseline_r1
            if isinstance(row.get("mean_best_rank1"), (int, float)) and isinstance(baseline_r1, (int, float))
            else None
        )
        row["delta_map"] = (
            row["mean_best_map"] - baseline_map
            if isinstance(row.get("mean_best_map"), (int, float)) and isinstance(baseline_map, (int, float))
            else None
        )

    return quick_rows, aggregate_rows


def build_precision_summary(quick_aggregate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    amp_baseline = next(
        (row for row in quick_aggregate_rows if row["experiment"] == "baseline_quick"),
        None,
    )
    fp32_baseline = next(
        (row for row in quick_aggregate_rows if row["experiment"] == "baseline_fp32_quick"),
        None,
    )

    amp_r1 = amp_baseline.get("mean_best_rank1") if amp_baseline else None
    amp_map = amp_baseline.get("mean_best_map") if amp_baseline else None
    fp32_r1 = fp32_baseline.get("mean_best_rank1") if fp32_baseline else None
    fp32_map = fp32_baseline.get("mean_best_map") if fp32_baseline else None
    delta_r1 = (
        fp32_r1 - amp_r1
        if isinstance(fp32_r1, (int, float)) and isinstance(amp_r1, (int, float))
        else None
    )
    delta_map = (
        fp32_map - amp_map
        if isinstance(fp32_map, (int, float)) and isinstance(amp_map, (int, float))
        else None
    )
    expected_trials = len(QUICK_TRIALS)
    amp_complete = amp_baseline.get("complete_trials", 0) if amp_baseline else 0
    fp32_complete = fp32_baseline.get("complete_trials", 0) if fp32_baseline else 0
    complete = bool(
        amp_baseline
        and fp32_baseline
        and amp_complete == expected_trials
        and fp32_complete == expected_trials
    )

    if not complete:
        status = "evidence_incomplete"
        conclusion = "FP32 baseline is still running or missing; wait for all quick trials before deciding whether numeric precision explains the gap."
    elif abs(delta_r1 or 0.0) <= 0.5 and abs(delta_map or 0.0) <= 0.5:
        status = "precision_unlikely"
        conclusion = "FP32 and AMP quick baselines are within 0.5 percentage points; numeric precision is unlikely to be the main source of the paper gap."
    elif (delta_r1 or 0.0) > 0 and (delta_map or 0.0) > 0:
        status = "fp32_slightly_higher"
        conclusion = "FP32 is higher than AMP on this quick baseline, so mixed precision has a measurable small effect, but it does not explain the remaining gap to the paper setting."
    elif (delta_r1 or 0.0) < 0 and (delta_map or 0.0) < 0:
        status = "fp32_lower"
        conclusion = "FP32 is lower than AMP on this quick baseline; numeric precision is not the reason the reproduction lags the paper."
    else:
        status = "precision_difference"
        conclusion = "FP32 and AMP quick baselines differ by more than 0.5 percentage points with mixed metric signs; inspect individual trials before ruling out numeric precision."

    return {
        "status": status,
        "complete": complete,
        "expected_trials": expected_trials,
        "amp_complete_trials": amp_complete,
        "fp32_complete_trials": fp32_complete,
        "amp_mean_best_rank1": amp_r1,
        "amp_mean_best_map": amp_map,
        "fp32_mean_best_rank1": fp32_r1,
        "fp32_mean_best_map": fp32_map,
        "delta_rank1": delta_r1,
        "delta_map": delta_map,
        "threshold_pp": 0.5,
        "conclusion": conclusion,
        "amp_folder": amp_baseline.get("folder") if amp_baseline else None,
        "fp32_folder": fp32_baseline.get("folder") if fp32_baseline else None,
    }


def collect(root: Path, log_root: Path, trials: list[int]) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    missing_logs: list[str] = []
    incomplete_runs: list[str] = []
    quick_rows, quick_aggregate_rows = collect_quick_ablations(log_root)
    precision_summary = build_precision_summary(quick_aggregate_rows)

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
    primary_full_rows = [row for row in quick_rows if row.get("group") == PRIMARY_REPORT_GROUP]
    primary_full_configs = [
        row for row in quick_aggregate_rows if row.get("group") == PRIMARY_REPORT_GROUP
    ]
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
        "quick_expected_run_count": len(quick_rows),
        "quick_complete_run_count": sum(1 for row in quick_rows if row["complete"]),
        "quick_expected_config_count": len(quick_aggregate_rows),
        "quick_complete_config_count": sum(1 for row in quick_aggregate_rows if row["complete"]),
        "primary_full_expected_run_count": len(primary_full_rows),
        "primary_full_complete_run_count": sum(1 for row in primary_full_rows if row["complete"]),
        "primary_full_expected_config_count": len(primary_full_configs),
        "primary_full_complete_config_count": sum(1 for row in primary_full_configs if row["complete"]),
        "precision_check_status": precision_summary["status"],
        "precision_check_complete": precision_summary["complete"],
        "precision_delta_rank1": precision_summary["delta_rank1"],
        "precision_delta_map": precision_summary["delta_map"],
    }
    validation["legacy_ok"] = (
        validation["actual_run_count"] == expected_run_count
        and not missing_logs
        and not incomplete_runs
    )
    validation["ok"] = validation["legacy_ok"]
    validation["primary_full_ok"] = (
        validation["primary_full_expected_run_count"] > 0
        and validation["primary_full_complete_run_count"] == validation["primary_full_expected_run_count"]
        and validation["primary_full_complete_config_count"] == validation["primary_full_expected_config_count"]
    )
    validation["report_ok"] = validation["primary_full_ok"]
    return {
        "manifest": {
            "script": "tools/build_regdb_stats.py",
            "repo_path": str(root),
            "log_root": str(log_root),
            "output": "htmls/stats.html",
            "metric_policy": "Rank and mAP values are parsed from FC evaluation lines; best Rank-1 and max mAP are computed independently. Checkpoint fields use the logged best epoch.",
            "split_policy": "Full reproduction uses RegDB trials 1-10. Ablation and paper-parameter checks use trials 1-3 unless otherwise noted. Training logs are visible-to-thermal.",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "comparison_source": "Literature rows are transcribed from the NeurIPS 2024 PCLHD paper tables, including venue/year columns. Reproduced AMP runs are reported in the ablation and paper-parameter tables.",
        },
        "summary_rows": summary_rows,
        "epoch_rows": epoch_rows,
        "quick_rows": quick_rows,
        "quick_aggregate_rows": quick_aggregate_rows,
        "precision_summary": precision_summary,
        "comparison_rows": build_comparison_rows(log_root),
        "validation": validation,
    }


def build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    validation = payload["validation"]
    status_text = "complete" if validation["report_ok"] else "incomplete"
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
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
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
    .audit-list {{
      display: grid;
      grid-template-columns: 150px 1fr;
      gap: 8px 12px;
      margin: 12px 0 0;
    }}
    .audit-list dt {{
      color: var(--ink);
      font-weight: 720;
    }}
    .audit-list dd {{
      margin: 0;
      color: var(--muted);
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
    .precision-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .precision-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      padding: 10px;
    }}
    .precision-card .label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .precision-card .metric {{
      margin-top: 4px;
      font-size: 20px;
      font-weight: 760;
      font-variant-numeric: tabular-nums;
    }}
    .precision-card .sub {{
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
    }}
    .precision-callout {{
      grid-column: 1 / -1;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
      color: var(--muted);
    }}
    .precision-callout.complete {{
      border-color: #abefc6;
      background: #f6fef9;
    }}
    .precision-callout.incomplete {{
      border-color: #fedf89;
      background: #fffcf5;
    }}
    .tag {{
      display: inline-block;
      margin-left: 6px;
      border: 1px solid #bfdbfe;
      border-radius: 999px;
      background: #eff6ff;
      color: #1d4ed8;
      padding: 1px 7px;
      font-size: 11px;
      font-weight: 760;
      vertical-align: middle;
    }}
    .ours-row td {{
      font-weight: 680;
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
      .precision-grid {{ grid-template-columns: 1fr; }}
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
          <h3>论文第 5 章创新点</h3>
          <p><code>MDUE</code> 在特征抽取时保持 Dropout 随机性，多次前向估计样本表征方差，并把低方差样本赋予更高置信度。本次快速复现按论文设置 <code>S=3</code>、<code>p=0.10</code>。</p>
          <p><code>CGCF</code> 在 All-memory 原型构造时，不再简单平均 RGB/IR 中心，而是用模态置信度和跨模态一致性加权融合簇中心。</p>
        </div>
        <div class="explain-block">
          <h3>本次复现设置</h3>
          <p>环境使用 Python 3.12、PyTorch 2.6.0、单张 Tesla V100 16GB。由于 batch 64 在后续 trial 出现显存边缘 OOM，剩余 trial 已改为 batch 32 续跑；表格中的 <code>Batch</code> 列会记录每个 run 的实际 batch size。</p>
          <p>完整复现保留 10 split 结果；论文创新点复现实验只跑 trial 1-3 的快速消融，并用 <code>AMP fp16</code> 加速训练。</p>
        </div>
      </div>
    </section>
    <section class="panel">
      <h2>算法流程</h2>
      <p>本项目把无监督跨模态 ReID 拆成“单模态伪标签初始化”和“跨模态关联优化”两段。论文第 5 章的最终方法在此基础上加入 MDUE 样本置信度和 CGCF 跨模态中心融合。</p>
      <ol class="flow-steps">
        <li><strong>1. 数据准备</strong><span>读取 RegDB/SYSU 的 RGB 与 IR 图像，按官方划分生成训练集、查询集和图库。</span></li>
        <li><strong>2. 特征初始化</strong><span>使用 ImageNet 预训练 backbone 提取两种模态的初始特征，并用 GeM/BNNeck 形成检索向量。</span></li>
        <li><strong>3. Stage1 聚类训练</strong><span>分别对 RGB 与 IR 特征做 DBSCAN 聚类，把聚类 ID 当作伪标签，用 DCL 和 ClusterMemory 训练初始模型。</span></li>
        <li><strong>4. MDUE 特征抽取</strong><span>开启 MC-Dropout，原图和水平翻转图各做多次随机前向，取均值作为稳健特征，并由方差得到样本置信度。</span></li>
        <li><strong>5. CGCF 原型融合</strong><span>对同一 All-cluster 的 RGB/IR 中心，用模态置信度和中心相似度计算权重，得到更可靠的统一跨模态原型。</span></li>
        <li><strong>6. Stage2 跨模态关联</strong><span>加载 stage1 checkpoint，构造 RGB、IR、All 三组 memory，通过跨模态相似度和二分图匹配建立 RGB-IR 原型对应关系。</span></li>
        <li><strong>7. 评估与汇总</strong><span>用最终 stage2 checkpoint 做 visible-to-thermal 与 thermal-to-visible 检索，汇总 Rank-1、mAP、mINP 和逐 trial 均值。</span></li>
      </ol>
    </section>
    <section class="panel">
      <h2>实现审计与参数对齐</h2>
      <p>这部分记录当前复现代码和论文方法的对应关系，便于解释“代码里是否真的包含 MDUE/CGCF”。</p>
      <dl class="audit-list">
        <dt>原始代码状态</dt>
        <dd>初始提交 <code>d9145c9</code> 中已有 <code>extract_features(..., mc_drop)</code> 和 <code>confidence_fusion_features()</code> 这类辅助函数雏形，但 <code>train_regdb.py</code> 没有暴露 <code>--mdue-samples</code>、<code>--use-cgcf</code>，也没有把它们接入 RegDB stage1/stage2 训练路径。</dd>
        <dt>本次接入</dt>
        <dd>当前版本在 <code>train_regdb.py</code> 中加入 MDUE/CGCF 开关：stage1 与 stage2 的聚类特征抽取使用 <code>S=3</code> 的 MC-Dropout 均值特征；stage2 的 All-memory 原型在开启 CGCF 时调用置信度引导的跨模态中心融合。</dd>
        <dt>Dropout 修正</dt>
        <dd><code>clustercontrast/models/agw.py</code> 使用 <code>mc_dropout_active</code> 控制 Dropout，仅在不确定性采样阶段激活；普通评估和 baseline 不受随机 Dropout 干扰。</dd>
        <dt>论文参数</dt>
        <dd>full AMP 组保持 <code>epochs=50</code>、<code>iters=100</code>、<code>eps=0.3</code>、<code>momentum=0.1</code>、<code>num_instances=16</code>；baseline 使用 <code>S=1</code>、<code>dropout=0</code>、不开 CGCF，MDUE/CGCF 组使用 <code>S=3</code>、<code>dropout=0.10</code>。</dd>
        <dt>环境差异</dt>
        <dd>允许差异集中在 Python 3.12、PyTorch 2.6.0、V100 16GB 和 AMP fp16。stage2 使用较小 batch 以适配显存，实际值在表格中记录。</dd>
        <dt>公式注意</dt>
        <dd>当前 CGCF 按论文公式同时用跨模态一致性 <code>s_vr</code> 乘到 RGB/IR 两侧权重；两侧共同归一化时，差异性权重主要来自 <code>c_rgb</code> 与 <code>c_ir</code>，这会影响后续解释 CGCF 增益的幅度。</dd>
      </dl>
    </section>
    <section class="panel">
      <h2>对比实验表格</h2>
      <div class="controls">
        <label>Types<select id="compareTypeFilter" multiple></select></label>
        <label>Search<input id="compareSearchInput" type="search" placeholder="method, type, venue, source"></label>
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
              <th data-key="venue" rowspan="2">来源</th>
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
              <th data-key="venue" rowspan="2">来源</th>
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
      <p class="source-note muted">文献行来自 NeurIPS 2024 PCLHD 论文表格；来源列沿用论文的 Venue/年份标注。论文最终方法 <code>PCLHD+MDUE+CGCF</code> 标为 Ours 并固定在默认排序的最后一行；本次 AMP 复现实验结果在下方消融与 paper-parameter 表中单独展示。</p>
    </section>
    <section class="panel">
      <h2>AMP vs FP32 Baseline Check</h2>
      <p>这里只比较同一套 3 split quick baseline：<code>baseline_quick</code> 使用 AMP，<code>baseline_fp32_quick</code> 关闭 AMP。它用于隔离“混合精度是否导致指标差异”，不等价于论文正式 10 split 结果。</p>
      <div class="precision-grid" id="precisionSummary"></div>
    </section>
    <section class="panel">
      <h2>Ablations and Paper-Parameter Runs</h2>
      <div class="controls">
        <label>Configs<select id="quickExperimentFilter" multiple></select></label>
        <label>Search<input id="quickSearchInput" type="search" placeholder="method, config, status, folder"></label>
        <button id="clearQuickFilters" type="button">Clear</button>
      </div>
      <p class="muted">这里汇总 quick 消融、修复版 MDUE smoke，以及按论文参数启动的 AMP 组。默认使用 RegDB trial 1-3 判断趋势；不等价于论文正式 10 split 均值。</p>
      <div class="table-wrap">
        <table id="quickAblationTable">
          <thead>
            <tr>
              <th data-key="experiment_label">配置</th>
              <th data-key="group">组</th>
              <th data-key="method">方法</th>
              <th data-key="mdue">MDUE</th>
              <th data-key="cgcf">CGCF</th>
              <th data-key="amp">AMP</th>
              <th data-key="mdue_samples">S</th>
              <th data-key="dropout">Dropout</th>
              <th data-key="complete_trials">Done</th>
              <th data-key="mean_best_rank1">Mean Best R1</th>
              <th data-key="mean_best_map">Mean Best mAP</th>
              <th data-key="delta_rank1">Delta R1</th>
              <th data-key="delta_map">Delta mAP</th>
              <th data-key="status">Status</th>
              <th data-key="baseline_experiment">Baseline</th>
              <th data-key="folder">Log folder</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="table-wrap" style="margin-top:12px;">
        <table id="quickTrialTable">
          <thead>
            <tr>
              <th data-key="experiment_label">配置</th>
              <th data-key="trial">Trial</th>
              <th data-key="status">Status</th>
              <th data-key="epoch_count">Epochs</th>
              <th data-key="best_rank1">Best R1</th>
              <th data-key="best_map">Best mAP</th>
              <th data-key="final_rank1">Final R1</th>
              <th data-key="final_map">Final mAP</th>
              <th data-key="amp">AMP</th>
              <th data-key="mdue_samples">S</th>
              <th data-key="dropout">Dropout</th>
              <th data-key="use_cgcf">CGCF</th>
              <th data-key="runtime">Runtime</th>
              <th data-key="log_path">Log</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
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
        <dt>MDUE</dt><dd>MC-Dropout Uncertainty Estimation。通过多次随机前向计算样本特征均值与方差，把不稳定样本转成低置信度。</dd>
        <dt>CGCF</dt><dd>Confidence-Guided Cross-modal Center Fusion。用样本置信度和 RGB/IR 中心一致性加权融合跨模态原型。</dd>
        <dt>AMP</dt><dd>Automatic Mixed Precision。训练前向和反向使用 fp16 混合精度，以降低显存占用并提升吞吐。</dd>
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
      sysuCompareSort: {{ key: 'sort_order', dir: 'asc' }},
      regdbCompareSort: {{ key: 'sort_order', dir: 'asc' }},
      quickAblationSort: {{ key: 'experiment', dir: 'asc' }},
      quickTrialSort: {{ key: 'experiment', dir: 'asc' }}
    }};

    const summaryColumns = ['trial', 'stage', 'status', 'epoch_count', 'best_epoch', 'best_rank1', 'best_map_epoch', 'best_map', 'best_minp', 'checkpoint_epoch', 'checkpoint_rank1', 'checkpoint_map', 'final_rank1', 'final_map', 'final_associate_rate', 'runtime', 'batch_size', 'checkpoint_exists', 'log_path'];
    const epochColumns = ['trial', 'stage', 'epoch', 'rank1', 'rank5', 'rank10', 'rank20', 'map', 'minp', 'best_r1', 'best_map', 'associate_rate', 'log_path'];
    const sysuCompareColumns = ['type', 'method', 'venue', 'sysu_all_rank1', 'sysu_all_map', 'sysu_indoor_rank1', 'sysu_indoor_map'];
    const regdbCompareColumns = ['type', 'method', 'venue', 'regdb_v2t_rank1', 'regdb_v2t_map', 'regdb_t2v_rank1', 'regdb_t2v_map'];
    const quickAblationColumns = ['experiment_label', 'group', 'method', 'mdue', 'cgcf', 'amp', 'mdue_samples', 'dropout', 'complete_trials', 'mean_best_rank1', 'mean_best_map', 'delta_rank1', 'delta_map', 'status', 'baseline_experiment', 'folder'];
    const quickTrialColumns = ['experiment_label', 'trial', 'status', 'epoch_count', 'best_rank1', 'best_map', 'final_rank1', 'final_map', 'amp', 'mdue_samples', 'dropout', 'use_cgcf', 'runtime', 'log_path'];
    const metricKeys = new Set(['best_rank1', 'best_rank5', 'best_rank10', 'best_rank20', 'best_map', 'best_minp', 'checkpoint_rank1', 'checkpoint_map', 'final_rank1', 'final_rank5', 'final_rank10', 'final_rank20', 'final_map', 'final_minp', 'rank1', 'rank5', 'rank10', 'rank20', 'map', 'minp', 'model_r1', 'model_map', 'best_r1', 'sysu_all_rank1', 'sysu_all_map', 'sysu_indoor_rank1', 'sysu_indoor_map', 'regdb_v2t_rank1', 'regdb_v2t_map', 'regdb_t2v_rank1', 'regdb_t2v_map', 'mean_best_rank1', 'mean_best_map', 'max_best_rank1', 'max_best_map']);
    const deltaKeys = new Set(['delta_rank1', 'delta_map']);

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
      if (key === 'method') {{
        const tag = row.is_ours ? '<span class="tag">Ours</span>' : '';
        return `<td>${{escapeHtml(value ?? '')}}${{tag}}</td>`;
      }}
      if (metricKeys.has(key)) {{
        if (value === null || value === undefined || Number.isNaN(Number(value))) return '<td class="num muted">-</td>';
        return `<td class="num" style="${{metricColor(value)}}">${{fmtNumber(value)}}</td>`;
      }}
      return `<td>${{escapeHtml(value ?? '')}}</td>`;
    }}

    function fmtDelta(value) {{
      const num = Number(value);
      if (value === null || value === undefined || Number.isNaN(num)) return '';
      return `${{num >= 0 ? '+' : ''}}${{num.toFixed(2)}}`;
    }}

    function quickCell(row, key) {{
      const value = row[key];
      if (key === 'status') return `<td>${{renderStatus(row)}}</td>`;
      if (typeof value === 'boolean') return `<td>${{value ? 'yes' : 'no'}}</td>`;
      if (key === 'dropout') return `<td class="num">${{fmtNumber(value)}}</td>`;
      if (deltaKeys.has(key)) return `<td class="num">${{fmtDelta(value)}}</td>`;
      if (metricKeys.has(key)) return `<td class="num" style="${{metricColor(value)}}">${{fmtPercent(value)}}</td>`;
      if (typeof value === 'number') return `<td class="num">${{fmtNumber(value, Number.isInteger(value) ? 0 : 2)}}</td>`;
      if (String(key).includes('path') || key === 'log_path' || key === 'folder') return `<td class="mono">${{escapeHtml(value ?? '')}}</td>`;
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

    function quickMatches(row) {{
      const experimentValues = selectedValues('quickExperimentFilter');
      if (experimentValues.length && !experimentValues.includes(String(row.experiment))) return false;
      const needle = document.getElementById('quickSearchInput').value.trim().toLowerCase();
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
      tbody.innerHTML = rows.map(row => `<tr class="${{row.is_ours ? 'ours-row' : ''}}">${{columns.map(key => compareCell(row, key)).join('')}}</tr>`).join('');
      updateHeaderState(tableId, sortState);
    }}

    function renderComparisonTables() {{
      renderComparisonTable('sysuCompareTable', sysuCompareColumns, state.sysuCompareSort);
      renderComparisonTable('regdbCompareTable', regdbCompareColumns, state.regdbCompareSort);
    }}

    function renderPrecisionSummary() {{
      const target = document.getElementById('precisionSummary');
      if (!target) return;
      const p = DATA.precision_summary || {{}};
      const statusClass = p.complete ? 'complete' : 'incomplete';
      const rows = [
        ['AMP quick baseline', p.amp_mean_best_rank1 === null ? '' : fmtPercent(p.amp_mean_best_rank1), `mAP ${{fmtPercent(p.amp_mean_best_map)}} · ${{p.amp_complete_trials ?? 0}}/${{p.expected_trials ?? 0}} trials`],
        ['FP32 quick baseline', p.fp32_mean_best_rank1 === null ? '' : fmtPercent(p.fp32_mean_best_rank1), `mAP ${{fmtPercent(p.fp32_mean_best_map)}} · ${{p.fp32_complete_trials ?? 0}}/${{p.expected_trials ?? 0}} trials`],
        ['Delta R1', p.delta_rank1 === null || p.delta_rank1 === undefined ? '' : `${{fmtDelta(p.delta_rank1)}} pp`, 'FP32 minus AMP'],
        ['Delta mAP', p.delta_map === null || p.delta_map === undefined ? '' : `${{fmtDelta(p.delta_map)}} pp`, `threshold ${{fmtNumber(p.threshold_pp ?? 0.5)}} pp`]
      ];
      target.innerHTML = rows.map(([label, metric, sub]) => `
        <article class="precision-card">
          <div class="label">${{escapeHtml(label)}}</div>
          <div class="metric">${{escapeHtml(metric || 'n/a')}}</div>
          <div class="sub">${{escapeHtml(sub || '')}}</div>
        </article>
      `).join('') + `
        <div class="precision-callout ${{statusClass}}">
          <strong>Status:</strong> ${{escapeHtml(p.status ?? 'unknown')}}.
          <strong>Conclusion:</strong> ${{escapeHtml(p.conclusion ?? '')}}
        </div>
      `;
    }}

    function renderQuickAblationTable() {{
      const tbody = document.querySelector('#quickAblationTable tbody');
      const rows = sortRows(DATA.quick_aggregate_rows.filter(quickMatches), state.quickAblationSort);
      tbody.innerHTML = rows.map(row => `<tr>${{quickAblationColumns.map(key => quickCell(row, key)).join('')}}</tr>`).join('');
      updateHeaderState('quickAblationTable', state.quickAblationSort);
    }}

    function renderQuickTrialTable() {{
      const tbody = document.querySelector('#quickTrialTable tbody');
      const rows = sortRows(DATA.quick_rows.filter(quickMatches), state.quickTrialSort);
      tbody.innerHTML = rows.map(row => `<tr>${{quickTrialColumns.map(key => quickCell(row, key)).join('')}}</tr>`).join('');
      updateHeaderState('quickTrialTable', state.quickTrialSort);
    }}

    function renderQuickTables() {{
      renderQuickAblationTable();
      renderQuickTrialTable();
    }}

    function renderCards() {{
      const validation = DATA.validation;
      const completeRuns = validation.complete_run_count;
      const expectedRuns = validation.expected_run_count;
      const primaryComplete = validation.primary_full_complete_run_count;
      const primaryExpected = validation.primary_full_expected_run_count;
      const quickComplete = validation.quick_complete_run_count;
      const quickExpected = validation.quick_expected_run_count;
      const maxR1 = validation.stage2_best_rank1_max;
      const meanR1 = validation.stage2_best_rank1_mean;
      const meanMap = validation.stage2_best_map_mean;
      const cards = [
        ['Report Status', validation.report_ok ? 'complete' : 'incomplete', validation.primary_full_ok ? 'full AMP runs complete' : 'waiting for full AMP runs'],
        ['Full AMP Runs', `${{primaryComplete}} / ${{primaryExpected}}`, 'paper-parameter baseline, MDUE, and MDUE+CGCF'],
        ['Legacy Runs', `${{completeRuns}} / ${{expectedRuns}}`, validation.legacy_ok ? 'all expected legacy logs complete' : 'legacy logs may be missing'],
        ['Stage2 Best R1', maxR1 === null ? '' : fmtPercent(maxR1), 'max over completed stage2 trials'],
        ['Stage2 Mean R1', meanR1 === null ? '' : fmtPercent(meanR1), `${{validation.complete_stage2_count}} completed stage2 trials`],
        ['Stage2 Mean mAP', meanMap === null ? '' : fmtPercent(meanMap), 'mean max mAP over completed stage2 trials'],
        ['Quick Ablations', `${{quickComplete}} / ${{quickExpected}}`, 'trial 1-3 ablation runs complete']
      ];
      document.getElementById('cards').innerHTML = cards.map(([label, value, note]) => `<article class="card"><div class="label">${{escapeHtml(label)}}</div><div class="value">${{escapeHtml(value || 'n/a')}}</div><div class="note">${{escapeHtml(note)}}</div></article>`).join('');
    }}

    function renderValidation() {{
      const validation = DATA.validation;
      const missing = validation.missing_logs.length ? validation.missing_logs.join(', ') : 'none';
      const incomplete = validation.incomplete_runs.length ? validation.incomplete_runs.join(', ') : 'none';
      const manifest = DATA.manifest;
      document.getElementById('validation').innerHTML = [
        `<div><strong>Report status:</strong> ${{validation.report_ok ? 'complete' : 'incomplete'}}</div>`,
        `<div><strong>Primary full AMP runs:</strong> ${{validation.primary_full_complete_run_count}} / ${{validation.primary_full_expected_run_count}}</div>`,
        `<div><strong>Primary full AMP configs:</strong> ${{validation.primary_full_complete_config_count}} / ${{validation.primary_full_expected_config_count}}</div>`,
        `<div><strong>Legacy status:</strong> ${{validation.legacy_ok ? 'complete' : 'incomplete'}}</div>`,
        `<div><strong>Summary rows:</strong> ${{validation.actual_run_count}} / ${{validation.expected_run_count}}</div>`,
        `<div><strong>Epoch rows:</strong> ${{validation.epoch_row_count}}</div>`,
        `<div><strong>Quick ablation runs:</strong> ${{validation.quick_complete_run_count}} / ${{validation.quick_expected_run_count}}</div>`,
        `<div><strong>Quick ablation configs:</strong> ${{validation.quick_complete_config_count}} / ${{validation.quick_expected_config_count}}</div>`,
        `<div><strong>Precision check:</strong> ${{escapeHtml(validation.precision_check_status)}} (complete=${{validation.precision_check_complete ? 'yes' : 'no'}}, ΔR1=${{fmtDelta(validation.precision_delta_rank1)}} pp, ΔmAP=${{fmtDelta(validation.precision_delta_map)}} pp)</div>`,
        `<div><strong>Missing logs:</strong> <span class="mono">${{escapeHtml(missing)}}</span></div>`,
        `<div><strong>Incomplete runs:</strong> <span class="mono">${{escapeHtml(incomplete)}}</span></div>`,
        `<div><strong>Metric policy:</strong> ${{escapeHtml(manifest.metric_policy)}}</div>`,
        `<div><strong>Repo:</strong> <span class="mono">${{escapeHtml(manifest.repo_path)}}</span></div>`
      ].join('');
    }}

    function renderAll() {{
      renderCards();
      renderComparisonTables();
      renderPrecisionSummary();
      renderQuickTables();
      renderSummaryTable();
      renderEpochTable();
      renderValidation();
    }}

    function setSort(target, key) {{
      const sortState = {{
        summary: state.summarySort,
        epoch: state.epochSort,
        sysuCompare: state.sysuCompareSort,
        regdbCompare: state.regdbCompareSort,
        quickAblation: state.quickAblationSort,
        quickTrial: state.quickTrialSort
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
      document.getElementById('quickExperimentFilter').innerHTML = optionHtml(uniqueValues(DATA.quick_aggregate_rows, 'experiment'));
      document.getElementById('trialFilter').addEventListener('change', renderAll);
      document.getElementById('stageFilter').addEventListener('change', renderAll);
      document.getElementById('searchInput').addEventListener('input', renderAll);
      document.getElementById('compareTypeFilter').addEventListener('change', renderAll);
      document.getElementById('compareSearchInput').addEventListener('input', renderAll);
      document.getElementById('quickExperimentFilter').addEventListener('change', renderAll);
      document.getElementById('quickSearchInput').addEventListener('input', renderAll);
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
      document.getElementById('clearQuickFilters').addEventListener('click', () => {{
        document.getElementById('quickExperimentFilter').selectedIndex = -1;
        document.getElementById('quickSearchInput').value = '';
        renderAll();
      }});
      document.querySelectorAll('#summaryTable th[data-key]').forEach(th => th.addEventListener('click', () => setSort('summary', th.dataset.key)));
      document.querySelectorAll('#epochTable th[data-key]').forEach(th => th.addEventListener('click', () => setSort('epoch', th.dataset.key)));
      document.querySelectorAll('#sysuCompareTable th[data-key]').forEach(th => th.addEventListener('click', () => setSort('sysuCompare', th.dataset.key)));
      document.querySelectorAll('#regdbCompareTable th[data-key]').forEach(th => th.addEventListener('click', () => setSort('regdbCompare', th.dataset.key)));
      document.querySelectorAll('#quickAblationTable th[data-key]').forEach(th => th.addEventListener('click', () => setSort('quickAblation', th.dataset.key)));
      document.querySelectorAll('#quickTrialTable th[data-key]').forEach(th => th.addEventListener('click', () => setSort('quickTrial', th.dataset.key)));
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
        "complete_runs={complete}/{expected} primary_full={primary_complete}/{primary_expected} "
        "epoch_rows={epochs} report_ok={report_ok} legacy_ok={legacy_ok}".format(
            complete=payload["validation"]["complete_run_count"],
            expected=payload["validation"]["expected_run_count"],
            primary_complete=payload["validation"]["primary_full_complete_run_count"],
            primary_expected=payload["validation"]["primary_full_expected_run_count"],
            epochs=payload["validation"]["epoch_row_count"],
            report_ok=payload["validation"]["report_ok"],
            legacy_ok=payload["validation"]["ok"],
        )
    )


if __name__ == "__main__":
    main()
