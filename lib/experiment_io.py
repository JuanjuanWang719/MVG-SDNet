"""实验目录与结果归档（experiments/<dataset>/）。"""
from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def default_run_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_experiment_dir(
    dataset_name: str,
    folder_dir: str,
    run_tag: str,
    root: str | Path = "experiments",
) -> Path:
    path = Path(root) / dataset_name / f"{folder_dir}_{run_tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run_metadata(
    params_path: str | Path,
    config_path: str,
    metadata: dict[str, Any],
) -> None:
    """保存本次实验的配置副本与元数据 JSON。"""
    params_path = Path(params_path)
    params_path.mkdir(parents=True, exist_ok=True)

    config_dst = params_path / "experiment_config.conf"
    if os.path.isfile(config_path):
        shutil.copy2(config_path, config_dst)

    meta_path = params_path / "experiment_meta.json"
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": os.path.abspath(config_path),
        **metadata,
    }
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_results_summary(params_path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(params_path) / "results_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def append_runs_summary(
    dataset_name: str,
    row: dict[str, Any],
    root: str | Path = "experiments",
) -> Path:
    """在 experiments/<dataset>/runs_summary.csv 追加一行实验记录。"""
    summary_dir = Path(root) / dataset_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "runs_summary.csv"

    row = {**row, "recorded_at": datetime.now().isoformat(timespec="seconds")}
    file_exists = csv_path.is_file() and csv_path.stat().st_size > 0
    fieldnames = list(row.keys())

    if file_exists:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            fieldnames = list(dict.fromkeys([*existing_fields, *fieldnames]))

    rows: list[dict[str, Any]] = []
    if file_exists:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return csv_path
