"""
Dataset assembly — consolidates individual experiment results into a
training-ready dataset.

Each experiment produces a result.json file with metadata, VMAF scores,
and references to .npy files containing traffic features. This module
scans all results and produces a single CSV for easy loading into
PyTorch/TensorFlow data pipelines.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_dataset(output_dir: Path, dataset_dir: Path) -> Path:
    """
    Scan all experiment results and build a consolidated dataset CSV.

    Expected directory structure under output_dir:
        output/
          L0_D0_J0_BW0_R0/
            result.json
            L0_D0_J0_BW0_R0_packet_sizes.npy
            L0_D0_J0_BW0_R0_inter_packet_times.npy
          L5_D50_J0_BW0_R0/
            result.json
            ...

    Output:
        dataset_dir/dataset.csv with columns:
          experiment_id, loss_percent, delay_ms, jitter_ms, bandwidth_kbps,
          repeat, mean_vmaf, frame_count, packet_count, traffic_duration_sec,
          packet_sizes_file, inter_packet_times_file, packet_timestamps_file,
          per_frame_vmaf_file, frame_times_file, recording_file, pcap_file

    Returns:
        Path to the generated CSV file.
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Find all result.json files
    result_files = sorted(output_dir.glob("*/result.json"))
    if not result_files:
        logger.warning(f"No experiment results found in {output_dir}")
        return dataset_dir / "dataset.csv"

    records = []
    for rf in result_files:
        try:
            with open(rf) as f:
                data = json.load(f)
            records.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Skipping corrupt result: {rf} ({e})")

    df = pd.DataFrame(records)

    # Sort by network condition for readability
    sort_cols = [c for c in ["loss_percent", "delay_ms", "jitter_ms", "repeat"]
                 if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    csv_path = dataset_dir / "dataset.csv"
    df.to_csv(csv_path, index=False)

    # Print summary statistics
    logger.info(f"Dataset built: {csv_path}")
    logger.info(f"  Experiments : {len(df)}")
    if "mean_vmaf" in df.columns and len(df) > 0:
        logger.info(f"  VMAF range  : {df['mean_vmaf'].min():.1f} - {df['mean_vmaf'].max():.1f}")
        logger.info(f"  VMAF mean   : {df['mean_vmaf'].mean():.1f}")
    if "loss_percent" in df.columns:
        conditions = df.groupby(
            ["loss_percent", "delay_ms", "jitter_ms"]
        ).ngroups
        logger.info(f"  Unique conditions: {conditions}")

    return csv_path


def load_experiment(dataset_csv: Path, experiment_id: str) -> dict:
    """
    Load features and label for a single experiment.

    Useful for model training data loaders.

    Returns:
        Dictionary with:
          - packet_sizes: np.ndarray (N,)
          - inter_packet_times: np.ndarray (N,)
          - packet_timestamps: np.ndarray (N,) — seconds from first packet
          - per_frame_vmaf: np.ndarray (F,) — VMAF per content frame
          - frame_times: np.ndarray (F,) — seconds from content start (i / fps)
          - mean_vmaf: float
          - metadata: dict (all CSV columns)
    """
    df = pd.read_csv(dataset_csv)
    row = df[df["experiment_id"] == experiment_id]
    if row.empty:
        raise ValueError(f"Experiment not found: {experiment_id}")

    row = row.iloc[0]
    result = {
        "packet_sizes": np.load(row["packet_sizes_file"]),
        "inter_packet_times": np.load(row["inter_packet_times_file"]),
        "mean_vmaf": float(row["mean_vmaf"]),
        "metadata": row.to_dict(),
    }

    # Load temporal arrays (available for experiments run after this feature)
    for key, col in [
        ("packet_timestamps", "packet_timestamps_file"),
        ("per_frame_vmaf", "per_frame_vmaf_file"),
        ("frame_times", "frame_times_file"),
    ]:
        if col in row and pd.notna(row[col]):
            result[key] = np.load(row[col])

    return result


def dataset_summary(dataset_csv: Path) -> str:
    """Print a human-readable summary of the dataset."""
    df = pd.read_csv(dataset_csv)

    lines = [
        f"Dataset: {dataset_csv}",
        f"Total experiments: {len(df)}",
        "",
    ]

    if len(df) == 0:
        return "\n".join(lines + ["(empty)"])

    lines.append("VMAF distribution:")
    lines.append(f"  Min    : {df['mean_vmaf'].min():.1f}")
    lines.append(f"  Mean   : {df['mean_vmaf'].mean():.1f}")
    lines.append(f"  Median : {df['mean_vmaf'].median():.1f}")
    lines.append(f"  Max    : {df['mean_vmaf'].max():.1f}")
    lines.append(f"  Std    : {df['mean_vmaf'].std():.1f}")
    lines.append("")

    lines.append("Breakdown by packet loss:")
    for loss, group in df.groupby("loss_percent"):
        vmaf_mean = group["mean_vmaf"].mean()
        lines.append(f"  loss={loss:5.1f}%  n={len(group):3d}  VMAF={vmaf_mean:.1f}")

    return "\n".join(lines)
