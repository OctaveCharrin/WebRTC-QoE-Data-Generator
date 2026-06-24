"""
Continuous QoS -> VMAF surrogate fitted over the experiment grid.

The Docker pipeline measures VMAF only at the discrete grid points
``(bitrate_kbps, loss_pct, delay_ms, jitter_ms)``. An RL reward needs a
function defined *everywhere* in that box, so this module:

  1. loads ``output/dataset/dataset.csv``,
  2. averages ``mean_vmaf`` across repeats per unique condition,
  3. fits a multilinear interpolant ``f(bitrate, loss, delay, jitter) -> vmaf``
     over the regular grid (full factorial of the swept axes),
  4. serializes the fitted model to portable artifacts.

Multilinear interpolation on the regular grid is robust and monotonicity-
preserving (no RBF/polynomial oscillation between grid planes), and is trivial
to evaluate in pure NumPy — so ``reward/qos_vmaf_reward.py`` consumes the
serialized model with numpy as its only dependency. Sparse/failed cells in the
grid are nearest-filled at fit time. Out-of-grid inputs are clamped to the
measured box at predict time, so the reward saturates at the nearest measured
boundary instead of extrapolating.

Artifacts written by :func:`build_surrogate`:
  - ``output/reward_model.npz``  — fitted RBF (arrays; the portable model)
  - ``output/reward_model.pkl``  — same model as a pickled dict (convenience)
  - ``output/reward_grid.npz``   — raw averaged grid (conditions, vmaf, counts)
  - ``reward/reward_model.npz``  — copy next to the reward module so the
                                   ``reward/`` package is self-contained to copy.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Order is the reward contract: [bitrate_kbps, loss_pct, delay_ms, jitter_ms].
FEATURE_COLUMNS = ["bitrate_kbps", "loss_pct", "delay_ms", "jitter_ms"]
# How those map onto dataset.csv column names.
_DATASET_COLUMNS = {
    "bitrate_kbps": "bitrate_kbps",
    "loss_pct": "loss_percent",
    "delay_ms": "delay_ms",
    "jitter_ms": "jitter_ms",
}


# ---------------------------------------------------------------------------
# Pure-numpy gridded multilinear interpolation
# ---------------------------------------------------------------------------

def _predict(params: dict, X: np.ndarray) -> np.ndarray:
    """Evaluate the multilinear interpolant at rows of X (m, 4).

    Each input is clamped to its axis range (saturating at the measured box),
    then blended from the 2^d surrounding grid corners.
    """
    axes = params["axes"]
    table = params["table"]
    d = len(axes)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    m = X.shape[0]

    lo = np.zeros((m, d), dtype=np.intp)   # lower corner index per dim
    frac = np.zeros((m, d), dtype=np.float64)  # interpolation weight per dim
    has_upper = np.zeros(d, dtype=bool)    # whether dim has >1 level
    for j, axis in enumerate(axes):
        x = np.clip(X[:, j], axis[0], axis[-1])
        if axis.size < 2:
            lo[:, j] = 0
            frac[:, j] = 0.0
            continue
        has_upper[j] = True
        idx = np.searchsorted(axis, x, side="right") - 1
        idx = np.clip(idx, 0, axis.size - 2)
        lo[:, j] = idx
        frac[:, j] = (x - axis[idx]) / (axis[idx + 1] - axis[idx])

    out = np.zeros(m, dtype=np.float64)
    for corner in range(1 << d):
        w = np.ones(m, dtype=np.float64)
        index = []
        valid = True
        for j in range(d):
            bit = (corner >> j) & 1
            if bit and not has_upper[j]:
                valid = False
                break
            w *= frac[:, j] if bit else (1.0 - frac[:, j])
            index.append(lo[:, j] + bit)
        if not valid:
            continue
        out += w * table[tuple(index)]
    return np.clip(out, 0.0, 100.0)


def _fill_missing(table: np.ndarray) -> np.ndarray:
    """Nearest-neighbor fill of NaN cells (failed/sparse grid combinations)."""
    if not np.isnan(table).any():
        return table
    filled = table.copy()
    known = np.array(np.where(~np.isnan(table))).T  # (k, d) indices
    if known.size == 0:
        raise ValueError("Grid is entirely empty after averaging")
    known_vals = table[tuple(known.T)]
    missing = np.array(np.where(np.isnan(table))).T
    n_missing = len(missing)
    logger.warning(f"Filling {n_missing} missing grid cell(s) by nearest neighbor")
    for cell in missing:
        dist2 = ((known - cell) ** 2).sum(axis=1)
        filled[tuple(cell)] = known_vals[int(dist2.argmin())]
    return filled


def fit_grid(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Fit a multilinear interpolant over the regular grid spanned by X.

    Args:
        X: (n, 4) feature matrix in FEATURE_COLUMNS order (the averaged grid).
        y: (n,) target VMAF values.

    Returns:
        A params dict (axes + value table) consumed by :func:`_predict` and
        serialized for the reward module.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"Need >=2 unique conditions to fit a surrogate, got {n}")

    d = X.shape[1]
    axes = [np.unique(X[:, j]) for j in range(d)]
    shape = tuple(axis.size for axis in axes)

    table = np.full(shape, np.nan, dtype=np.float64)
    for row in range(n):
        cell = tuple(int(np.searchsorted(axes[j], X[row, j])) for j in range(d))
        table[cell] = y[row]
    table = _fill_missing(table)

    params = {
        "method": "multilinear",
        "columns": list(FEATURE_COLUMNS),
        "axes": axes,
        "table": table,
        "bounds_min": np.array([axis[0] for axis in axes]),
        "bounds_max": np.array([axis[-1] for axis in axes]),
    }

    # Sanity check: at grid points the interpolant must return the table value.
    resid = _predict(params, X) - y
    logger.info(
        f"Fitted multilinear grid {shape}: train RMSE={np.sqrt((resid ** 2).mean()):.3f}, "
        f"max|err|={np.abs(resid).max():.3f} VMAF (should be ~0 at grid nodes)"
    )
    return params


# ---------------------------------------------------------------------------
# Grid averaging + serialization
# ---------------------------------------------------------------------------

def average_grid(dataset_csv: Path, vmaf_column: str = "mean_vmaf"):
    """
    Load dataset.csv and average the VMAF target across repeats per condition.

    Falls back to ``mean_vmaf_masked`` if the requested column is missing/empty.

    Returns:
        (X, y, counts) where X is (n, 4) in FEATURE_COLUMNS order, y is (n,)
        mean VMAF, and counts is (n,) the number of repeats averaged.
    """
    df = pd.read_csv(dataset_csv)
    if df.empty:
        raise ValueError(f"Dataset is empty: {dataset_csv}")

    if vmaf_column not in df.columns or df[vmaf_column].notna().sum() == 0:
        fallback = "mean_vmaf_masked"
        if fallback in df.columns and df[fallback].notna().sum() > 0:
            logger.warning(f"'{vmaf_column}' unavailable; using '{fallback}'")
            vmaf_column = fallback
        else:
            raise ValueError(f"No usable VMAF column ('{vmaf_column}'/'{fallback}')")

    missing = [c for c in _DATASET_COLUMNS.values() if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing condition columns {missing}. Re-run the grid with "
            f"the bitrate sweep (see README)."
        )

    df = df.dropna(subset=[vmaf_column])
    group_cols = [_DATASET_COLUMNS[f] for f in FEATURE_COLUMNS]
    grouped = (
        df.groupby(group_cols)[vmaf_column]
        .agg(["mean", "count"])
        .reset_index()
    )

    X = grouped[group_cols].to_numpy(dtype=np.float64)
    y = grouped["mean"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.int64)
    logger.info(
        f"Averaged {len(df)} runs into {len(X)} unique conditions "
        f"(target='{vmaf_column}')"
    )
    return X, y, counts


def _save_npz(path: Path, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Axes are ragged (different lengths per dim), so store each separately.
    axis_arrays = {f"axis_{j}": axis for j, axis in enumerate(params["axes"])}
    np.savez(
        path,
        method=np.array(params["method"]),
        columns=np.array(params["columns"]),
        n_dims=np.array(len(params["axes"])),
        table=params["table"],
        bounds_min=params["bounds_min"],
        bounds_max=params["bounds_max"],
        **axis_arrays,
    )


def build_surrogate(
    dataset_csv: Path,
    output_dir: Path,
    reward_dir: Path,
    vmaf_column: str = "mean_vmaf",
) -> dict:
    """
    Fit the surrogate from dataset.csv and write all artifacts.

    Returns the fitted params dict.
    """
    X, y, counts = average_grid(dataset_csv, vmaf_column=vmaf_column)
    params = fit_grid(X, y)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_npz = output_dir / "reward_model.npz"
    model_pkl = output_dir / "reward_model.pkl"
    grid_npz = output_dir / "reward_grid.npz"

    _save_npz(model_npz, params)
    with open(model_pkl, "wb") as fh:
        pickle.dump(params, fh)
    np.savez(
        grid_npz,
        columns=np.array(FEATURE_COLUMNS),
        conditions=X,
        vmaf=y,
        counts=counts,
    )

    # Portable copy next to the reward module so reward/ is self-contained.
    reward_dir.mkdir(parents=True, exist_ok=True)
    _save_npz(reward_dir / "reward_model.npz", params)

    logger.info(f"Wrote surrogate model -> {model_npz}")
    logger.info(f"Wrote surrogate model -> {model_pkl}")
    logger.info(f"Wrote raw averaged grid -> {grid_npz}")
    logger.info(f"Wrote portable copy -> {reward_dir / 'reward_model.npz'}")
    return params


# ---------------------------------------------------------------------------
# Discrimination report (does VMAF actually vary?)
# ---------------------------------------------------------------------------

def reward_surface_report(
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    narrow_threshold: float = 10.0,
) -> str:
    """
    Render the fitted VMAF surface across bitrate x loss and flag flat surfaces.

    Delay and jitter are held at the median of the measured grid. Returns a
    printable multi-line report; appends a WARNING when the measured VMAF range
    spans less than ``narrow_threshold`` points (WebRTC resilience can keep VMAF
    flat, which would make the reward non-discriminative).
    """
    bitrates = np.unique(X[:, 0])
    losses = np.unique(X[:, 1])
    delay_med = float(np.median(X[:, 2]))
    jitter_med = float(np.median(X[:, 3]))

    lines = [
        "VMAF surface  f(bitrate, loss)  "
        f"@ delay={delay_med:.0f}ms, jitter={jitter_med:.0f}ms",
        "(surrogate evaluated on the measured bitrate/loss axes)",
        "",
    ]
    header = "bitrate\\loss |" + "".join(f"{l:8.1f}%" for l in losses)
    lines.append(header)
    lines.append("-" * len(header))
    for br in bitrates:
        query = np.array([[br, l, delay_med, jitter_med] for l in losses])
        row = _predict(params, query)
        lines.append(
            f"{br:8.0f}kbps |" + "".join(f"{v:9.1f}" for v in row)
        )

    # Discrimination metrics on the *measured* averaged grid.
    vmaf_range = float(y.max() - y.min())
    lines.append("")
    lines.append(f"Measured VMAF: min={y.min():.1f} max={y.max():.1f} "
                 f"range={vmaf_range:.1f} (n={len(y)} conditions)")

    # Per-axis spread: how much does VMAF move along each input alone?
    for j, name in enumerate(FEATURE_COLUMNS):
        vals = np.unique(X[:, j])
        if len(vals) < 2:
            lines.append(f"  {name:13s}: single value ({vals[0]:.0f}) — not swept")
            continue
        means = [y[X[:, j] == v].mean() for v in vals]
        lines.append(
            f"  {name:13s}: VMAF {min(means):.1f}..{max(means):.1f} "
            f"(spread {max(means) - min(means):.1f}) over {len(vals)} levels"
        )

    if vmaf_range < narrow_threshold:
        lines.append("")
        lines.append(
            f"WARNING: VMAF varies by only {vmaf_range:.1f} points (< "
            f"{narrow_threshold:.0f}). The reward may be weakly discriminative — "
            f"consider lower bitrates / higher loss, longer durations, or "
            f"disabling WebRTC error resilience to widen the dynamic range."
        )
    return "\n".join(lines)
