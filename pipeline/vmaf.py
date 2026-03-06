from __future__ import annotations

"""
VMAF computation with padding detection and frame alignment.

Pipeline for each received video:
  1. Convert received WebM to Y4M (normalize resolution/fps)
  2. Detect and remove testsrc padding frames (find content boundaries)
  3. Compute VMAF comparing the trimmed received video against the reference

The padding detection replaces the complex ImageMagick pixel-sampling
approach from the original calculate_qoe_metrics.sh with a simpler
FFmpeg + numpy approach that samples a few pixels and checks for the
known testsrc color bar pattern.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def check_ffmpeg_vmaf() -> bool:
    """Check if FFmpeg has libvmaf support."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True, text=True, check=True,
        )
        return "libvmaf" in result.stdout
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Padding detection
# ---------------------------------------------------------------------------

# The testsrc pattern from FFmpeg produces color bars. At y = height/3,
# specific x positions have known RGB values. These match the values used
# by the original project (calculate_qoe_metrics.sh lines 206-230).
# Positions are given as fractions of the video width.
TESTSRC_COLORS = [
    # (x_fraction, R, G, B)
    (0.1875, 0, 255, 255),    # cyan
    (0.3125, 255, 0, 253),    # purple/magenta
    (0.4375, 0, 0, 253),      # blue
    (0.5625, 253, 255, 0),    # yellow
    (0.6875, 0, 255, 0),      # green
    (0.8125, 253, 0, 0),      # red
]


def _extract_frame_rgb(video_path: Path, frame_number: int,
                       width: int, height: int) -> np.ndarray | None:
    """
    Extract a single frame as an RGB numpy array using FFmpeg.

    Returns ndarray of shape (height, width, 3) or None on failure.
    """
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"select=eq(n\\,{frame_number})",
        "-vframes", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or len(result.stdout) != width * height * 3:
        return None
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)


def _is_padding_frame(frame_rgb: np.ndarray, width: int, height: int,
                      threshold: int = 50) -> bool:
    """
    Check if a frame matches the testsrc color bar pattern.

    Samples pixel colors at known positions and compares against expected
    values with a tolerance threshold. All sample points must match.
    """
    sample_y = height // 3  # Middle of the color bar band

    for x_frac, exp_r, exp_g, exp_b in TESTSRC_COLORS:
        sample_x = int(x_frac * width)
        # Clamp to valid range
        sample_x = min(sample_x, width - 1)
        sample_y_clamped = min(sample_y, height - 1)

        pixel = frame_rgb[sample_y_clamped, sample_x]
        r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])

        if (abs(r - exp_r) > threshold or
                abs(g - exp_g) > threshold or
                abs(b - exp_b) > threshold):
            return False

    return True


def _get_frame_count(video_path: Path) -> int:
    """Get the total number of frames in a video using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


def detect_padding_boundaries(
    video_path: Path, width: int, height: int, fps: int,
    padding_duration_sec: int = 5, threshold: int = 50,
) -> tuple[int, int]:
    """
    Find the first and last content frame in a video with padding.

    The video has the structure:
      [testsrc padding] [content with frame numbers] [testsrc padding]

    We scan forward from the start to find where padding ends (content starts),
    and backward from the end to find where content ends (padding starts).

    Instead of checking every frame (slow), we use the known padding duration
    as a hint and do a binary-search-like scan around the expected boundaries.

    Args:
        video_path: Path to the normalized Y4M video.
        width, height: Video dimensions.
        fps: Frame rate.
        padding_duration_sec: Expected padding duration (used as hint).
        threshold: RGB matching threshold per channel.

    Returns:
        (first_content_frame, last_content_frame) — 0-indexed frame numbers.
    """
    total_frames = _get_frame_count(video_path)
    expected_padding_frames = padding_duration_sec * fps

    logger.info(
        f"Detecting padding in {total_frames} frames "
        f"(expected ~{expected_padding_frames} padding frames per side)..."
    )

    # --- Find start of content (scan forward from expected boundary) ---
    # Start searching around where we expect the transition
    search_start = max(0, expected_padding_frames - fps)  # 1 second before expected
    search_end = min(total_frames, expected_padding_frames + fps)  # 1 second after

    first_content = 0
    for i in range(search_start, search_end):
        frame = _extract_frame_rgb(video_path, i, width, height)
        if frame is None:
            continue
        if not _is_padding_frame(frame, width, height, threshold):
            first_content = i
            logger.info(f"  Content starts at frame {i} (expected ~{expected_padding_frames})")
            break
    else:
        # Fallback: scan from beginning
        logger.warning("  Could not find content start near expected position, scanning from 0...")
        for i in range(0, min(total_frames, expected_padding_frames * 3)):
            frame = _extract_frame_rgb(video_path, i, width, height)
            if frame is None:
                continue
            if not _is_padding_frame(frame, width, height, threshold):
                first_content = i
                logger.info(f"  Content starts at frame {i}")
                break

    # --- Find end of content (scan backward from expected boundary) ---
    end_padding_start = total_frames - expected_padding_frames
    search_start_rev = min(total_frames - 1, end_padding_start + fps)
    search_end_rev = max(0, end_padding_start - fps)

    last_content = total_frames - 1
    for i in range(search_start_rev, search_end_rev, -1):
        frame = _extract_frame_rgb(video_path, i, width, height)
        if frame is None:
            continue
        if not _is_padding_frame(frame, width, height, threshold):
            last_content = i
            logger.info(f"  Content ends at frame {i} (expected ~{end_padding_start})")
            break
    else:
        # Fallback: scan from end
        logger.warning("  Could not find content end near expected position, scanning from end...")
        for i in range(total_frames - 1, max(0, total_frames - expected_padding_frames * 3), -1):
            frame = _extract_frame_rgb(video_path, i, width, height)
            if frame is None:
                continue
            if not _is_padding_frame(frame, width, height, threshold):
                last_content = i
                logger.info(f"  Content ends at frame {i}")
                break

    content_frames = last_content - first_content + 1
    logger.info(
        f"  Padding detection result: frames {first_content}-{last_content} "
        f"({content_frames} content frames, "
        f"{content_frames / fps:.1f}s)"
    )
    return first_content, last_content


# ---------------------------------------------------------------------------
# VMAF computation
# ---------------------------------------------------------------------------

def compute_vmaf(
    received_video: Path,
    reference_video: Path,
    output_json: Path,
    width: int = 640,
    height: int = 480,
    fps: int = 24,
    padding_duration_sec: int = 5,
    padding_threshold: int = 50,
) -> dict:
    """
    Compute VMAF score with automatic padding removal and alignment.

    Steps:
      1. Convert received WebM to Y4M (normalize resolution/fps)
      2. Detect padding boundaries in the received video
      3. Trim the received video to content-only region
      4. Compute VMAF comparing trimmed received vs reference

    The reference video should already be content-only (no padding).

    Args:
        received_video: Path to the recorded WebM from the receiver.
        reference_video: Path to the reference Y4M (content only, no padding).
        output_json: Where to write the VMAF JSON results.
        width, height: Target dimensions for comparison.
        fps: Target frame rate for comparison.
        padding_duration_sec: Expected padding duration per side (hint for detection).
        padding_threshold: RGB threshold for padding color matching.

    Returns:
        Dictionary with:
          - mean_vmaf:       float (0-100)
          - per_frame_vmaf:  list[float]
          - frame_count:     int
          - content_start_frame: int (first content frame in received video)
          - content_end_frame:   int (last content frame in received video)
    """
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Convert received WebM to Y4M ---
    received_y4m = output_json.with_suffix(".received.y4m")

    logger.info(f"Converting received video to {width}x{height}@{fps}fps...")
    convert_cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", str(received_video),
        "-vf", f"scale={width}:{height},fps={fps}",
        "-pix_fmt", "yuv420p",
        str(received_y4m),
    ]
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg convert failed: {result.stderr}")
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

    # --- Step 2: Detect padding boundaries ---
    first_content, last_content = detect_padding_boundaries(
        received_y4m, width, height, fps,
        padding_duration_sec=padding_duration_sec,
        threshold=padding_threshold,
    )

    # --- Step 3: Trim to content-only ---
    trimmed_y4m = output_json.with_suffix(".trimmed.y4m")
    start_time = first_content / fps
    content_duration = (last_content - first_content + 1) / fps

    logger.info(
        f"Trimming received video: start={start_time:.2f}s, "
        f"duration={content_duration:.2f}s"
    )
    trim_cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", str(received_y4m),
        "-ss", f"{start_time:.4f}",
        "-t", f"{content_duration:.4f}",
        "-pix_fmt", "yuv420p",
        str(trimmed_y4m),
    ]
    result = subprocess.run(trim_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg trim failed: {result.stderr}")
        raise RuntimeError(f"FFmpeg trim failed: {result.stderr}")

    # --- Step 4: Compute VMAF ---
    # The reference is content-only. The trimmed received is now also content-only.
    # FFmpeg's libvmaf will compare frame-by-frame in order.
    # If the received video has fewer frames (due to drops), libvmaf compares
    # up to the shorter of the two.
    logger.info("Computing VMAF (trimmed received vs reference)...")
    vmaf_cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", str(trimmed_y4m),        # distorted (received, trimmed)
        "-i", str(reference_video),     # reference (content only)
        "-lavfi",
        f"libvmaf=log_path={output_json}:log_fmt=json",
        "-f", "null", "-",
    ]
    result = subprocess.run(vmaf_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"VMAF computation failed: {result.stderr}")
        raise RuntimeError(f"VMAF computation failed: {result.stderr}")

    # --- Step 5: Parse results ---
    with open(output_json) as f:
        vmaf_data = json.load(f)

    per_frame = [frame["metrics"]["vmaf"] for frame in vmaf_data["frames"]]
    mean_vmaf = vmaf_data["pooled_metrics"]["vmaf"]["mean"]

    # Clean up intermediate files
    received_y4m.unlink(missing_ok=True)
    trimmed_y4m.unlink(missing_ok=True)

    logger.info(f"VMAF: mean={mean_vmaf:.2f}, frames={len(per_frame)}")
    return {
        "mean_vmaf": mean_vmaf,
        "per_frame_vmaf": per_frame,
        "frame_count": len(per_frame),
        "content_start_frame": first_content,
        "content_end_frame": last_content,
    }
