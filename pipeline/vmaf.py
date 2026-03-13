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

    We scan forward from frame 0 to find where padding ends (content starts),
    and backward from the last frame to find where content ends (padding
    starts again).

    Args:
        video_path: Path to the normalized Y4M video.
        width, height: Video dimensions.
        fps: Frame rate.
        padding_duration_sec: Expected padding duration (used for logging).
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

    # --- Find start of content (scan forward from frame 0) ---
    # Check whether the first frame is padding at all.
    first_content = 0
    first_frame = _extract_frame_rgb(video_path, 0, width, height)
    if first_frame is not None and _is_padding_frame(first_frame, width, height, threshold):
        # There IS padding at the start — find where it ends.
        # Scan forward until we hit a non-padding frame.
        max_scan = min(total_frames, expected_padding_frames * 3)
        for i in range(1, max_scan):
            frame = _extract_frame_rgb(video_path, i, width, height)
            if frame is None:
                continue
            if not _is_padding_frame(frame, width, height, threshold):
                first_content = i
                break
        logger.info(f"  Content starts at frame {first_content}")
    else:
        logger.info("  No padding detected at start, content starts at frame 0")

    # --- Find end of content (scan backward from end) ---
    last_content = total_frames - 1
    last_frame = _extract_frame_rgb(video_path, last_content, width, height)
    if last_frame is not None and _is_padding_frame(last_frame, width, height, threshold):
        # There IS padding at the end — find where it starts.
        min_scan = max(0, total_frames - expected_padding_frames * 3)
        for i in range(total_frames - 2, min_scan, -1):
            frame = _extract_frame_rgb(video_path, i, width, height)
            if frame is None:
                continue
            if not _is_padding_frame(frame, width, height, threshold):
                last_content = i
                break
        logger.info(f"  Content ends at frame {last_content}")
    else:
        logger.info(
            f"  No padding detected at end, content ends at frame {last_content}"
        )

    content_frames = last_content - first_content + 1
    logger.info(
        f"  Padding detection result: frames {first_content}-{last_content} "
        f"({content_frames} content frames, "
        f"{content_frames / fps:.1f}s)"
    )
    return first_content, last_content


# ---------------------------------------------------------------------------
# Frame comparison (alignment debugging)
# ---------------------------------------------------------------------------


def _save_rgb_as_png(rgb_array: np.ndarray, output_path: Path,
                     width: int, height: int) -> None:
    """Save a raw RGB numpy array as a PNG file using FFmpeg."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-i", "-",
        "-f", "image2",
        str(output_path),
    ]
    subprocess.run(cmd, input=rgb_array.tobytes(), check=True)


def generate_frame_comparison(
    trimmed_video: Path,
    reference_video: Path,
    output_dir: Path,
    width: int,
    height: int,
    frame_overlay_crop_height: int = 80,
    per_frame_vmaf: list[float] | None = None,
    mean_vmaf: float | None = None,
    per_frame_vmaf_masked: list[float] | None = None,
    mean_vmaf_masked: float | None = None,
    step: int = 10,
    mask_region: tuple[int, int, int, int] | None = None,
) -> Path:
    """
    Generate debug comparison PNGs with 3 rows per frame.

    Layout per PNG (6 frames total):
      Row 1: Original  — [Reference] | [Received]   (with frame number overlay)
      Row 2: Cropped   — [Reference] | [Received]   (bottom cropped, + VMAF)
      Row 3: Masked    — [Reference] | [Received]   (white box over overlay, + VMAF)
      Bottom: Annotation bar with frame info and mean scores.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    trimmed_count = _get_frame_count(trimmed_video)
    ref_count = _get_frame_count(reference_video)
    frame_count = min(trimmed_count, ref_count)

    separator_w = 4
    row_label_h = 24
    bar_h = 40
    cropped_h = height - frame_overlay_crop_height
    pair_w = width * 2 + separator_w

    # Total composite height: 3 rows with labels + annotation bar
    composite_h = (
        row_label_h + height +           # row 1: original
        row_label_h + cropped_h +         # row 2: cropped
        row_label_h + height +            # row 3: masked
        bar_h                             # annotation bar
    )
    composite_w = pair_w

    generated = 0
    for frame_idx in range(0, frame_count, step):
        ref_frame = _extract_frame_rgb(reference_video, frame_idx, width, height)
        rec_frame = _extract_frame_rgb(trimmed_video, frame_idx, width, height)

        if ref_frame is None or rec_frame is None:
            logger.warning(f"  Could not extract frame {frame_idx}, skipping")
            continue

        sep_full = np.zeros((height, separator_w, 3), dtype=np.uint8)
        sep_cropped = np.zeros((cropped_h, separator_w, 3), dtype=np.uint8)

        # --- Row 1: Original frames ---
        row1_pair = np.concatenate([ref_frame, sep_full, rec_frame], axis=1)

        # --- Row 2: Cropped frames (remove bottom overlay region) ---
        ref_cropped = ref_frame[:cropped_h, :, :]
        rec_cropped = rec_frame[:cropped_h, :, :]
        row2_pair = np.concatenate([ref_cropped, sep_cropped, rec_cropped], axis=1)

        # --- Row 3: Masked frames (white box over overlay region) ---
        ref_masked = ref_frame.copy()
        rec_masked = rec_frame.copy()
        if mask_region is not None:
            # Mask only the overlay box region
            x, y, box_w, box_h = mask_region
            # Clamp to frame bounds
            x1, x2 = max(0, x), min(width, x + box_w)
            y1, y2 = max(0, y), min(height, y + box_h)
            ref_masked[y1:y2, x1:x2, :] = 255
            rec_masked[y1:y2, x1:x2, :] = 255
        else:
            # Fall back to cropping if no mask region provided
            ref_masked[frame_overlay_crop_height:, :, :] = 255
            rec_masked[frame_overlay_crop_height:, :, :] = 255
        row3_pair = np.concatenate([ref_masked, sep_full, rec_masked], axis=1)

        # --- Build row labels (gray background) ---
        def _make_label(text: str, h: int = row_label_h) -> np.ndarray:
            # Gray bar; text will be overlaid by ffmpeg drawtext later
            bar = np.full((h, composite_w, 3), 220, dtype=np.uint8)
            return bar

        # Build VMAF annotation strings for row labels
        crop_vmaf_str = ""
        if per_frame_vmaf is not None and frame_idx < len(per_frame_vmaf):
            crop_vmaf_str = f" - VMAF: {per_frame_vmaf[frame_idx]:.1f}"
        mask_vmaf_str = ""
        if per_frame_vmaf_masked is not None and frame_idx < len(per_frame_vmaf_masked):
            mask_vmaf_str = f" - VMAF: {per_frame_vmaf_masked[frame_idx]:.1f}"

        label1 = _make_label("Original")
        label2 = _make_label(f"Cropped{crop_vmaf_str}")
        label3 = _make_label(f"Masked{mask_vmaf_str}")

        # --- Bottom annotation bar ---
        bottom_bar = np.full((bar_h, composite_w, 3), 255, dtype=np.uint8)

        # --- Assemble composite ---
        composite = np.concatenate([
            label1, row1_pair,
            label2, row2_pair,
            label3, row3_pair,
            bottom_bar,
        ], axis=0)

        png_path = output_dir / f"frame_{frame_idx:04d}.png"
        _save_rgb_as_png(composite, png_path, composite_w, composite_h)

        # --- Overlay text annotations with ffmpeg drawtext ---
        mean_crop_str = f"  Mean cropped\\: {mean_vmaf:.1f}" if mean_vmaf is not None else ""
        mean_mask_str = f"  Mean masked\\: {mean_vmaf_masked:.1f}" if mean_vmaf_masked is not None else ""

        # Escape colons for ffmpeg drawtext
        crop_vmaf_esc = crop_vmaf_str.replace(":", "\\:")
        mask_vmaf_esc = mask_vmaf_str.replace(":", "\\:")

        # Y offsets for each row label
        y_label1 = 4
        y_label2 = row_label_h + height + 4
        y_label3 = row_label_h + height + row_label_h + cropped_h + 4
        y_bottom = composite_h - bar_h + 8

        bottom_text = (
            f"Frame {frame_idx}/{frame_count}"
            f"{mean_crop_str}{mean_mask_str}"
            f"    [Reference (Sender)]  vs  [Received]"
        )

        drawtext_filters = (
            f"drawtext=text='Original':x=10:y={y_label1}:fontsize=16:fontcolor=black,"
            f"drawtext=text='Cropped{crop_vmaf_esc}':x=10:y={y_label2}:fontsize=16:fontcolor=black,"
            f"drawtext=text='Masked{mask_vmaf_esc}':x=10:y={y_label3}:fontsize=16:fontcolor=black,"
            f"drawtext=text='{bottom_text}':x=10:y={y_bottom}:fontsize=18:fontcolor=black"
        )

        annotated_path = output_dir / f"frame_{frame_idx:04d}_tmp.png"
        text_cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", str(png_path),
            "-vf", drawtext_filters,
            str(annotated_path),
        ]
        result = subprocess.run(text_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            annotated_path.rename(png_path)
        else:
            annotated_path.unlink(missing_ok=True)

        generated += 1

    logger.info(
        f"Generated {generated} comparison frames in {output_dir}"
    )
    return output_dir


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
    frame_overlay_crop_height: int = 80,
    padding_duration_sec: int = 5,
    padding_threshold: int = 50,
    debug_dir: Path | None = None,
    debug_frame_step: int = 10,
    vmaf_mode: str = "both",
    mask_region: tuple[int, int, int, int] | None = None,
) -> dict:
    """
    Compute VMAF score with automatic padding removal and alignment.

    Steps:
      1. Convert received WebM to Y4M (normalize resolution/fps)
      2. Detect padding boundaries in the received video
      3. Trim the received video to content-only region
      4. Compute VMAF comparing trimmed received vs reference (one or both modes)

    The reference video should already be content-only (no padding).

    Args:
        received_video: Path to the recorded WebM from the receiver.
        reference_video: Path to the reference Y4M (content only, no padding).
        output_json: Where to write the VMAF JSON results (cropped).
        width, height: Target dimensions for comparison.
        fps: Target frame rate for comparison.
        frame_overlay_crop_height: Pixels to crop from the bottom of each frame
            before VMAF comparison. Removes the burned-in frame number overlay
            so that ±1 frame alignment errors don't penalize VMAF scores.
            Set to 0 to disable cropping.
        padding_duration_sec: Expected padding duration per side (hint for detection).
        padding_threshold: RGB threshold for padding color matching.
        debug_dir: If set, generate side-by-side frame comparison PNGs here.
        debug_frame_step: Extract every Nth frame for debug comparison.
        vmaf_mode: "cropped" (crop overlay), "masked" (white box), or "both" (default).
        mask_region: (x, y, w, h) tuple for the overlay box. If None, falls back to
            cropping-based masking.

    Returns:
        Dictionary with:
          - mean_vmaf:              float — cropped VMAF mean (0-100), or 0.0 if not computed
          - per_frame_vmaf:         list[float] — cropped VMAF per frame (empty list if not computed)
          - mean_vmaf_masked:       float — masked VMAF mean (0-100), or 0.0 if not computed
          - per_frame_vmaf_masked:  list[float] — masked VMAF per frame (empty list if not computed)
          - frame_times:            list[float] — time in seconds (i / fps)
          - frame_count:            int
          - content_start_frame:    int
          - content_end_frame:      int
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

    # --- Step 4: Compute VMAF (one or both methods) ---
    # The reference is content-only. The trimmed received is now also content-only.
    # FFmpeg's libvmaf will compare frame-by-frame in order.
    # If the received video has fewer frames (due to drops), libvmaf compares
    # up to the shorter of the two.
    #
    # We can compute VMAF in up to two modes, excluding the burned-in frame number overlay:
    #   - Cropped: remove the bottom N pixels entirely
    #   - Masked: paint a white box over the overlay region (preserves dimensions)
    # The vmaf_mode parameter controls which to compute.

    def _run_vmaf(lavfi: str, json_path: Path, label: str) -> tuple[list[float], float]:
        """Run a single VMAF computation and return (per_frame, mean)."""
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", str(trimmed_y4m),
            "-i", str(reference_video),
            "-lavfi", lavfi,
            "-f", "null", "-",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error(f"VMAF ({label}) failed: {res.stderr}")
            raise RuntimeError(f"VMAF ({label}) failed: {res.stderr}")
        with open(json_path) as fh:
            data = json.load(fh)
        pf = [frame["metrics"]["vmaf"] for frame in data["frames"]]
        mn = data["pooled_metrics"]["vmaf"]["mean"]
        return pf, mn

    per_frame_cropped = []
    mean_vmaf_cropped = 0.0
    per_frame_masked = []
    mean_vmaf_masked = 0.0

    # Compute cropped VMAF if requested
    if vmaf_mode in ("cropped", "both"):
        logger.info("Computing VMAF with cropped overlay region...")
        if frame_overlay_crop_height > 0:
            crop = f"crop=iw:ih-{frame_overlay_crop_height}:0:0"
            cropped_lavfi = (
                f"[0:v]{crop}[distorted];"
                f"[1:v]{crop}[ref];"
                f"[distorted][ref]libvmaf=model='version=vmaf_v0.6.1\\:phone_model=1'"
                f":log_path={output_json}:log_fmt=json"
            )
        else:
            cropped_lavfi = (
                f"libvmaf=model='version=vmaf_v0.6.1\\:phone_model=1'"
                f":log_path={output_json}:log_fmt=json"
            )
        per_frame_cropped, mean_vmaf_cropped = _run_vmaf(
            cropped_lavfi, output_json, "cropped"
        )

    # Compute masked VMAF if requested
    if vmaf_mode in ("masked", "both"):
        masked_json = output_json.with_name(output_json.stem + "_masked.json")
        logger.info("Computing VMAF with white-box masked overlay region...")
        if frame_overlay_crop_height > 0:
            # If mask_region is provided, create a drawbox filter for just that region
            if mask_region is not None:
                x, y, box_w, box_h = mask_region
                box = (f"drawbox=x={x}:y={y}:w={box_w}:h={box_h}:"
                       f"color=white:t=fill")
            else:
                # Fall back to masking the bottom portion
                box = f"drawbox=x=0:y=ih-{frame_overlay_crop_height}:w=iw:h={frame_overlay_crop_height}:color=white:t=fill"
            masked_lavfi = (
                f"[0:v]{box}[distorted];"
                f"[1:v]{box}[ref];"
                f"[distorted][ref]libvmaf=model='version=vmaf_v0.6.1\\:phone_model=1'"
                f":log_path={masked_json}:log_fmt=json"
            )
        else:
            masked_lavfi = (
                f"libvmaf=model='version=vmaf_v0.6.1\\:phone_model=1'"
                f":log_path={masked_json}:log_fmt=json"
            )
        per_frame_masked, mean_vmaf_masked = _run_vmaf(
            masked_lavfi, masked_json, "masked"
        )
    else:
        masked_json = None

    # --- Step 5: Derive frame times ---
    frame_count = len(per_frame_cropped) if per_frame_cropped else len(per_frame_masked)
    frame_times = [i / fps for i in range(frame_count)]

    # --- Optional: generate debug frame comparison ---
    if debug_dir is not None:
        logger.info("Generating debug frame comparisons...")
        generate_frame_comparison(
            trimmed_video=trimmed_y4m,
            reference_video=reference_video,
            output_dir=debug_dir,
            width=width,
            height=height,
            frame_overlay_crop_height=frame_overlay_crop_height,
            per_frame_vmaf=per_frame_cropped if per_frame_cropped else None,
            mean_vmaf=mean_vmaf_cropped if per_frame_cropped else None,
            per_frame_vmaf_masked=per_frame_masked if per_frame_masked else None,
            mean_vmaf_masked=mean_vmaf_masked if per_frame_masked else None,
            step=debug_frame_step,
            mask_region=mask_region,
        )

    # Clean up intermediate files
    received_y4m.unlink(missing_ok=True)
    trimmed_y4m.unlink(missing_ok=True)

    logger.info(
        f"VMAF: cropped={mean_vmaf_cropped:.2f}, "
        f"masked={mean_vmaf_masked:.2f}, frames={frame_count}"
    )
    return {
        "mean_vmaf": mean_vmaf_cropped,
        "per_frame_vmaf": per_frame_cropped,
        "mean_vmaf_masked": mean_vmaf_masked,
        "per_frame_vmaf_masked": per_frame_masked,
        "frame_times": frame_times,
        "frame_count": frame_count,
        "content_start_frame": first_content,
        "content_end_frame": last_content,
    }
