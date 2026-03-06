#!/usr/bin/env python3
"""
WebRTC QoE Training Data Generator — CLI Entry Point

Usage:
    # Step 1: Generate the test input video
    uv run python run.py generate-video

    # Step 2: Start Docker containers (do this separately)
    docker compose up -d

    # Step 3: Run experiments
    uv run python run.py run --loss 0 5 10 20 --delay 0 100 --jitter 0 --repeats 2

    # Step 4: Assemble the training dataset
    uv run python run.py build-dataset

    # Optional: show dataset summary
    uv run python run.py summary
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Ensure project root is on the module path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)


# ---- Commands ---------------------------------------------------------------


def cmd_generate_video(args: argparse.Namespace) -> None:
    """Generate the test input video (Y4M + WAV) for Chrome's fake capture."""
    script = PROJECT_ROOT / "scripts" / "generate_input_video.sh"

    if not script.exists():
        print(f"ERROR: Script not found: {script}")
        sys.exit(1)

    cmd = [
        "bash", str(script),
        args.source_url,
        str(args.width),
        str(args.height),
        str(args.fps),
        str(args.duration),
        str(args.padding),
        "media",
    ]
    print(f"Generating test video ({args.width}x{args.height}@{args.fps}fps, "
          f"{args.duration}s content + {args.padding}s padding per side)...")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def cmd_run(args: argparse.Namespace) -> None:
    """Run the experiment pipeline."""
    from pipeline.orchestrator import Orchestrator
    from pipeline.vmaf import check_ffmpeg_vmaf

    # Pre-flight checks
    _check_docker_running()
    if not check_ffmpeg_vmaf():
        print("WARNING: FFmpeg does not have libvmaf support.")
        print("VMAF computation will fail. Install ffmpeg with --enable-libvmaf.")
        print("On Ubuntu/Debian: sudo apt install ffmpeg")
        print("On macOS: brew install ffmpeg")
        if not args.skip_vmaf:
            sys.exit(1)

    ref_video = PROJECT_ROOT / "media" / "reference.y4m"
    if not ref_video.exists():
        print("ERROR: Reference video not found. Run 'python run.py generate-video' first.")
        sys.exit(1)

    # Build config from CLI args
    config = Config()
    if args.loss is not None:
        config.loss_values = args.loss
    if args.delay is not None:
        config.delay_values = args.delay
    if args.jitter is not None:
        config.jitter_values = args.jitter
    if args.bandwidth is not None:
        config.bandwidth_values = args.bandwidth
    if args.repeats is not None:
        config.repeats = args.repeats
    if args.duration is not None:
        config.test_duration_sec = args.duration
    if args.debug_frames:
        config.debug_frames = True

    orchestrator = Orchestrator(config)

    grid = orchestrator.generate_grid()
    print(f"Experiment grid: {len(grid)} experiments")
    print(f"  Loss:     {config.loss_values}")
    print(f"  Delay:    {config.delay_values}")
    print(f"  Jitter:   {config.jitter_values}")
    print(f"  Repeats:  {config.repeats}")
    print(f"  Duration: {config.test_duration_sec}s per experiment")
    estimated_time = len(grid) * (config.test_duration_sec + 15)
    print(f"  Estimated total time: ~{estimated_time // 60} minutes")
    print()

    results = orchestrator.run_all(resume=args.resume)
    print(f"\nCompleted {len(results)} experiments.")

    if results:
        vmaf_scores = [r["mean_vmaf"] for r in results]
        print(f"VMAF range: {min(vmaf_scores):.1f} - {max(vmaf_scores):.1f}")


def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Assemble individual experiment results into a training dataset."""
    from pipeline.dataset import build_dataset

    config = Config()
    csv_path = build_dataset(config.output_dir, config.dataset_dir)
    print(f"Dataset written to: {csv_path}")


def cmd_summary(args: argparse.Namespace) -> None:
    """Show a summary of the current dataset."""
    from pipeline.dataset import dataset_summary

    config = Config()
    csv_path = config.dataset_dir / "dataset.csv"

    if not csv_path.exists():
        print("No dataset found. Run 'python run.py build-dataset' first.")
        sys.exit(1)

    print(dataset_summary(csv_path))


def cmd_debug_alignment(args: argparse.Namespace) -> None:
    """Generate side-by-side frame comparison PNGs for an existing experiment."""
    from pipeline.vmaf import compute_vmaf, generate_frame_comparison
    from pipeline.vmaf import detect_padding_boundaries, _get_frame_count

    config = Config()
    experiment_id = args.experiment_id

    # Find the recording
    recording_path = config.recordings_dir / f"{experiment_id}.webm"
    if not recording_path.exists():
        print(f"ERROR: Recording not found: {recording_path}")
        print(f"Available recordings:")
        for f in sorted(config.recordings_dir.glob("*.webm")):
            print(f"  {f.stem}")
        sys.exit(1)

    reference_video = config.media_dir / "reference.y4m"
    if not reference_video.exists():
        print("ERROR: Reference video not found. Run 'python run.py generate-video' first.")
        sys.exit(1)

    output_dir = config.output_dir / experiment_id / "debug_frames"
    vmaf_json = config.vmaf_dir / f"{experiment_id}_vmaf.json"

    # Load per-frame VMAF scores if available
    per_frame_vmaf = None
    mean_vmaf = None
    if vmaf_json.exists():
        import json
        with open(vmaf_json) as f:
            vmaf_data = json.load(f)
        per_frame_vmaf = [frame["metrics"]["vmaf"] for frame in vmaf_data["frames"]]
        mean_vmaf = vmaf_data["pooled_metrics"]["vmaf"]["mean"]

    # Re-run normalization and trimming (intermediates were deleted)
    print(f"Processing {experiment_id}...")
    width, height, fps = config.video_width, config.video_height, config.video_fps

    # Convert WebM to Y4M
    received_y4m = output_dir / f"{experiment_id}.received.y4m"
    output_dir.mkdir(parents=True, exist_ok=True)
    convert_cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", str(recording_path),
        "-vf", f"scale={width}:{height},fps={fps}",
        "-pix_fmt", "yuv420p",
        str(received_y4m),
    ]
    subprocess.run(convert_cmd, check=True)

    # Detect padding
    first_content, last_content = detect_padding_boundaries(
        received_y4m, width, height, fps,
        padding_duration_sec=config.padding_duration_sec,
        threshold=config.padding_color_threshold,
    )

    # Trim to content
    trimmed_y4m = output_dir / f"{experiment_id}.trimmed.y4m"
    start_time = first_content / fps
    content_duration = (last_content - first_content + 1) / fps
    trim_cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", str(received_y4m),
        "-ss", f"{start_time:.4f}",
        "-t", f"{content_duration:.4f}",
        "-pix_fmt", "yuv420p",
        str(trimmed_y4m),
    ]
    subprocess.run(trim_cmd, check=True)

    # Generate comparison frames
    generate_frame_comparison(
        trimmed_video=trimmed_y4m,
        reference_video=reference_video,
        output_dir=output_dir,
        width=width,
        height=height,
        per_frame_vmaf=per_frame_vmaf,
        mean_vmaf=mean_vmaf,
        step=args.step,
    )

    # Clean up intermediates
    received_y4m.unlink(missing_ok=True)
    trimmed_y4m.unlink(missing_ok=True)

    png_count = len(list(output_dir.glob("*.png")))
    print(f"\nGenerated {png_count} comparison frames in:")
    print(f"  {output_dir}")
    print(f"\nOpen them to verify frame alignment between sender and receiver.")


# ---- Utilities --------------------------------------------------------------


def _check_docker_running() -> None:
    """Verify Docker containers are up."""
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True, text=True, check=True,
            cwd=str(PROJECT_ROOT),
        )
        if "webrtc-sender" not in result.stdout or "webrtc-receiver" not in result.stdout:
            print("WARNING: Docker containers may not be running.")
            print("Start them with: docker compose up -d")
            print("(from the webrtc-qoe-data-generator directory)")
            print()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: Could not check Docker status.")
        print("Make sure Docker is running and containers are up:")
        print("  cd webrtc-qoe-data-generator && docker compose up -d")
        print()


# ---- Main -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WebRTC QoE Training Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python run.py generate-video
  docker compose up -d
  uv run python run.py run --loss 0 5 10 --delay 0 100 --jitter 0 --repeats 2 --duration 20
  uv run python run.py build-dataset
  uv run python run.py summary
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate-video ---
    gen = subparsers.add_parser(
        "generate-video",
        help="Generate test input video (Y4M + WAV)",
    )
    gen.add_argument("--source-url", default=(
        "https://archive.org/download/"
        "e-dv548_lwe08_christa_casebeer_003.ogg/"
        "e-dv548_lwe08_christa_casebeer_003.mp4"
    ), help="URL of source video to download")
    gen.add_argument("--width", type=int, default=640)
    gen.add_argument("--height", type=int, default=480)
    gen.add_argument("--fps", type=int, default=24)
    gen.add_argument("--duration", type=int, default=30,
                     help="Content video duration in seconds")
    gen.add_argument("--padding", type=int, default=5,
                     help="Padding duration in seconds (testsrc color bars on each side)")
    gen.set_defaults(func=cmd_generate_video)

    # --- run ---
    run = subparsers.add_parser(
        "run",
        help="Run experiments across network conditions",
    )
    run.add_argument("--loss", nargs="*", type=float,
                     help="Packet loss percentages (e.g., 0 5 10 20)")
    run.add_argument("--delay", nargs="*", type=int,
                     help="Delay values in ms (e.g., 0 50 100)")
    run.add_argument("--jitter", nargs="*", type=int,
                     help="Jitter values in ms (e.g., 0 25 50)")
    run.add_argument("--bandwidth", nargs="*", type=int,
                     help="Bandwidth limits in kbps (0 = unlimited)")
    run.add_argument("--repeats", type=int,
                     help="Number of repeats per condition (default: 3)")
    run.add_argument("--duration", type=int,
                     help="Test duration in seconds (default: 30)")
    run.add_argument("--resume", action="store_true", default=True,
                     help="Skip already-completed experiments (default)")
    run.add_argument("--no-resume", dest="resume", action="store_false",
                     help="Re-run all experiments even if results exist")
    run.add_argument("--skip-vmaf", action="store_true",
                     help="Continue even if FFmpeg lacks libvmaf")
    run.add_argument("--debug-frames", action="store_true",
                     help="Generate side-by-side frame comparison PNGs for alignment verification")
    run.set_defaults(func=cmd_run)

    # --- build-dataset ---
    ds = subparsers.add_parser(
        "build-dataset",
        help="Assemble training dataset from experiment results",
    )
    ds.set_defaults(func=cmd_build_dataset)

    # --- summary ---
    sm = subparsers.add_parser(
        "summary",
        help="Show dataset summary statistics",
    )
    sm.set_defaults(func=cmd_summary)

    # --- debug-alignment ---
    da = subparsers.add_parser(
        "debug-alignment",
        help="Generate side-by-side frame comparison PNGs for an experiment",
    )
    da.add_argument("experiment_id",
                    help="Experiment ID (e.g., L0.0_D0_J0_BW0_R0)")
    da.add_argument("--step", type=int, default=10,
                    help="Extract every Nth frame (default: 10)")
    da.set_defaults(func=cmd_debug_alignment)

    # Parse and dispatch
    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
