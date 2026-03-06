from __future__ import annotations

"""
Main experiment orchestrator.

Ties together all pipeline modules to run the full data generation loop:
browser control, network impairment, traffic capture, video recording,
VMAF computation, and result storage.

Replaces the Java test classes from the original project:
  - OpenViduBasicConferencePacketLossTest
  - AppRtcAdvancedTest
  - etc.
"""

import itertools
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config
from pipeline.browser import BrowserController
from pipeline.network import NetworkController, NetworkCondition
from pipeline.traffic import parse_pcap, save_traffic_features
from pipeline.vmaf import compute_vmaf

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Runs WebRTC QoE experiments across a grid of network conditions.

    Each experiment:
      1. Establishes a WebRTC call (sender -> receiver)
      2. Applies network impairment via tc/netem
      3. Records the received video
      4. Captures network traffic (pcap)
      5. Computes VMAF (ground truth label)
      6. Extracts traffic features (model input)
    """

    def __init__(self, config: Config):
        self.config = config

        self.sender = BrowserController(
            config.sender_selenium_url, role="sender"
        )
        self.receiver = BrowserController(
            config.receiver_selenium_url, role="receiver"
        )
        self.network = NetworkController(
            config.sender_container, config.network_interface
        )

    # ---- Experiment grid ---------------------------------------------------

    def generate_grid(self) -> list[tuple[NetworkCondition, int]]:
        """
        Generate all (condition, repeat) pairs from the configured ranges.

        Default grid: 10 loss x 6 delay x 4 jitter x 1 bw x 3 repeats = 720.
        """
        grid = []
        for loss, delay, jitter, bw in itertools.product(
            self.config.loss_values,
            self.config.delay_values,
            self.config.jitter_values,
            self.config.bandwidth_values,
        ):
            for repeat in range(self.config.repeats):
                condition = NetworkCondition(
                    loss_percent=loss,
                    delay_ms=delay,
                    jitter_ms=jitter,
                    bandwidth_kbps=bw,
                )
                grid.append((condition, repeat))
        return grid

    def experiment_done(self, experiment_id: str) -> bool:
        """Check if an experiment was already completed (for resumability)."""
        result_file = self.config.output_dir / experiment_id / "result.json"
        return result_file.exists()

    # ---- Single experiment -------------------------------------------------

    def run_single(
        self, condition: NetworkCondition, repeat: int
    ) -> dict:
        """
        Run one experiment with the given network condition.

        This implements the 14-step flow described in the plan:
          1-2.  Navigate sender and receiver to the WebRTC page
          3.    Wait for WebRTC connection
          4.    Start tcpdump
          5.    Start MediaRecorder on receiver
          6.    Apply tc/netem
          7.    Wait for test duration
          8.    Reset tc/netem
          9.    Stop recording, retrieve WebM
          10.   Stop tcpdump, copy pcap
          11.   Refresh browsers
          12-13. Post-process and save results
        """
        experiment_id = f"{condition.experiment_id}_R{repeat}"
        exp_dir = self.config.output_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        pcap_container_path = f"/home/seluser/pcaps/{experiment_id}.pcap"
        pcap_local = self.config.pcap_dir / f"{experiment_id}.pcap"
        recording_path = self.config.recordings_dir / f"{experiment_id}.webm"
        vmaf_json = self.config.vmaf_dir / f"{experiment_id}_vmaf.json"

        logger.info(f"{'=' * 60}")
        logger.info(f"Experiment: {experiment_id}")
        logger.info(f"Condition:  {condition}")
        logger.info(f"{'=' * 60}")

        try:
            # --- Steps 1-2: Navigate browsers to the WebRTC page ---
            # Use a unique room per experiment so stale signaling state
            # from a previous run doesn't interfere.
            room = experiment_id
            base_url = self.config.signaling_url_internal

            self.receiver.navigate(f"{base_url}/?role=receiver&room={room}")
            time.sleep(2)  # Let receiver's WebSocket connect before sender joins
            self.sender.navigate(f"{base_url}/?role=sender&room={room}")
            time.sleep(1)  # Let sender's page settle before polling

            # --- Step 3: Wait for WebRTC connection ---
            sender_ok = self.sender.wait_for_connection(
                self.config.webrtc_connect_timeout_sec
            )
            receiver_ok = self.receiver.wait_for_connection(
                self.config.webrtc_connect_timeout_sec
            )
            if not (sender_ok and receiver_ok):
                raise RuntimeError("WebRTC connection failed to establish")

            # Small settle time for media flow to stabilize
            time.sleep(2)

            # --- Step 4: Start tcpdump ---
            self.network.start_tcpdump(pcap_container_path)

            # --- Step 5: Start MediaRecorder on receiver ---
            self.receiver.start_recording()

            # --- Step 6: Apply network impairment ---
            self.network.apply_netem(condition)

            # --- Step 7: Wait for test duration ---
            logger.info(
                f"Recording for {self.config.test_duration_sec}s "
                f"under {condition}..."
            )
            time.sleep(self.config.test_duration_sec)

            # --- Step 8: Reset network ---
            self.network.reset_netem()

            # --- Step 9: Stop recording and retrieve WebM ---
            self.receiver.stop_recording()
            time.sleep(1)  # Let recording finalize
            self.receiver.save_recording(recording_path)

            # --- Step 10: Stop tcpdump and copy pcap ---
            self.network.stop_tcpdump()
            time.sleep(0.5)
            self.network.copy_pcap(pcap_container_path, pcap_local)

            # --- Step 11: Refresh browsers for next experiment ---
            self.sender.refresh()
            self.receiver.refresh()
            time.sleep(1)

            # --- Step 12: Post-process ---

            # 12a: Compute VMAF
            reference_video = self.config.media_dir / "reference.y4m"
            vmaf_result = compute_vmaf(
                received_video=recording_path,
                reference_video=reference_video,
                output_json=vmaf_json,
                width=self.config.video_width,
                height=self.config.video_height,
                fps=self.config.video_fps,
                padding_duration_sec=self.config.padding_duration_sec,
                padding_threshold=self.config.padding_color_threshold,
            )

            # 12b: Parse pcap and extract traffic features
            traffic_data = parse_pcap(pcap_local)
            traffic_paths = save_traffic_features(
                traffic_data, exp_dir, experiment_id
            )

            # 12c: Save per-frame VMAF scores and frame timestamps as .npy
            per_frame_vmaf_path = exp_dir / f"{experiment_id}_per_frame_vmaf.npy"
            frame_times_path = exp_dir / f"{experiment_id}_frame_times.npy"
            np.save(per_frame_vmaf_path, np.array(vmaf_result["per_frame_vmaf"], dtype=np.float64))
            np.save(frame_times_path, np.array(vmaf_result["frame_times"], dtype=np.float64))

            # --- Step 13: Save experiment result ---
            result = {
                "experiment_id": experiment_id,
                "loss_percent": condition.loss_percent,
                "delay_ms": condition.delay_ms,
                "jitter_ms": condition.jitter_ms,
                "bandwidth_kbps": condition.bandwidth_kbps,
                "repeat": repeat,
                "mean_vmaf": vmaf_result["mean_vmaf"],
                "frame_count": vmaf_result["frame_count"],
                "packet_count": traffic_data["total_packets"],
                "traffic_duration_sec": traffic_data["duration_sec"],
                "packet_sizes_file": str(traffic_paths["packet_sizes"]),
                "inter_packet_times_file": str(traffic_paths["inter_packet_times"]),
                "packet_timestamps_file": str(traffic_paths["packet_timestamps"]),
                "per_frame_vmaf_file": str(per_frame_vmaf_path),
                "frame_times_file": str(frame_times_path),
                "recording_file": str(recording_path),
                "pcap_file": str(pcap_local),
            }

            with open(exp_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2)

            logger.info(
                f"Result: VMAF={vmaf_result['mean_vmaf']:.2f}, "
                f"packets={traffic_data['total_packets']}, "
                f"frames={vmaf_result['frame_count']}"
            )
            return result

        except Exception as e:
            logger.error(f"EXPERIMENT FAILED: {e}", exc_info=True)
            # Always clean up network state to avoid leaking into next run
            try:
                self.network.reset_netem()
            except Exception:
                pass
            try:
                self.network.stop_tcpdump()
            except Exception:
                pass
            raise

    # ---- Full experiment sweep ---------------------------------------------

    def run_all(self, resume: bool = True) -> list[dict]:
        """
        Run all experiments in the configured grid.

        Args:
            resume: If True, skip experiments whose result.json already exists.

        Returns:
            List of result dictionaries for completed experiments.
        """
        self.config.ensure_dirs()
        grid = self.generate_grid()
        total = len(grid)
        results = []
        skipped = 0
        failed = 0

        logger.info(f"Experiment grid: {total} total experiments")
        logger.info(
            f"  Loss values:   {self.config.loss_values}"
        )
        logger.info(
            f"  Delay values:  {self.config.delay_values}"
        )
        logger.info(
            f"  Jitter values: {self.config.jitter_values}"
        )
        logger.info(
            f"  Repeats:       {self.config.repeats}"
        )
        logger.info(
            f"  Duration:      {self.config.test_duration_sec}s per experiment"
        )

        # Connect browsers once and reuse across all experiments
        logger.info("Connecting to browsers...")
        self.sender.connect(self.config.sender_chrome_flags)
        self.receiver.connect(self.config.receiver_chrome_flags)

        try:
            for i, (condition, repeat) in enumerate(grid):
                experiment_id = f"{condition.experiment_id}_R{repeat}"

                # Resumability: skip if already done
                if resume and self.experiment_done(experiment_id):
                    skipped += 1
                    logger.info(
                        f"[{i + 1}/{total}] SKIP {experiment_id} (already done)"
                    )
                    continue

                logger.info(f"[{i + 1}/{total}] Running {experiment_id}")
                try:
                    result = self.run_single(condition, repeat)
                    results.append(result)
                except Exception as e:
                    failed += 1
                    logger.error(f"[{i + 1}/{total}] FAILED: {e}")
                    # Try to recover browser state for next experiment
                    try:
                        self.sender.refresh()
                        self.receiver.refresh()
                        time.sleep(2)
                    except Exception:
                        # If browser is unrecoverable, reconnect
                        logger.warning("Reconnecting browsers after failure...")
                        self.sender.quit()
                        self.receiver.quit()
                        time.sleep(2)
                        self.sender.connect(self.config.sender_chrome_flags)
                        self.receiver.connect(self.config.receiver_chrome_flags)

        finally:
            self.sender.quit()
            self.receiver.quit()

        # Summary
        logger.info(f"{'=' * 60}")
        logger.info(f"COMPLETE: {len(results)} succeeded, {skipped} skipped, {failed} failed")
        logger.info(f"{'=' * 60}")

        return results
