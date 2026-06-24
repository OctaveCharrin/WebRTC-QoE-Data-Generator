from __future__ import annotations

"""
Network impairment (tc/netem) and traffic capture (tcpdump) via docker exec.

Replaces the network simulation logic from the original Java project:
  - ElasTestRemoteControlParent.simulateNetwork()
  - ElasTestRemoteControlParent.resetNetwork()

Adds the critical missing component: network traffic capture via tcpdump.
"""

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NetworkCondition:
    """Parameters for a single network impairment experiment."""

    loss_percent: float = 0.0
    delay_ms: int = 0
    jitter_ms: int = 0
    bitrate_kbps: int = 0  # encoder target + egress rate limit; 0 = unlimited

    @property
    def experiment_id(self) -> str:
        """Unique string identifier for this network condition."""
        return f"L{self.loss_percent}_D{self.delay_ms}_J{self.jitter_ms}_B{self.bitrate_kbps}"

    @property
    def has_netem(self) -> bool:
        """Whether loss/delay/jitter require a netem qdisc."""
        return (self.loss_percent > 0 or self.delay_ms > 0
                or self.jitter_ms > 0)

    @property
    def has_impairment(self) -> bool:
        return self.has_netem or self.bitrate_kbps > 0

    def __str__(self) -> str:
        parts = []
        if self.loss_percent > 0:
            parts.append(f"loss={self.loss_percent}%")
        if self.delay_ms > 0:
            parts.append(f"delay={self.delay_ms}ms")
        if self.jitter_ms > 0:
            parts.append(f"jitter={self.jitter_ms}ms")
        if self.bitrate_kbps > 0:
            parts.append(f"bitrate={self.bitrate_kbps}kbps")
        return ", ".join(parts) if parts else "no impairment"


class NetworkController:
    """
    Applies tc/netem rules and runs tcpdump inside Docker containers.

    All commands are executed via `docker exec` so this works from the host
    on both Linux and macOS (Docker Desktop).
    """

    def __init__(self, container_name: str, interface: str = "eth0"):
        self.container_name = container_name
        self.interface = interface
        self._tcpdump_proc: Optional[subprocess.Popen] = None

    # ---- Docker exec helper ------------------------------------------------

    def _docker_exec(
        self, cmd: list[str], check: bool = True, user: str = "root"
    ) -> subprocess.CompletedProcess:
        """Execute a command inside the Docker container."""
        full_cmd = ["docker", "exec", "--user", user, self.container_name] + cmd
        logger.debug(f"  exec: {' '.join(cmd)}")
        result = subprocess.run(
            full_cmd, capture_output=True, text=True, check=False
        )
        if check and result.returncode != 0:
            logger.error(f"  stderr: {result.stderr.strip()}")
            result.check_returncode()
        return result

    # ---- tc/netem ----------------------------------------------------------

    def apply_netem(self, condition: NetworkCondition) -> None:
        """
        Apply tc/netem (+ optional tbf rate limit) for the given condition.

        The original Java code applied loss, delay, and jitter as separate
        tc commands which would overwrite each other (only one root qdisc
        allowed). This version combines all parameters into a single command.

        The bitrate cap is enforced with a tbf qdisc. When netem is also
        present, tbf is attached as a child of the netem qdisc; when the
        condition is bitrate-only, tbf is the root qdisc. Together with the
        encoder-side cap (signaling page), this pins the egress rate to the
        swept bitrate so VMAF depends on it.
        """
        if not condition.has_impairment:
            logger.info("No network impairment to apply (baseline condition)")
            return

        if condition.has_netem:
            # Build and apply the netem qdisc as root (handle 1:)
            netem_args: list[str] = []
            if condition.delay_ms > 0:
                if condition.jitter_ms > 0:
                    # Delay with jitter using normal distribution
                    netem_args += [
                        "delay", f"{condition.delay_ms}ms",
                        f"{condition.jitter_ms}ms", "distribution", "normal",
                    ]
                else:
                    netem_args += ["delay", f"{condition.delay_ms}ms"]
            if condition.loss_percent > 0:
                netem_args += ["loss", f"{condition.loss_percent}%"]

            self._docker_exec([
                "tc", "qdisc", "add", "dev", self.interface,
                "root", "handle", "1:", "netem",
            ] + netem_args)
            logger.info(f"Applied netem: {condition}")

            if condition.bitrate_kbps > 0:
                self._docker_exec(
                    self._tbf_cmd(condition.bitrate_kbps,
                                  parent=["parent", "1:", "handle", "2:"]),
                    check=False,
                )
                logger.info(f"Applied bitrate rate limit: {condition.bitrate_kbps} kbps")
        elif condition.bitrate_kbps > 0:
            # Bitrate-only condition: tbf is the root qdisc.
            self._docker_exec(
                self._tbf_cmd(condition.bitrate_kbps,
                              parent=["root", "handle", "1:"]),
                check=False,
            )
            logger.info(f"Applied bitrate rate limit: {condition.bitrate_kbps} kbps")

    def _tbf_cmd(self, rate_kbps: int, parent: list[str]) -> list[str]:
        """Build a tbf qdisc command for a given rate (kbps).

        burst/latency are sized generously so the limiter shapes to the rate
        without starving the video flow (a too-small burst throttles below the
        nominal rate at low bitrates).
        """
        burst_kbit = max(32, rate_kbps // 8)
        return [
            "tc", "qdisc", "add", "dev", self.interface,
        ] + parent + [
            "tbf",
            "rate", f"{rate_kbps}kbit",
            "burst", f"{burst_kbit}kbit",
            "latency", "300ms",
        ]

    def reset_netem(self) -> None:
        """
        Remove all tc/netem rules, restoring normal network behavior.

        Uses 'tc qdisc del' which is cleaner than the original project's
        approach of 'tc qdisc replace ... loss 0%'.
        """
        result = self._docker_exec(
            ["tc", "qdisc", "del", "dev", self.interface, "root"],
            check=False,
        )
        if result.returncode == 0:
            logger.info("Reset netem rules")
        elif "No such file or directory" in result.stderr:
            # No qdisc to delete — already clean
            pass
        else:
            logger.warning(f"netem reset warning: {result.stderr.strip()}")

    # ---- tcpdump -----------------------------------------------------------

    def start_tcpdump(self, pcap_path: str) -> None:
        """
        Start tcpdump as a background process inside the container.

        This is the critical new component that was missing from the original
        project. It captures UDP packets (WebRTC media uses RTP over UDP).

        Args:
            pcap_path: Path inside the container where the pcap will be saved
                       (e.g., /home/seluser/pcaps/experiment_001.pcap).
        """
        cmd = [
            "docker", "exec", "--user", "root", "-d",
            self.container_name,
            "tcpdump",
            "-i", self.interface,
            "-w", pcap_path,
            "-U",          # Packet-buffered output (flush after each packet)
            "udp",         # Only capture UDP (WebRTC media)
        ]
        self._tcpdump_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        # Give tcpdump a moment to initialize
        time.sleep(0.5)
        logger.info(f"Started tcpdump -> {pcap_path}")

    def stop_tcpdump(self) -> None:
        """Stop tcpdump gracefully with SIGTERM."""
        self._docker_exec(
            ["pkill", "-SIGTERM", "tcpdump"], check=False, user="root"
        )
        if self._tcpdump_proc:
            try:
                self._tcpdump_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._tcpdump_proc.kill()
            self._tcpdump_proc = None
        logger.info("Stopped tcpdump")

    def copy_pcap(self, container_path: str, local_path: Path) -> Path:
        """Copy a pcap file from the container to the host filesystem."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["docker", "cp",
             f"{self.container_name}:{container_path}",
             str(local_path)],
            check=True,
        )
        size_kb = local_path.stat().st_size / 1024
        logger.info(f"Copied pcap: {local_path.name} ({size_kb:.1f} KB)")
        return local_path
