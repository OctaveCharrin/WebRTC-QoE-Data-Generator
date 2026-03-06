"""Central configuration for the WebRTC QoE data generation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    # --- Paths ---
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)

    @property
    def media_dir(self) -> Path:
        return self.project_root / "media"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output"

    @property
    def pcap_dir(self) -> Path:
        return self.output_dir / "pcaps"

    @property
    def recordings_dir(self) -> Path:
        return self.output_dir / "recordings"

    @property
    def vmaf_dir(self) -> Path:
        return self.output_dir / "vmaf"

    @property
    def dataset_dir(self) -> Path:
        return self.output_dir / "dataset"

    # --- Video settings (matches original generate_input_video.sh defaults) ---
    video_width: int = 640
    video_height: int = 480
    video_fps: int = 24
    video_duration_sec: int = 30
    padding_duration_sec: int = 5
    audio_sample_rate: int = 48000
    audio_channels: int = 2

    # --- Padding detection ---
    # The testsrc pattern has known RGB colors at specific positions.
    # These are sampled at y = HEIGHT/3 (middle of the color bar band).
    # Format: list of (x_fraction, R, G, B) where x_fraction is relative
    # to video width. Threshold of 50 per channel for matching.
    padding_color_threshold: int = 50

    # --- Docker container names (must match docker-compose.yml) ---
    sender_container: str = "webrtc-sender"
    receiver_container: str = "webrtc-receiver"
    signaling_container: str = "webrtc-signaling"

    # --- Selenium remote WebDriver URLs ---
    sender_selenium_url: str = "http://localhost:4444"
    receiver_selenium_url: str = "http://localhost:4445"

    # --- Signaling server URL (as seen from inside Docker network) ---
    signaling_url_internal: str = "http://signaling:8080"

    # --- Network impairment parameter ranges ---
    loss_values: List[float] = field(
        default_factory=lambda: [0, 2, 5, 10, 15, 20, 25, 30, 40, 50]
    )
    delay_values: List[int] = field(
        default_factory=lambda: [0, 50, 100, 200, 300, 500]
    )
    jitter_values: List[int] = field(
        default_factory=lambda: [0, 25, 50, 100]
    )
    bandwidth_values: List[int] = field(
        default_factory=lambda: [0]  # 0 = unlimited
    )
    repeats: int = 3

    # --- Experiment timing ---
    test_duration_sec: int = 30
    webrtc_connect_timeout_sec: int = 45
    poll_interval_sec: float = 0.5

    # --- Chrome flags for the sender browser ---
    # Adapted from ElasTestRemoteControlParent.java constants.
    # Paths are inside the Docker container.
    sender_chrome_flags: List[str] = field(default_factory=lambda: [
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        "--use-file-for-fake-video-capture=/home/seluser/media/test.y4m",
        "--use-file-for-fake-audio-capture=/home/seluser/media/test.wav",
        "--disable-rtc-smoothness-algorithm",
        "--ignore-certificate-errors",
        "--autoplay-policy=no-user-gesture-required",
        "--no-sandbox",
        "--unsafely-treat-insecure-origin-as-secure=http://signaling:8080",
    ])

    # Receiver only needs basic flags (no fake media — it receives the stream)
    receiver_chrome_flags: List[str] = field(default_factory=lambda: [
        "--use-fake-ui-for-media-stream",
        "--ignore-certificate-errors",
        "--autoplay-policy=no-user-gesture-required",
        "--no-sandbox",
    ])

    # --- Network interface inside Docker containers ---
    network_interface: str = "eth0"

    def ensure_dirs(self) -> None:
        """Create all output directories."""
        for d in [self.media_dir, self.output_dir, self.pcap_dir,
                  self.recordings_dir, self.vmaf_dir, self.dataset_dir]:
            d.mkdir(parents=True, exist_ok=True)
