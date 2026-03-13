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

    # Drawtext parameters matching generate_input_video.sh overlay.
    # Used to compute crop/mask regions that exclude the frame number
    # overlay from VMAF comparison. These must match the values in the
    # generate script so the pipeline knows exactly where the overlay is.
    overlay_fontsize: int = 40
    overlay_boxborderw: int = 10

    # VMAF computation mode: "cropped", "masked", or "both".
    # - cropped: removes bottom pixels containing the overlay entirely
    # - masked: paints a white box over just the overlay region
    # - both: computes VMAF both ways (default)
    vmaf_mode: str = "both"

    # --- Padding detection ---
    # The testsrc pattern has known RGB colors at specific positions.
    # These are sampled at y = HEIGHT/3 (middle of the color bar band).
    # Format: list of (x_fraction, R, G, B) where x_fraction is relative
    # to video width. Threshold of 50 per channel for matching.
    padding_color_threshold: int = 50

    # --- Debug / visualization ---
    debug_frames: bool = False
    debug_frame_step: int = 10

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

    @property
    def frame_overlay_crop_height(self) -> int:
        """Height to crop from the bottom to remove frame number overlay."""
        return self.calculate_crop_height()

    def calculate_crop_height(self) -> int:
        """
        Calculate how many pixels to crop from the bottom of each frame.

        The overlay is positioned at: y = height - (2 * fontsize) + offset
        The box extends above by boxborderw and below by boxborderw.
        We crop enough to remove the entire text + box.
        """
        # Line height approximates fontsize
        line_height = self.overlay_fontsize
        # Text baseline is at: height - (2 * line_height) + offset
        # Box extends boxborderw pixels above and below the text
        # So the bottom of the box is at: height - line_height + boxborderw
        # (approximately, accounting for text metrics)
        text_bottom = self.overlay_fontsize + self.overlay_boxborderw
        return text_bottom

    def calculate_mask_region(self, height: int, width: int) -> tuple[int, int, int, int]:
        """
        Calculate the bounding box for masking just the white overlay box.

        Returns (x, y, box_width, box_height) for the white box containing frame numbers.
        The box is centered horizontally and positioned near the bottom.

        The drawtext filter draws text centered at x = (w - tw) / 2,
        where tw is the text width. For "9999" this is roughly 150-160 pixels.
        Add boxborderw on each side.
        """
        line_height = self.overlay_fontsize
        # Approximate text width for frame numbers (4 digits, fontsize 40)
        # This is roughly 150-160 pixels, but let's use a more conservative estimate
        approx_text_width = self.overlay_fontsize * 4

        # Box dimensions
        box_width = approx_text_width + 2 * self.overlay_boxborderw
        box_height = line_height + 2 * self.overlay_boxborderw

        # Centered horizontally
        box_x = (width - box_width) // 2

        # Position vertically: text is at y = h - (2*lh) + offset
        # Box extends boxborderw above and below
        overlay_y_offset = 15  # From generate_input_video.sh
        box_y = height - 2 * line_height + overlay_y_offset - self.overlay_boxborderw

        return (box_x, box_y, box_width, box_height)
