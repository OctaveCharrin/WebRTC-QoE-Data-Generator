# WebRTC QoE Training Data Generator

Generates labeled training data for a deep learning model that predicts
video Quality of Experience (VMAF) from WebRTC network traffic features
(packet sizes and inter-packet timings).

## Architecture

```
Host (Python orchestrator via Selenium)
  |
  +-- [sender container]     Chrome + fake video --> WebRTC --> [receiver container]
  |     tc/netem (impairment)                                   Chrome + MediaRecorder
  |     tcpdump (traffic capture)
  |
  +-- [signaling container]  Python aiohttp (WebSocket relay)

Post-processing (on host):
  reference.y4m + received.webm  -->  ffmpeg libvmaf  -->  VMAF score (label)
  capture.pcap                   -->  scapy           -->  packet sizes + timings (features)
```

Each experiment produces:
- A **pcap file** with all UDP packets from the WebRTC call
- A **VMAF score** computed by comparing sent vs. received video frames
- Together: `(packet_sizes, inter_packet_times) -> VMAF` training pairs

## Prerequisites

| Tool | Install |
|------|---------|
| **Docker + Docker Compose** | [docker.com](https://docs.docker.com/get-docker/) |
| **uv** (Python package manager) | [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| **FFmpeg with libvmaf** | `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Ubuntu 22+) |
| **bash + wget/curl** | Pre-installed on macOS/Linux |

Verify FFmpeg has VMAF support:
```bash
ffmpeg -filters 2>/dev/null | grep libvmaf
# Should print: ... libvmaf ...
```

## Quick Start

```bash
cd webrtc-qoe-data-generator

# 1. Install Python dependencies
uv sync

# 2. Generate the test input video (640x480, 24fps, 30s content + 5s padding)
uv run python run.py generate-video

# 3. Build and start Docker containers
docker compose up -d --build

# 4. Run a small test (2 loss x 2 bitrate x 1 repeat = 4 experiments)
uv run python run.py run --loss 0 10 --delay 0 --jitter 0 --bitrate 600 1500 --repeats 1 --duration 15

# 5. Assemble the dataset
uv run python run.py build-dataset

# 6. View results
uv run python run.py summary

# 7. Fit the QoS->VMAF reward surrogate (for rl-mpquic)
uv run python run.py fit-reward
```

## Usage

### Generate test video

```bash
uv run python run.py generate-video [--width 640] [--height 480] [--fps 24] [--duration 30] [--padding 5]
```

Produces:
- `media/test.y4m` — video with padding: `[5s testsrc color bars] + [30s content] + [5s testsrc]`
- `media/test.wav` — matching audio
- `media/reference.y4m` — content-only video (no padding), used as VMAF reference

The padding (testsrc color bar pattern) is essential for synchronization: it allows
the pipeline to detect exactly where the actual content begins and ends in the
received recording, compensating for the variable startup delay between sender and
receiver. Frame numbers are overlaid on the content for future per-frame alignment.

### Run experiments

```bash
uv run python run.py run \
    --loss 0 5 10 15 20 25 30 40 50 \
    --delay 0 50 100 200 \
    --jitter 0 25 50 \
    --bitrate 300 600 1000 1500 2500 \
    --repeats 3 \
    --duration 30
```

Options:
- `--loss`: Packet loss percentages to test
- `--delay`: One-way delay values in ms
- `--jitter`: Jitter values in ms (applied with normal distribution)
- `--bitrate`: Encoder target bitrates in kbps to sweep (0 = uncapped). Each
  value is enforced two ways so the encoded bitrate equals the swept value:
  the sender's WebRTC encoder is capped (`setParameters` maxBitrate + SDP
  `b=AS` in the signaling page) **and** the sender container's egress is
  rate-limited to it via `tc`/`tbf`. The realized wire bitrate is recorded.
- `--repeats`: Number of repeats per condition
- `--duration`: Recording duration in seconds
- `--vmaf-mode`: Mode to compute vmaf: choose between 'cropped', 'masked', or 'both' (default)
- `--resume / --no-resume`: Skip already-completed experiments (default: resume)
- `--debug-frames`: Generate side-by-side frame comparison PNGs for alignment verification
- `-v`: Verbose debug logging

The pipeline is **resumable** — if interrupted, re-run the same command and
it will skip experiments that already have a `result.json`.

### Build dataset

```bash
uv run python run.py build-dataset
```

Produces `output/dataset/dataset.csv` with one row per experiment.

### View summary

```bash
uv run python run.py summary
```

### Debug frame alignment

After running experiments, verify that the pipeline correctly aligns sender and
receiver frames before VMAF scoring:

```bash
# Post-hoc: generate comparison PNGs for a specific experiment
uv run python run.py debug-alignment L0.0_D0_J0_B1000_R0

# Extract every 5th frame instead of every 10th
uv run python run.py debug-alignment L0.0_D0_J0_B1000_R0 --step 5
```

This produces side-by-side PNGs in `output/<experiment_id>/debug_frames/` showing
the reference (sender) frame on the left and the received frame on the right,
annotated with frame number and VMAF score (if available).

> **Note on alignment:** the received stream typically leads the reference by a
> dozen-or-so frames (Chrome's looping fake capture doesn't reset to frame 0 on
> `replaceTrack`). `compute_vmaf` corrects this with a frame-offset search before
> scoring — without it, VMAF is capped around ~68 on a perfectly good image
> purely from the temporal shift. The detected offset is logged and stored as
> `frame_offset` in `result.json`.

### Re-score existing recordings

After a change to the VMAF/alignment logic, re-score the saved recordings
without re-running the (slow) WebRTC capture:

```bash
uv run python run.py reprocess-vmaf                    # all experiments
uv run python run.py reprocess-vmaf L0.0_D0_J0_B2500_R0 --debug-frames
```

This rewrites each `result.json` and per-frame `.npy` in place (traffic features
untouched), then re-run `build-dataset` and `fit-reward`.

You can also generate these during a run with `--debug-frames`:

```bash
uv run python run.py run --loss 0 10 --delay 0 --jitter 0 --repeats 1 --debug-frames
```

## Output Structure

```
output/
  pcaps/                    # Raw packet captures
    L0_D0_J0_B600_R0.pcap
    L10_D0_J0_B600_R0.pcap
    ...
  recordings/               # Received video recordings
    L0_D0_J0_B600_R0.webm
    ...
  vmaf/                     # VMAF computation results
    L0_D0_J0_B600_R0_vmaf.json
    ...
  L0_D0_J0_B600_R0/        # Per-experiment directory
    result.json             # Metadata + scores
    ..._packet_sizes.npy    # numpy array: packet sizes
    ..._inter_packet_times.npy  # numpy array: inter-packet gaps
    ..._packet_timestamps.npy   # numpy array: seconds from first packet
    ..._per_frame_vmaf.npy      # numpy array: VMAF per content frame (cropped)
    ..._per_frame_vmaf_masked.npy      # numpy array: VMAF per content frame (masked)
    ..._frame_times.npy         # numpy array: seconds from content start
    debug_frames/           # (optional) side-by-side alignment PNGs
      frame_0000.png
      frame_0010.png
      ...
  dataset/
    dataset.csv             # Consolidated training dataset
  reward_model.npz          # Fitted QoS->VMAF surrogate (portable, numpy-only)
  reward_model.pkl          # Same model, pickled (convenience)
  reward_grid.npz           # Raw averaged grid (conditions + mean VMAF)
```

Experiment IDs follow `L{loss}_D{delay}_J{jitter}_B{bitrate}_R{repeat}`
(e.g., `L10_D50_J0_B600_R2`).

### Dataset CSV columns

| Column | Description |
|--------|-------------|
| `experiment_id` | Unique ID (e.g., `L10_D50_J0_B600_R0`) |
| `loss_percent` | Applied packet loss % |
| `delay_ms` | Applied one-way delay ms |
| `jitter_ms` | Applied jitter ms |
| `bitrate_kbps` | Configured encoder target / egress cap (0 = uncapped) |
| `realized_bitrate_kbps` | Measured wire bitrate from the pcap (sanity check) |
| `repeat` | Repeat index |
| `mean_vmaf` | Mean VMAF score (cropped) (0-100, higher = better) |
| `mean_vmaf_masked` | Mean VMAF score (masked) (0-100, higher = better) |
| `frame_count` | Number of video frames compared |
| `packet_count` | Number of UDP packets captured |
| `traffic_duration_sec` | Duration of the traffic capture |
| `packet_sizes_file` | Path to .npy with packet sizes array |
| `inter_packet_times_file` | Path to .npy with inter-packet times array |
| `packet_timestamps_file` | Path to .npy with packet times (seconds from first packet) |
| `per_frame_vmaf_file` | Path to .npy with per-frame VMAF scores (cropped) |
| `per_frame_vmaf_masked_file` | Path to .npy with per-frame VMAF scores (masked) |
| `frame_times_file` | Path to .npy with frame times (seconds from content start) |

### Loading data for model training

```python
import numpy as np
import pandas as pd

df = pd.read_csv("output/dataset/dataset.csv")

for _, row in df.iterrows():
    sizes = np.load(row["packet_sizes_file"])           # shape: (N,)
    timings = np.load(row["inter_packet_times_file"])    # shape: (N,)
    
    # Optional handling of which VMAF score metric was generated
    vmaf = row["mean_vmaf"]                              # float, 0-100 (cropped)
    # vmaf_masked = row["mean_vmaf_masked"]              # float, 0-100 (masked)
    # Feed into your model...
```

### Loading temporal data for real-time VMAF inference

Each experiment also saves time-aligned arrays for training models that predict
VMAF in real time (e.g., a transformer over sliding windows of packets):

```python
import numpy as np
import pandas as pd

df = pd.read_csv("output/dataset/dataset.csv")
row = df.iloc[0]

# Packet-level features with timestamps (t=0 at capture start)
packet_sizes = np.load(row["packet_sizes_file"])          # (N,) int32
packet_times = np.load(row["packet_timestamps_file"])     # (N,) float64, seconds

# Per-frame VMAF with timestamps (t=0 at content start)
# Note: Can use "per_frame_vmaf_masked_file" if vmaf-mode was set to 'masked'
frame_vmaf = np.load(row["per_frame_vmaf_file"])          # (F,) float64, 0-100
frame_times = np.load(row["frame_times_file"])            # (F,) float64, seconds

# Example: for each frame, gather all packets up to that frame's time
for i, t in enumerate(frame_times):
    mask = packet_times <= t
    window_sizes = packet_sizes[mask]
    target_vmaf = frame_vmaf[i]
    # Feed window_sizes -> target_vmaf into your model...
```

## Debugging

- **Watch the browser**: Open `http://localhost:7900` (sender) or
  `http://localhost:7901` (receiver) in your browser for live noVNC view.
  Password: `secret`.

- **Check container logs**:
  ```bash
  docker compose logs signaling
  docker compose logs sender
  docker compose logs receiver
  ```

- **Test signaling server**: Open `http://localhost:8080/health` — should return
  `{"status": "ok", ...}`.

- **`Permission denied` writing `output/recordings` or `output/pcaps`**: Docker
  creates bind-mounted directories as `root` when they don't already exist, so
  the host process can't write into them. Fix the ownership from inside the
  containers (which share the mount; no host `sudo` needed):
  ```bash
  docker exec --user root webrtc-receiver chown -R "$(id -u):$(id -g)" /home/seluser/recordings
  docker exec --user root webrtc-sender  chown -R "$(id -u):$(id -g)" /home/seluser/pcaps
  ```

- **Manual WebRTC test**: Open `http://localhost:8080/?role=sender` in one tab
  and `http://localhost:8080/?role=receiver` in another (from host, not inside
  Docker).

## Project Structure

```
webrtc-qoe-data-generator/
  run.py                     # CLI entry point
  config.py                  # Central configuration
  pyproject.toml             # Python dependencies (uv)
  uv.lock                    # Locked dependency versions
  docker-compose.yml         # Container orchestration
  docker/
    Dockerfile.browser       # Chrome + tc + tcpdump
    Dockerfile.signaling     # Python signaling server
  signaling/
    server.py                # WebSocket relay + static server
    static/
      index.html             # WebRTC sender/receiver page
  scripts/
    generate_input_video.sh  # FFmpeg video generation
  pipeline/
    orchestrator.py          # Main experiment loop
    browser.py               # Selenium WebDriver control
    network.py               # tc/netem + tcpdump
    traffic.py               # pcap parsing (scapy)
    vmaf.py                  # VMAF computation (ffmpeg)
    dataset.py               # Dataset assembly
    surrogate.py             # QoS->VMAF multilinear surrogate fitting
  reward/
    qos_vmaf_reward.py       # Portable reward module (numpy-only, for rl-mpquic)
    reward_model.npz         # Fitted model (written by `fit-reward`)
  media/                     # Generated test video/audio
  output/                    # Experiment results
```

## Network Impairment

Network degradation is applied inside the sender Docker container using
Linux `tc`/`netem`:

| Parameter | Tool | Example |
|-----------|------|---------|
| Packet loss | `tc qdisc add dev eth0 root netem loss 10%` | 10% random loss |
| Delay | `tc qdisc add dev eth0 root netem delay 100ms` | 100ms one-way |
| Jitter | `tc qdisc add dev eth0 root netem delay 100ms 50ms distribution normal` | 100ms +/- 50ms |
| Bitrate | `tc qdisc add ... tbf rate 1000kbit ...` (+ encoder cap) | 1 Mbps egress + encode |

All parameters can be combined in a single experiment. The bitrate `tbf`
attaches as a child of the netem qdisc when netem is present, or as the root
qdisc for bitrate-only conditions.

## QoS → VMAF reward (for rl-mpquic)

The sibling RL project (`../rl-mpquic`) needs a continuous, real-VMAF-grounded
quality signal. After building the dataset, fit a surrogate over the grid:

```bash
uv run python run.py fit-reward
```

This averages `mean_vmaf` across repeats per `(bitrate, loss, delay, jitter)`
condition, fits a **multilinear interpolant** over the regular grid, and writes
`output/reward_model.{npz,pkl}`, `output/reward_grid.npz`, and a portable copy
at `reward/reward_model.npz`. It also prints the VMAF surface across bitrate ×
loss and **warns if the measured VMAF range is too narrow** (WebRTC error
resilience can keep VMAF flat — if so, widen the dynamic range with lower
bitrates / higher loss / longer durations).

### Consuming the reward from rl-mpquic

`reward/qos_vmaf_reward.py` is self-contained (numpy only) and has no dependency
on the rest of this repo. Copy both the module and its model into rl-mpquic:

```bash
cp reward/qos_vmaf_reward.py reward/reward_model.npz  /path/to/rl-mpquic/src/ns3env/
```

Then call it with the exact contract signature:

```python
from qos_vmaf_reward import QoSVmafReward

reward = QoSVmafReward()           # loads reward_model.npz next to the module
                                   # (or set $QOS_VMAF_MODEL to a path)

vmaf = reward.vmaf(
    bitrate_kbps=target_bitrate_kbps,   # App action: encoder target send rate
    loss_pct=packet_loss * 100,         # App state; multiply if reported in [0,1]
    delay_ms=aggregate_rtt_ms / 2,      # one-way delay: pass RTT/2 (netem axis is one-way)
    jitter_ms=jitter_ms,                # App state: interarrival jitter / latency std-dev
)                                       # -> VMAF in [0, 100]
```

**Important:** the grid's `delay_ms` axis is **one-way** (netem applies one-way
delay), while rl-mpquic's transport state reports an aggregate RTT — so callers
must pass `rtt / 2`. Inputs are clamped to the measured grid box, so
out-of-range queries saturate at the nearest measured boundary. For fast
training, `reward.vmaf_batch(X)` takes an `(n, 4)` array with columns in the
fixed order `[bitrate_kbps, loss_pct, delay_ms, jitter_ms]`.
