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

# 4. Run a small test (2 conditions x 1 repeat = 2 experiments)
uv run python run.py run --loss 0 10 --delay 0 --jitter 0 --repeats 1 --duration 15

# 5. Assemble the dataset
uv run python run.py build-dataset

# 6. View results
uv run python run.py summary
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
    --repeats 3 \
    --duration 30
```

Options:
- `--loss`: Packet loss percentages to test
- `--delay`: One-way delay values in ms
- `--jitter`: Jitter values in ms (applied with normal distribution)
- `--bandwidth`: Bandwidth limits in kbps (0 = unlimited)
- `--repeats`: Number of repeats per condition
- `--duration`: Recording duration in seconds
- `--resume / --no-resume`: Skip already-completed experiments (default: resume)
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

## Output Structure

```
output/
  pcaps/                    # Raw packet captures
    L0_D0_J0_BW0_R0.pcap
    L10_D0_J0_BW0_R0.pcap
    ...
  recordings/               # Received video recordings
    L0_D0_J0_BW0_R0.webm
    ...
  vmaf/                     # VMAF computation results
    L0_D0_J0_BW0_R0_vmaf.json
    ...
  L0_D0_J0_BW0_R0/         # Per-experiment directory
    result.json             # Metadata + scores
    ..._packet_sizes.npy    # numpy array: packet sizes
    ..._inter_packet_times.npy  # numpy array: inter-packet gaps
  dataset/
    dataset.csv             # Consolidated training dataset
```

### Dataset CSV columns

| Column | Description |
|--------|-------------|
| `experiment_id` | Unique ID (e.g., `L10_D50_J0_BW0_R0`) |
| `loss_percent` | Applied packet loss % |
| `delay_ms` | Applied one-way delay ms |
| `jitter_ms` | Applied jitter ms |
| `bandwidth_kbps` | Applied bandwidth limit (0 = unlimited) |
| `repeat` | Repeat index |
| `mean_vmaf` | Mean VMAF score (0-100, higher = better) |
| `frame_count` | Number of video frames compared |
| `packet_count` | Number of UDP packets captured |
| `traffic_duration_sec` | Duration of the traffic capture |
| `packet_sizes_file` | Path to .npy with packet sizes array |
| `inter_packet_times_file` | Path to .npy with inter-packet times array |

### Loading data for model training

```python
import numpy as np
import pandas as pd

df = pd.read_csv("output/dataset/dataset.csv")

for _, row in df.iterrows():
    sizes = np.load(row["packet_sizes_file"])           # shape: (N,)
    timings = np.load(row["inter_packet_times_file"])    # shape: (N,)
    vmaf = row["mean_vmaf"]                              # float, 0-100
    # Feed into your model...
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
| Bandwidth | `tc qdisc add ... tbf rate 1mbit burst 32kbit latency 50ms` | 1 Mbps limit |

All parameters can be combined in a single experiment.
