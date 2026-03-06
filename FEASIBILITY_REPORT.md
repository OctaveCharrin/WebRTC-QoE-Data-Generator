# Feasibility Report: QoE Data Generation for WebRTC Deep Learning Model

## 1. Executive Summary

**Goal:** Generate labeled training data consisting of (network traffic features, VMAF score) pairs, where network traffic features are packet sizes and inter-packet timings observed from the network provider side during a WebRTC call, and VMAF scores are the ground-truth QoE values computed by comparing sent vs received video frames.

**Verdict: FEASIBLE, with significant modernization work required.** The existing `elastest-webrtc-qoe-meter` project provides a solid conceptual foundation and a proven pipeline architecture, but nearly every component is outdated (2017-2019 era) and needs updating or replacing. The core ideas (fake video injection, browser automation, network impairment via tc/netem, VMAF comparison) are all sound and remain the standard approach.

---

## 2. Existing Codebase Analysis

### 2.1 Project Structure

```
elastest-webrtc-qoe-meter/
  pom.xml                          # Maven build (Java 8, Selenium 3)
  scripts/
    generate_input_video.sh         # FFmpeg pipeline to create test video
    calculate_qoe_metrics.sh        # Post-processing: VMAF, VQMT, PESQ, ViSQOL
  src/main/java/.../
    ElasTestRemoteControlParent.java  # Core: browser control, recording, network sim
  src/main/resources/
    js/elastest-remote-control.js     # Injected JS: RTCPeerConnection monkey-patch + RecordRTC
  src/test/java/.../
    apprtc/AppRtcAdvancedTest.java    # Tests against AppRTC (DEAD service)
    openvidu/OpenVidu*Test.java       # Tests against OpenVidu demos
    janus/JanusConferenceTest.java    # Tests against Janus gateway
    webrtcsamples/WebRtc*Test.java    # Tests against Google WebRTC samples
  data/
    openvidu_objective_results_*%.csv  # Sample results (VMAF per frame at 0-50% loss)
```

### 2.2 Current Pipeline (3 Stages)

1. **Generate Input Video** (`generate_input_video.sh`):
   - Downloads a sample video, scales to 640x480@24fps
   - Overlays frame numbers (for later alignment)
   - Adds test-pattern padding (for start/end detection)
   - Outputs Y4M video + WAV audio for Chrome's fake media capture

2. **Run WebRTC Call with Network Impairment** (Java/Selenium tests):
   - Launches Chrome in Docker with fake video/audio device
   - Opens a WebRTC application (AppRTC, OpenVidu, etc.)
   - Injects JavaScript to monkey-patch `RTCPeerConnection` and record streams via RecordRTC
   - Applies network impairment (loss/delay/jitter) via `tc qdisc add dev eth0 root netem ...`
   - Records both presenter (local) and viewer (remote) streams as WebM
   - Retrieves recordings via Base64 transfer through JavaScript

3. **Calculate QoE Metrics** (`calculate_qoe_metrics.sh`):
   - Remuxes WebM recordings to fixed bitrate/resolution
   - Detects and removes test-pattern padding frames
   - Optionally aligns frames using OCR on frame numbers (via gocr)
   - Converts to YUV and runs VMAF, VQMT, PSNR, SSIM, VIFp, PESQ, ViSQOL
   - Outputs per-frame CSV results

### 2.3 What Works Conceptually

- The 3-stage pipeline architecture is exactly what you need
- Frame numbering overlay for temporal alignment is clever and necessary
- Test-pattern padding for automatic start/end detection is well designed
- The idea of running Chrome with `--use-file-for-fake-video-capture` is the standard approach
- Network impairment via Linux `tc`/`netem` is the correct tool
- VMAF as the primary QoE metric is the industry standard

### 2.4 What Is Broken or Obsolete

| Component | Issue | Severity |
|-----------|-------|----------|
| **AppRTC** (`appr.tc`) | Service is **dead/shutdown** | CRITICAL |
| **OpenVidu demos** | `demos.openvidu.io` may no longer exist or API changed significantly | CRITICAL |
| **Selenium 3.141.59** | End-of-life; Selenium 4.x is current | HIGH |
| **selenium-jupiter 3.3.4** | Outdated; current is 4.x+ with different API | HIGH |
| **Java 8** | Old; Java 17+ recommended | MEDIUM |
| **RecordRTC 5.5.8 from CDN** | Loaded from CDN (fragile); version is old | MEDIUM |
| **`pc.getLocalStreams()` / `pc.getRemoteStreams()`** | **Deprecated WebRTC APIs** - removed in modern Chrome | CRITICAL |
| **Docker browser integration** | selenium-jupiter's Docker API has changed | HIGH |
| **VMAF `run_vmaf` Python script** | Netflix VMAF now uses `libvmaf` with FFmpeg filter | MEDIUM |
| **VQMT binary** | Separate tool; modern FFmpeg has built-in SSIM/PSNR filters | LOW |
| **gocr for OCR** | Obscure tool; Tesseract is standard and better | LOW |
| **No traffic capture** | The project records video only - **no packet capture at all** | CRITICAL |

---

## 3. Critical Gap: Network Traffic Capture

**The existing project does NOT capture network traffic.** It only records video streams and computes quality metrics. For your deep learning model, you need the network-side view (packet sizes and inter-packet timings). This is an entirely new component that must be added.

### What Needs to Be Captured

- **Packet sizes** (bytes) of all RTP/RTCP packets in the WebRTC media flow
- **Packet timestamps** (microsecond precision) to derive inter-packet gaps
- **Direction** (uplink vs downlink)
- Optionally: STUN/TURN packets, DTLS handshake, RTCP feedback

### How to Capture

- **`tcpdump`** or **`tshark`** running on the Docker bridge or host network interface
- Filter on the UDP ports used by WebRTC (dynamically allocated, but can be constrained)
- Save as `.pcap` files, then post-process with `tshark` or `scapy` (Python) to extract per-packet features

---

## 4. Recommended Architecture (Modernized)

### 4.1 Overview

```
[Input Video] --> [Chrome Sender in Docker] --WebRTC--> [Chrome Receiver in Docker]
                        |                                        |
                        |                                        |
                   [tc/netem on                            [Record received
                    container network]                      video stream]
                        |
                   [tcpdump on Docker
                    bridge captures traffic]

Post-processing:
  Sent video + Received video --> VMAF score (ground truth label)
  pcap file --> packet sizes & inter-packet timings (model input features)
```

### 4.2 Technology Stack

| Component | Recommended Tool | Rationale |
|-----------|-----------------|-----------|
| **Language** | Python | Simpler automation, better ML ecosystem, easier pcap processing |
| **Browser Automation** | Selenium 4 with `webdriver-manager`, or Playwright | Modern, maintained, good Docker support |
| **WebRTC Application** | Self-hosted simple peer-to-peer page | No dependency on external services that can shut down |
| **Signaling Server** | Simple Node.js/Python WebSocket server | Minimal; just exchanges SDP/ICE |
| **Containerization** | Docker + docker-compose | Isolate sender, receiver, and network |
| **Network Impairment** | `tc`/`netem` on Docker container | Same approach as original, proven and standard |
| **Video Recording** | Chrome `--auto-select-desktop-capture-source` + MediaRecorder API, or `getDisplayMedia` | Replace deprecated `getRemoteStreams()` |
| **Traffic Capture** | `tcpdump` on host / `tshark` in container | Capture pcaps of WebRTC traffic |
| **Traffic Parsing** | `scapy` or `pyshark` (Python) | Extract packet sizes and timestamps |
| **VMAF Calculation** | FFmpeg with `libvmaf` filter | Modern, single-tool solution: `ffmpeg -i ref -i dist -lavfi libvmaf -f null -` |
| **Data Format** | CSV or Parquet per experiment | Easy to load into PyTorch/TensorFlow |

---

## 5. Step-by-Step Implementation Guide

### Step 1: Set Up the Self-Hosted WebRTC Application

**Why:** External services (AppRTC, OpenVidu demos) have shut down or will. You need a controlled, minimal WebRTC app.

**What to build:**
- A simple HTML page with two roles: "sender" and "receiver"
- A minimal WebSocket signaling server (Node.js or Python, ~50 lines)
- The sender captures video from `getUserMedia()` (which Chrome can feed from a fake device)
- The receiver displays and records the incoming stream using `MediaRecorder` API

**Key files:**
- `signaling-server.js` (or `.py`) - WebSocket relay for SDP/ICE
- `sender.html` - joins room, sends media
- `receiver.html` - joins room, receives and records media
- `Dockerfile` for the signaling server

**Chrome flags for sender:**
```
--use-fake-device-for-media-stream
--use-fake-ui-for-media-stream
--use-file-for-fake-video-capture=/path/to/test.y4m
--use-file-for-fake-audio-capture=/path/to/test.wav
--disable-rtc-smoothness-algorithm
--allow-running-insecure-content
--ignore-certificate-errors
```

### Step 2: Generate the Input Video

**Reuse and simplify** `generate_input_video.sh`:

```bash
# 1. Get a source video (or use a standard test sequence like Big Buck Bunny)
# 2. Scale, set framerate, overlay frame numbers
ffmpeg -i source.mp4 -ss 0 -t 30 -vf "scale=1280:720,setsar=1:1,\
  drawtext=text='%{frame_num}':start_number=1:x=(w-tw)/2:y=h-50:\
  fontcolor=white:fontsize=40:box=1:boxcolor=black@0.5:boxborderw=5" \
  -r 24 -pix_fmt yuv420p test.y4m

# 3. Generate audio
ffmpeg -f lavfi -i sine=frequency=440:duration=30 -ar 48000 -ac 2 test.wav
```

**Simplification:** Remove the padding+concat step from the original script. Instead, use a fixed-duration video and handle alignment purely through frame numbers during post-processing. This removes the complex and fragile padding detection code.

### Step 3: Containerized Test Environment

**docker-compose.yml:**
```yaml
services:
  signaling:
    build: ./signaling-server
    ports: ["8080:8080"]
    networks: [webrtc-net]

  sender:
    image: selenium/standalone-chrome:latest
    volumes:
      - ./test-media:/home/seluser/media
      - ./recordings:/home/seluser/recordings
    networks: [webrtc-net]
    cap_add: [NET_ADMIN]  # Required for tc/netem

  receiver:
    image: selenium/standalone-chrome:latest
    volumes:
      - ./recordings:/home/seluser/recordings
    networks: [webrtc-net]
    cap_add: [NET_ADMIN]

  traffic-capture:
    image: nicolaka/netshoot  # or custom with tcpdump
    network_mode: "service:sender"  # shares network namespace
    command: tcpdump -i eth0 -w /captures/capture.pcap udp
    volumes:
      - ./captures:/captures

networks:
  webrtc-net:
    driver: bridge
```

### Step 4: Orchestration Script (Python)

```python
# orchestrate.py - pseudocode
def run_experiment(network_params):
    """
    network_params: dict with keys like:
      - loss: 0-50 (percent)
      - delay: 0-500 (ms)
      - jitter: 0-100 (ms)
      - bandwidth: 100-10000 (kbps)
    """
    # 1. Start containers (docker-compose up)
    # 2. Start tcpdump in sender container
    # 3. Use Selenium to open sender.html in sender Chrome
    # 4. Use Selenium to open receiver.html in receiver Chrome
    # 5. Wait for WebRTC connection established
    # 6. Start MediaRecorder on receiver side (via JS injection)
    # 7. Apply network impairment:
    #    docker exec sender tc qdisc add dev eth0 root netem \
    #        loss {loss}% delay {delay}ms {jitter}ms
    # 8. Wait for test duration (e.g., 30 seconds)
    # 9. Reset network: tc qdisc del dev eth0 root
    # 10. Stop MediaRecorder, retrieve recording
    # 11. Stop tcpdump, retrieve pcap
    # 12. Post-process: compute VMAF, extract traffic features
    # 13. Save data point: (traffic_features, vmaf_score)
```

### Step 5: Apply Network Impairment

Use `tc`/`netem` with varied parameters. For training data diversity, sweep across:

| Parameter | Range | Step |
|-----------|-------|------|
| Packet loss | 0% - 50% | 5% |
| Delay | 0ms - 500ms | 50ms |
| Jitter | 0ms - 200ms | 25ms |
| Bandwidth limit | 100kbps - 10Mbps | varies |

Combination examples:
- Pure loss: (0%, 5%, 10%, 15%, 20%, 25%, 30%, 40%, 50%) x no delay
- Pure delay: no loss x (0, 50, 100, 200, 300, 500)ms
- Combined: loss x delay x jitter grid
- Bandwidth: tbf (Token Bucket Filter) for bandwidth limiting

Commands:
```bash
# Packet loss
tc qdisc add dev eth0 root netem loss 10%

# Fixed delay
tc qdisc add dev eth0 root netem delay 100ms

# Delay with jitter (normal distribution)
tc qdisc add dev eth0 root netem delay 100ms 50ms distribution normal

# Bandwidth limit (add after netem as child qdisc)
tc qdisc add dev eth0 root handle 1: netem delay 50ms loss 5%
tc qdisc add dev eth0 parent 1: handle 2: tbf rate 1mbit burst 32kbit latency 50ms

# Reset
tc qdisc del dev eth0 root
```

### Step 6: Record the Received Video

**Modern approach** (replaces deprecated `getRemoteStreams()`):

```javascript
// Inject into receiver page
const pc = /* get RTCPeerConnection reference */;
pc.ontrack = (event) => {
    const stream = event.streams[0];
    const recorder = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp8',
        videoBitsPerSecond: 3000000
    });
    const chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        // Download or transfer blob
    };
    recorder.start();
    setTimeout(() => recorder.stop(), TEST_DURATION_MS);
};
```

### Step 7: Capture and Parse Network Traffic

**During the experiment:**
```bash
# In sender container (or on Docker bridge)
tcpdump -i eth0 -w /captures/experiment_001.pcap udp
```

**Post-processing (Python):**
```python
from scapy.all import rdpcap, UDP

def extract_traffic_features(pcap_path):
    packets = rdpcap(pcap_path)
    features = []
    for pkt in packets:
        if UDP in pkt:
            features.append({
                'timestamp': float(pkt.time),
                'size': len(pkt),
                'src_port': pkt[UDP].sport,
                'dst_port': pkt[UDP].dport,
            })

    # Compute inter-packet times
    for i in range(1, len(features)):
        features[i]['inter_packet_time'] = (
            features[i]['timestamp'] - features[i-1]['timestamp']
        )
    features[0]['inter_packet_time'] = 0.0
    return features
```

### Step 8: Compute VMAF (Ground Truth)

**Modern approach using FFmpeg's libvmaf filter** (replaces the old `run_vmaf` script):

```bash
# Convert received video to same format as reference
ffmpeg -i received.webm -vf "scale=1280:720,fps=24" -pix_fmt yuv420p received.y4m

# Compute VMAF (per-frame scores in JSON)
ffmpeg -i received.y4m -i reference.y4m \
  -lavfi "libvmaf=model=version=vmaf_v0.6.1:log_path=vmaf_output.json:log_fmt=json" \
  -f null -
```

**Per-frame VMAF extraction:**
```python
import json

with open('vmaf_output.json') as f:
    data = json.load(f)
    per_frame_vmaf = [frame['metrics']['vmaf'] for frame in data['frames']]
    mean_vmaf = data['pooled_metrics']['vmaf']['mean']
```

### Step 9: Align Traffic and VMAF Data

Each experiment produces:
1. A **pcap file** covering the call duration
2. A **VMAF JSON** with per-frame scores

For the deep learning model, you need to create windowed samples:

```python
def create_training_samples(traffic_features, mean_vmaf, window_size_sec=5):
    """
    Split the traffic trace into windows and label each with the
    overall VMAF score for that experiment.
    """
    samples = []
    start_time = traffic_features[0]['timestamp']
    end_time = traffic_features[-1]['timestamp']

    t = start_time
    while t + window_size_sec <= end_time:
        window_packets = [p for p in traffic_features
                         if t <= p['timestamp'] < t + window_size_sec]
        sizes = [p['size'] for p in window_packets]
        ipts = [p['inter_packet_time'] for p in window_packets]

        samples.append({
            'packet_sizes': sizes,
            'inter_packet_times': ipts,
            'label': mean_vmaf  # or per-window VMAF if aligned
        })
        t += window_size_sec

    return samples
```

### Step 10: Run at Scale and Build Dataset

```python
import itertools

loss_values = [0, 5, 10, 15, 20, 25, 30, 40, 50]
delay_values = [0, 50, 100, 200, 300]
jitter_values = [0, 25, 50, 100]
repeats = 3  # statistical robustness

for loss, delay, jitter in itertools.product(loss_values, delay_values, jitter_values):
    for r in range(repeats):
        run_experiment({
            'loss': loss,
            'delay': delay,
            'jitter': jitter,
            'experiment_id': f"L{loss}_D{delay}_J{jitter}_R{r}"
        })
```

**Expected dataset size:** 9 x 5 x 4 x 3 = **540 experiments**. At 30 seconds each with 5-second windows, that's ~3,240 training samples. Increase repeats or add bandwidth variation for more data.

---

## 6. What to Remove from the Existing Code

| Component | Action | Reason |
|-----------|--------|--------|
| `AppRtcBasicTest.java`, `AppRtcAdvancedTest.java` | DELETE | AppRTC is dead |
| `OpenVidu*Test.java` (all 4) | DELETE | External demo dependency, API changed |
| `JanusConferenceTest.java` | DELETE | External service dependency |
| `WebRtcSamples*Test.java` (all 6) | DELETE | Mostly demo tests, not data generation |
| `ElasTestRemoteControlParent.java` | REWRITE | Outdated APIs (`getRemoteStreams`), Java 8, Selenium 3 |
| `elastest-remote-control.js` | REWRITE | Uses deprecated `getLocalStreams`/`getRemoteStreams`, old RecordRTC |
| `pom.xml` | REPLACE | Switch to Python project with `requirements.txt` |
| Complex padding logic in `calculate_qoe_metrics.sh` | SIMPLIFY | Use FFmpeg libvmaf filter directly |
| `gocr` OCR alignment | REPLACE with Tesseract or eliminate | gocr is unmaintained and less accurate |
| VQMT dependency | REMOVE | FFmpeg handles SSIM/PSNR natively |
| PESQ/ViSQOL | OPTIONAL REMOVE | Focus on VMAF as primary metric for video QoE |

### What to Keep (as reference/adaptation)

| Component | Action | Reason |
|-----------|--------|--------|
| `generate_input_video.sh` | ADAPT | Core FFmpeg pipeline is sound; simplify padding |
| `calculate_qoe_metrics.sh` | ADAPT | Use as reference for new Python-based VMAF calculation |
| Network simulation logic (`tc`/`netem`) | KEEP the approach | Proven, correct method for network impairment |
| Frame numbering overlay | KEEP | Essential for temporal alignment of sent/received frames |
| Chrome fake device flags | KEEP | Standard approach, still works in modern Chrome |
| `data/*.csv` | KEEP | Reference data for validation |

---

## 7. Tools and Dependencies

### Required Software

| Tool | Version | Purpose |
|------|---------|---------|
| **Docker + Docker Compose** | Latest | Container orchestration |
| **Python** | 3.10+ | Orchestration, data processing, ML |
| **FFmpeg** | 6.0+ (with `--enable-libvmaf`) | Video processing, VMAF calculation |
| **Chrome/Chromium** | Latest | WebRTC client (via Selenium containers) |
| **Node.js** | 18+ | Signaling server |
| **tcpdump** or **tshark** | Latest | Traffic capture |

### Python Packages

```
selenium>=4.15
webdriver-manager
scapy
pyshark  # alternative to scapy for pcap parsing
numpy
pandas
torch  # or tensorflow, for eventual model training
```

### Docker Images

```
selenium/standalone-chrome:latest
node:18-alpine  # for signaling server
nicolaka/netshoot  # for traffic capture tools
```

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Chrome fake video flags change behavior | Low | High | Pin Chrome version in Docker image |
| `MediaRecorder` output format issues | Medium | Medium | Use VP8/WebM, test alignment thoroughly |
| VMAF frame alignment errors | Medium | High | Frame numbers + robust alignment code |
| `tc`/`netem` insufficient for realistic impairment | Low | Medium | Add bandwidth shaping with `tbf` |
| Docker networking complexity | Medium | Medium | Test with simple ping/iperf first |
| Large dataset storage | Low | Low | ~500MB per experiment; total ~250GB manageable |
| WebRTC codec changes affecting quality | Low | Medium | Fix VP8 codec in SDP constraints |

---

## 9. Estimated Effort Breakdown

| Task | Complexity |
|------|-----------|
| Build self-hosted WebRTC app + signaling server | Moderate |
| Adapt input video generation script | Simple |
| Set up Docker Compose environment | Moderate |
| Write Python orchestration script | Moderate |
| Implement traffic capture + parsing | Simple |
| Implement VMAF calculation pipeline | Simple |
| Implement frame alignment logic | Moderate |
| Build data aggregation pipeline | Simple |
| Run experiments (540 runs automated) | Mostly automated compute time |
| Validate dataset quality | Moderate |

---

## 10. Recommendations

1. **Start with a minimal proof-of-concept**: Get one experiment running end-to-end (send video -> capture traffic -> record received video -> compute VMAF) before scaling.

2. **Use Python for everything except the WebRTC app**: The existing Java/Maven stack adds unnecessary complexity. Python with Selenium 4 can do the same thing with fewer lines.

3. **Self-host the WebRTC application**: This is non-negotiable. All external services the original project depended on are dead or unreliable. A simple HTML+JS page with a WebSocket signaling server is sufficient.

4. **Keep the video simple**: Use a single well-known test video (e.g., Big Buck Bunny segment). The content variety is not important; the network conditions are what vary.

5. **Capture traffic at the Docker bridge level**: This gives you the network provider perspective (encrypted packets, no payload visibility) which matches your inference scenario.

6. **Focus on VMAF only**: Drop PESQ, ViSQOL, VQMT. VMAF is the most relevant single metric for video QoE and is what Netflix, YouTube, and academia use.

7. **Run on Linux**: The `tc`/`netem` tooling for network impairment is Linux-only. Use a Linux VM or cloud instance if your development machine is Windows.

8. **Consider using `tc-netem` profiles**: Rather than just fixed values, use time-varying network profiles (e.g., "good for 10s, then bad for 5s, then recover") — this creates more realistic and diverse training data.

---

## 11. Conclusion

The project is **feasible**. The existing codebase provides a validated architecture and useful reference implementations, but it needs substantial modernization. The most critical gap is the absence of network traffic capture, which is the entire input to your model. The recommended approach is to:

1. Build a minimal self-hosted WebRTC app (replacing all dead external dependencies)
2. Rewrite the orchestration in Python (replacing the Java/Selenium 3 stack)
3. Add `tcpdump`-based traffic capture as a new component
4. Modernize the VMAF calculation using FFmpeg's built-in `libvmaf` filter
5. Automate the entire pipeline for large-scale data generation

The end result will be a dataset of (packet_sizes_sequence, inter_packet_times_sequence) -> VMAF_score pairs suitable for training a deep learning model for network-side QoE inference.
