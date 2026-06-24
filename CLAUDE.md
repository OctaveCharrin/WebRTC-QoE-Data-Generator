# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A data-generation harness, not an application. It drives two headless Chrome
browsers through a real WebRTC call, degrades the network between them, and
records two aligned signals per run: the received video's VMAF (the label) and
the UDP packet trace (the features). The product is `output/dataset/dataset.csv`
plus per-experiment `.npy` arrays — training data for a model that predicts QoE
from network traffic.

## Commands

```bash
uv sync                                # install deps (Python 3.11+, uv-managed)

# One-time setup before any run:
uv run python run.py generate-video    # build media/{test.y4m,test.wav,reference.y4m}
docker compose up -d --build           # start signaling + sender + receiver containers

# Core loop:
uv run python run.py run --loss 0 10 --delay 0 --jitter 0 --repeats 1 --duration 15
uv run python run.py build-dataset     # consolidate result.json files -> dataset.csv
uv run python run.py summary           # VMAF distribution + per-loss breakdown

# Debugging frame alignment (the most error-prone part):
uv run python run.py debug-alignment L0.0_D0_J0_B1000_R0 [--step N]
uv run python run.py run ... --debug-frames     # generate comparison PNGs inline
uv run python run.py run ... -v                 # verbose logging
```

There is **no test suite, linter, or build step.** Verification is manual: run a
small grid and inspect VMAF ranges and the `debug-alignment` PNGs.

The `run` command is **resumable** — it skips any experiment whose
`output/<id>/result.json` already exists. Use `--no-resume` to force re-runs.

## Architecture

The host Python process (`pipeline/orchestrator.py`) is the brain; everything
heavy runs in Docker and is controlled from the host:

```
Host (orchestrator, Selenium client, ffmpeg, scapy)
  ├─ docker exec ─> sender container:   Chrome (fake Y4M capture) + tc/netem + tcpdump
  ├─ Selenium    ─> receiver container: Chrome + MediaRecorder
  └─ WebSocket   ─> signaling container: aiohttp relay (signaling/server.py)
```

- **Containers are long-lived; browsers are reused.** `run_all()` connects to
  both Selenium grids once, then loops experiments, refreshing pages between
  them. A failed experiment triggers a page refresh, and an unrecoverable
  browser triggers a full reconnect.
- **All network impairment and capture happen via `docker exec`** into the
  sender container (`pipeline/network.py`), not on the host. The containers need
  `NET_ADMIN`/`NET_RAW` caps (set in `docker-compose.yml`) for `tc` and
  `tcpdump`. tcpdump captures `udp` only (WebRTC media is RTP/UDP).
- **netem is applied as a single combined qdisc.** Loss, delay, and jitter go
  into one `netem` command (only one root qdisc is allowed); the bitrate cap is
  a `tbf` qdisc (child of netem when netem is present, root otherwise).
  `apply_netem` happens *after* recording starts so padding frames arrive clean;
  `reset_netem` always runs in the experiment's `finally` path to avoid leaking
  impairment into the next run.
- **Bitrate is a swept axis** (`config.bitrate_values`, `NetworkCondition.
  bitrate_kbps`). Each value is pinned both at the encoder (signaling page, via
  the `maxbitrate` URL param) *and* on the path (tbf), so the encoded bitrate
  equals the swept value and VMAF depends on it. `realized_bitrate_kbps` (wire
  bytes/sec from the pcap) is recorded to verify the cap took effect. The
  surrogate + reward are fitted over `(bitrate, loss, delay, jitter)`.

### The frame-alignment problem (central design constraint)

Sender and receiver start at unpredictable offsets, so received frames cannot be
matched to reference frames by index alone. The whole media format exists to
solve this:

- `generate_input_video.sh` builds `test.y4m` as
  **`[5s testsrc color bars] + [30s content with burned-in frame numbers] + [5s color bars]`**,
  and a separate **`reference.y4m`** that is content-only (the VMAF reference).
- At experiment start, `BrowserController.reset_media_tracks()` calls
  `getUserMedia()` again and `replaceTrack()` to **restart the fake capture from
  frame 0** without renegotiation, so the receiver records the full
  padding→content→padding structure.
- `pipeline/vmaf.py::detect_padding_boundaries` finds the content region by
  sampling known testsrc bar colors (`TESTSRC_COLORS`) at `y = height/3`, then
  trims the received video to content-only before VMAF.
- **Padding detection alone is not enough**: Chrome's looping fake capture does
  not reliably restart at frame 0 on `replaceTrack`, so the received content
  usually leads the reference by ~10-15 frames and the testsrc padding is often
  never captured. `compute_vmaf` therefore runs a **frame-offset search**
  (`estimate_frame_offset`, 8×8 gray fingerprints on the overlay-cropped region)
  and drops the leading frames of whichever stream leads before scoring.
  Skipping this step silently caps VMAF at ~68 even on a perfect image —
  symptom of a temporal shift, not quality loss. Re-score saved recordings after
  any VMAF/alignment change with `run.py reprocess-vmaf` (no WebRTC re-capture).
- The burned-in frame-number overlay would penalize VMAF on ±1-frame
  misalignment, so it is excluded two ways, controlled by `--vmaf-mode`
  (default `both`): **cropped** removes the bottom ~20% of the frame; **masked**
  paints a white box over just the overlay (`Config.calculate_mask_region`,
  whose geometry must stay in sync with the `drawtext` params in the generate
  script). Each mode produces its own VMAF JSON, mean, and per-frame `.npy`.

If you touch the overlay position, font size, padding duration, or video
resolution, the crop height, mask region, and padding-detection all depend on
those values — change them together (`config.py` mirrors the generate script's
defaults).

### Per-experiment output

`run_single` writes a directory `output/<experiment_id>/` (id format
`L{loss}_D{delay}_J{jitter}_B{bitrate}_R{repeat}`) containing `result.json` plus
`.npy` arrays. Intermediate `.y4m` files are deleted after VMAF. Two
time-aligned coordinate systems are saved so models can window packets against
frames: `packet_timestamps` is seconds from first packet; `frame_times` is
seconds from content start (`i / fps`). `build_dataset` simply globs
`*/result.json` into one CSV — it does not re-run anything.

## Pipeline module map

| Module | Responsibility |
|--------|----------------|
| `pipeline/orchestrator.py` | Experiment grid + the timed per-experiment sequence (the canonical control flow) |
| `pipeline/browser.py` | Selenium control; bridges to JS functions in `signaling/static/index.html` (`resetMediaTracks`, `startRecording`, `getRecordingBase64`) via `window.__connectionState` polling |
| `pipeline/network.py` | `docker exec` for tc/netem + tcpdump; `NetworkCondition` dataclass owns the experiment_id format |
| `pipeline/vmaf.py` | WebM→Y4M, padding detection, trim, cropped/masked VMAF, debug PNGs |
| `pipeline/traffic.py` | scapy pcap parsing → packet sizes / inter-packet times / timestamps |
| `pipeline/dataset.py` | result.json → dataset.csv; `load_experiment` is the training-side loader |
| `pipeline/surrogate.py` | fits the multilinear QoS→VMAF surrogate from dataset.csv; writes `output/reward_model.{npz,pkl}` + a portable copy in `reward/` |
| `reward/qos_vmaf_reward.py` | **self-contained numpy-only** reward module copied into `../rl-mpquic`; `QoSVmafReward.vmaf(bitrate_kbps, loss_pct, delay_ms, jitter_ms)` |
| `signaling/static/index.html` | Browser-side WebRTC + MediaRecorder; recordings transferred to host as base64; caps the encoder bitrate (setParameters + SDP `b=AS`) from the `maxbitrate` URL param |

## External dependencies (must exist on host)

- **FFmpeg with libvmaf** (`ffmpeg -filters | grep libvmaf`) — `run` aborts
  without it unless `--skip-vmaf`. Also needs `ffprobe`.
- **Docker + Compose**, **uv**, and **wget/curl** (for `generate-video`'s source
  download).

## Debugging entry points

- noVNC live browser view: `http://localhost:7900` (sender),
  `http://localhost:7901` (receiver), password `secret`.
- Signaling health: `http://localhost:8080/health`.
- Container logs: `docker compose logs {signaling,sender,receiver}`.
- Suspect VMAF scores → run `debug-alignment` and inspect the side-by-side PNGs
  before assuming the model or capture is wrong; misalignment is the usual cause.
