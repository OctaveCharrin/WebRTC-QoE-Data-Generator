from __future__ import annotations

"""
Network traffic parsing from pcap files using scapy.

This is entirely new functionality — the original project had NO traffic
capture capability, which was identified as the critical gap in the
feasibility report.

Extracts per-packet features (sizes and inter-packet timings) that serve
as input to the deep learning QoE inference model.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def parse_pcap(pcap_path: Path) -> dict:
    """
    Parse a pcap file and extract UDP packet features.

    Args:
        pcap_path: Path to the .pcap file captured by tcpdump.

    Returns:
        Dictionary with:
          - timestamps:         np.array of float64 (epoch seconds, microsecond precision)
          - packet_sizes:       np.array of int32 (total packet size in bytes)
          - inter_packet_times: np.array of float64 (seconds between consecutive packets)
          - src_ports:          np.array of int32
          - dst_ports:          np.array of int32
          - total_packets:      int
          - duration_sec:       float (time span of the capture)
    """
    # Import scapy here to avoid slow import at module level
    from scapy.all import rdpcap, UDP

    logger.info(f"Parsing pcap: {pcap_path.name}")
    packets = rdpcap(str(pcap_path))

    timestamps = []
    sizes = []
    src_ports = []
    dst_ports = []

    for pkt in packets:
        if UDP in pkt:
            timestamps.append(float(pkt.time))
            sizes.append(len(pkt))
            src_ports.append(pkt[UDP].sport)
            dst_ports.append(pkt[UDP].dport)

    if not timestamps:
        logger.warning(f"No UDP packets found in {pcap_path.name}")
        return {
            "timestamps": np.array([], dtype=np.float64),
            "packet_sizes": np.array([], dtype=np.int32),
            "inter_packet_times": np.array([], dtype=np.float64),
            "src_ports": np.array([], dtype=np.int32),
            "dst_ports": np.array([], dtype=np.int32),
            "total_packets": 0,
            "duration_sec": 0.0,
        }

    timestamps = np.array(timestamps, dtype=np.float64)
    sizes = np.array(sizes, dtype=np.int32)
    src_ports = np.array(src_ports, dtype=np.int32)
    dst_ports = np.array(dst_ports, dtype=np.int32)

    # Compute inter-packet times
    if len(timestamps) > 1:
        ipt = np.diff(timestamps)
        ipt = np.insert(ipt, 0, 0.0)
    else:
        ipt = np.array([0.0], dtype=np.float64)

    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

    logger.info(
        f"Parsed {len(timestamps)} UDP packets over {duration:.1f}s "
        f"(avg size: {sizes.mean():.0f} bytes)"
    )

    return {
        "timestamps": timestamps,
        "packet_sizes": sizes,
        "inter_packet_times": ipt,
        "src_ports": src_ports,
        "dst_ports": dst_ports,
        "total_packets": len(timestamps),
        "duration_sec": duration,
    }


def save_traffic_features(
    traffic_data: dict, output_dir: Path, experiment_id: str
) -> dict[str, Path]:
    """
    Save traffic features as numpy .npy files for model training.

    Arrays saved:
      - packet_sizes:       shape (N,) int32
      - inter_packet_times: shape (N,) float64
      - packet_timestamps:  shape (N,) float64 — seconds relative to first packet

    Returns:
        Dictionary mapping feature name to the saved file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for key in ["packet_sizes", "inter_packet_times"]:
        path = output_dir / f"{experiment_id}_{key}.npy"
        np.save(path, traffic_data[key])
        paths[key] = path

    # Save timestamps relative to the first packet (t=0 at capture start)
    ts = traffic_data["timestamps"]
    relative_ts = ts - ts[0] if len(ts) > 0 else ts
    path = output_dir / f"{experiment_id}_packet_timestamps.npy"
    np.save(path, relative_ts)
    paths["packet_timestamps"] = path

    logger.info(
        f"Saved traffic features for {experiment_id}: "
        f"{traffic_data['total_packets']} packets"
    )
    return paths
