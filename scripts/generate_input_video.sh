#!/usr/bin/env bash
# ============================================================================
# Generate test input video for the WebRTC QoE data generation pipeline.
#
# Produces:
#   media/test.y4m       — YUV4MPEG2 video WITH padding (fed to Chrome)
#   media/test.wav       — Audio WITH padding tone (fed to Chrome)
#   media/reference.y4m  — Content-only video WITHOUT padding (VMAF reference)
#
# The video structure is:
#   [5s padding: testsrc color bars] + [30s content with frame numbers] + [5s padding]
#
# The padding allows automatic detection of where the actual content starts
# and ends in the received recording, which is essential for accurate VMAF
# computation. Without it, the variable delay between sender and receiver
# makes frame alignment unreliable.
#
# Adapted from elastest-webrtc-qoe-meter/scripts/generate_input_video.sh.
# ============================================================================

set -euo pipefail

# --- Defaults ----------------------------------------------------------------
VIDEO_SAMPLE_URL="${1:-https://archive.org/download/e-dv548_lwe08_christa_casebeer_003.ogg/e-dv548_lwe08_christa_casebeer_003.mp4}"
WIDTH="${2:-640}"
HEIGHT="${3:-480}"
FPS="${4:-24}"
DURATION_SEC="${5:-30}"
PADDING_SEC="${6:-5}"
OUTPUT_DIR="${7:-media}"

# --- Setup -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"

# Total duration including padding on both sides
TOTAL_SEC=$((PADDING_SEC + DURATION_SEC + PADDING_SEC))

echo "=== WebRTC QoE — Input Video Generator ==="
echo "  Resolution : ${WIDTH}x${HEIGHT}"
echo "  Frame rate : ${FPS} fps"
echo "  Content    : ${DURATION_SEC}s"
echo "  Padding    : ${PADDING_SEC}s on each side"
echo "  Total      : ${TOTAL_SEC}s"
echo "  Output dir : ${OUTPUT_DIR}"
echo ""

# --- Step 1: Get source video ------------------------------------------------
SOURCE_FILE="${OUTPUT_DIR}/source.mp4"
if [ -f "$SOURCE_FILE" ]; then
    echo "[1/6] Source video already exists: ${SOURCE_FILE}"
elif [ -f "$VIDEO_SAMPLE_URL" ]; then
    echo "[1/6] Copying local source video..."
    cp "$VIDEO_SAMPLE_URL" "$SOURCE_FILE"
else
    echo "[1/6] Downloading source video..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$SOURCE_FILE" "$VIDEO_SAMPLE_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$SOURCE_FILE" "$VIDEO_SAMPLE_URL"
    else
        echo "ERROR: Neither wget nor curl found. Install one of them."
        exit 1
    fi
fi

# --- Step 2: Process content video -------------------------------------------
# Scale to target resolution, set frame rate, overlay frame numbers.
echo "[2/6] Processing content video: ${WIDTH}x${HEIGHT}@${FPS}fps, ${DURATION_SEC}s..."
ffmpeg -loglevel warning -y \
    -i "$SOURCE_FILE" \
    -ss 00:00:00 -t "$DURATION_SEC" \
    -vf "scale=${WIDTH}:${HEIGHT}:force_original_aspect_ratio=decrease,\
pad=${WIDTH}:${HEIGHT}:(ow-iw)/2:(oh-ih)/2,\
setsar=1:1,\
drawtext=text='%{frame_num}':start_number=1:\
x=(w-tw)/2:y=h-(2*lh)+15:\
fontcolor=black:fontsize=40:\
box=1:boxcolor=white:boxborderw=10" \
    -r "$FPS" \
    -pix_fmt yuv420p \
    "$OUTPUT_DIR/content.mp4"

# --- Step 3: Generate padding video ------------------------------------------
# Uses ffmpeg's testsrc which produces a distinctive color bar pattern.
# This pattern is reliably detectable in the received video to find where
# the actual content starts and ends.
echo "[3/6] Generating ${PADDING_SEC}s padding (testsrc color bars)..."
ffmpeg -loglevel warning -y \
    -f lavfi -i "testsrc=duration=${PADDING_SEC}:size=${WIDTH}x${HEIGHT}:rate=${FPS}" \
    -f lavfi -i "sine=frequency=1000:duration=${PADDING_SEC}" \
    -ar 48000 -ac 2 \
    -pix_fmt yuv420p \
    "$OUTPUT_DIR/padding.mp4"

# --- Step 4: Concatenate: padding + content + padding ------------------------
echo "[4/6] Concatenating: [padding ${PADDING_SEC}s] + [content ${DURATION_SEC}s] + [padding ${PADDING_SEC}s]..."
ffmpeg -loglevel warning -y \
    -i "$OUTPUT_DIR/padding.mp4" \
    -i "$OUTPUT_DIR/content.mp4" \
    -i "$OUTPUT_DIR/padding.mp4" \
    -filter_complex "concat=n=3:v=1:a=1" \
    -pix_fmt yuv420p \
    "$OUTPUT_DIR/test_combined.mp4"

# Convert to Y4M (for Chrome fake capture) and WAV (for fake audio)
ffmpeg -loglevel warning -y -i "$OUTPUT_DIR/test_combined.mp4" -pix_fmt yuv420p "$OUTPUT_DIR/test.y4m"
ffmpeg -loglevel warning -y -i "$OUTPUT_DIR/test_combined.mp4" -vn -acodec pcm_s16le -ar 48000 -ac 2 "$OUTPUT_DIR/test.wav"

# --- Step 5: Create reference video (content only, NO padding) ---------------
# This is what we compare against for VMAF. It must NOT have padding frames.
echo "[5/6] Creating reference video (content only, no padding)..."
ffmpeg -loglevel warning -y \
    -i "$OUTPUT_DIR/content.mp4" \
    -pix_fmt yuv420p \
    "$OUTPUT_DIR/reference.y4m"

# --- Step 6: Clean up intermediate files -------------------------------------
echo "[6/6] Cleaning up intermediate files..."
rm -f "$OUTPUT_DIR/content.mp4" "$OUTPUT_DIR/padding.mp4" "$OUTPUT_DIR/test_combined.mp4"

# --- Summary -----------------------------------------------------------------
echo ""
echo "Done. Generated files:"
ls -lh "$OUTPUT_DIR/test.y4m" "$OUTPUT_DIR/test.wav" "$OUTPUT_DIR/reference.y4m"
echo ""
echo "Video structure: [${PADDING_SEC}s padding] + [${DURATION_SEC}s content] + [${PADDING_SEC}s padding]"
echo "Reference video: ${DURATION_SEC}s content only (for VMAF comparison)"
echo ""
echo "To verify, play with: ffplay ${OUTPUT_DIR}/test.y4m"
