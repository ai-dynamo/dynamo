#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Qualify MiniMax-H3 T2VA against one already-running aggregated worker.
set -euo pipefail

API_URL="${DYN_H3_API_URL:-http://127.0.0.1:8000/v1/videos}"
MODEL="${DYN_H3_MODEL:-MiniMaxAI/MiniMax-H3}"
MODEL_REVISION="${DYN_H3_MODEL_REVISION:-42ed227ee7df40d41602854ae760620d6eb651fe}"
FASTH3_LORA_PATH="${DYN_H3_FASTH3_LORA_PATH:-}"
FASTH3_VARIANT="${DYN_H3_FASTH3_VARIANT:-}"
FASTH3_REVISION="${DYN_H3_FASTH3_REVISION:-bcf40ca6f457ed66f8badf13514943e390205fca}"
FASTH3_SHA256="${DYN_H3_FASTH3_SHA256:-}"
ATTENTION_BACKEND="${DYN_H3_ATTENTION_BACKEND:-}"
FASTVIDEO_VSA_TOPK="${DYN_H3_FASTVIDEO_VSA_TOPK:-64}"
FASTVIDEO_VSA_SM100A="${DYN_H3_FASTVIDEO_VSA_SM100A:-0}"
QUAL_DIR="${DYN_H3_QUAL_DIR:-/tmp/dynamo_minimax_h3_qualification}"
OUTPUT_DIR="$QUAL_DIR/outputs"
CASE_NAME="cat-playing-canon-in-d-grand-piano"
REQUEST_FILE="$OUTPUT_DIR/${CASE_NAME}.request.json"
RESPONSE_FILE="$OUTPUT_DIR/${CASE_NAME}.response.json"
VIDEO_FILE="$OUTPUT_DIR/${CASE_NAME}.mp4"
PROBE_FILE="$OUTPUT_DIR/${CASE_NAME}.ffprobe.json"
SOURCE_FILE="$OUTPUT_DIR/${CASE_NAME}.source-url.txt"
SHA_FILE="$OUTPUT_DIR/${CASE_NAME}.sha256"
HARDWARE_FILE="$OUTPUT_DIR/${CASE_NAME}.hardware.json"
METADATA_FILE="$OUTPUT_DIR/${CASE_NAME}.qualification.json"
AUDIO_STATS_FILE="$OUTPUT_DIR/${CASE_NAME}.audio-stats.txt"
TIMING_FILE="$OUTPUT_DIR/${CASE_NAME}.curl-time-seconds.txt"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
DYNAMO_REVISION="${DYN_H3_DYNAMO_REVISION:-$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || true)}"
IMAGE_REF="${DYN_H3_IMAGE_REF:-}"

if [[ -z "$DYNAMO_REVISION" ]]; then
    echo "Set DYN_H3_DYNAMO_REVISION to the tested Dynamo commit" >&2
    exit 1
fi
if [[ -z "$IMAGE_REF" ]]; then
    echo "Set DYN_H3_IMAGE_REF to the tested container image ID or digest" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ -n "$FASTH3_VARIANT" && -z "$FASTH3_LORA_PATH" ]]; then
    echo "DYN_H3_FASTH3_VARIANT requires DYN_H3_FASTH3_LORA_PATH" >&2
    exit 1
fi
if [[ -n "$FASTH3_LORA_PATH" ]]; then
    if [[ ! -f "$FASTH3_LORA_PATH" ]]; then
        echo "FastH3 adapter not found: $FASTH3_LORA_PATH" >&2
        exit 1
    fi
    if [[ -z "$FASTH3_VARIANT" ]]; then
        FASTH3_VARIANT="$(basename "$(dirname "$FASTH3_LORA_PATH")")"
    fi
    if [[ -z "$FASTH3_SHA256" ]]; then
        case "$FASTH3_VARIANT" in
            dense-datafree)
                FASTH3_SHA256="4ce198c83132251b7fd0de2503823aa49c53983f068318f66cb19eaefb7fcc12"
                ;;
            vsa-datafree)
                FASTH3_SHA256="42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470"
                ;;
            vsa-synthetic-step1300)
                FASTH3_SHA256="de6af1ea8b2f4b31a7ac3752b4836f88671199fdf277a5dd922f9cca3bea7b65"
                ;;
            vsa-synthetic-step1900)
                FASTH3_SHA256="bbac632dffb828d99123ab06c7ff3efc246351bed63d1860b4252e61dafeaed8"
                ;;
            *)
                echo "Set DYN_H3_FASTH3_SHA256 for unrecognized variant: $FASTH3_VARIANT" >&2
                exit 1
                ;;
        esac
    fi
    if [[ "$FASTH3_VARIANT" == vsa-* ]]; then
        ATTENTION_BACKEND="${ATTENTION_BACKEND:-FASTVIDEO_VSA}"
        if [[ "$ATTENTION_BACKEND" != "FASTVIDEO_VSA" ]]; then
            echo "FastH3 VSA qualification requires FASTVIDEO_VSA" >&2
            exit 1
        fi
        if [[ ! "$FASTVIDEO_VSA_TOPK" =~ ^[1-9][0-9]*$ ]]; then
            echo "DYN_H3_FASTVIDEO_VSA_TOPK must be a positive integer" >&2
            exit 1
        fi
    else
        ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRTLLM_ATTN}"
    fi
    actual_fasth3_sha256="$(sha256sum "$FASTH3_LORA_PATH" | awk '{print $1}')"
    if [[ "$actual_fasth3_sha256" != "$FASTH3_SHA256" ]]; then
        echo "FastH3 adapter checksum mismatch: $actual_fasth3_sha256" >&2
        exit 1
    fi
    inference_steps=4
    fasth3=true
else
    inference_steps=50
    fasth3=false
fi

jq -n \
    --arg model "$MODEL" \
    --arg prompt "A photorealistic orange tabby cat seated at a polished black grand piano, visibly pressing the keys with both front paws while performing Pachelbel's Canon in D. Elegant concert hall, cinematic lighting, realistic paw and key motion, synchronized clear solo grand-piano audio playing the recognizable Canon in D melody, no speech, no other instruments." \
    --argjson inference_steps "$inference_steps" \
    --argjson fasth3 "$fasth3" \
    '{
        model: $model,
        prompt: $prompt,
        size: "448x256",
        response_format: "url",
        output_format: "mp4",
        nvext: {
            fps: 24,
            num_inference_steps: $inference_steps,
            seed: 42
        },
        task: "t2va",
        duration: 10.0,
        aspect_ratio: "16:9"
    } | if $fasth3 then . else . + {
        flow_shift: 12.0,
        audio_flow_shift: 3.0
    } end' > "$REQUEST_FILE"

python3 - "$HARDWARE_FILE" <<'PY'
import json
import sys

import torch

devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
if len(devices) != 4:
    raise SystemExit(f"Expected exactly 4 visible GPUs, found {len(devices)}: {devices}")
supported_products = ("B200", "GB300")
if not any(all(product in device.upper() for device in devices) for product in supported_products):
    raise SystemExit(f"Expected 4 identical B200 or GB300 GPUs, found: {devices}")

with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump({"visible_gpu_count": len(devices), "devices": devices}, output, indent=2)
    output.write("\n")
PY

models_url="${API_URL%/v1/videos}/v1/models"
echo "Waiting for MiniMax-H3 to register with the frontend..."
for _ in $(seq 1 360); do
    if curl -fsS --max-time 10 "$models_url" | jq -e \
        --arg model "$MODEL" '.data[]? | select(.id == $model)' >/dev/null; then
        break
    fi
    sleep 10
done
curl -fsS --max-time 10 "$models_url" | jq -e \
    --arg model "$MODEL" '.data[]? | select(.id == $model)' >/dev/null

echo "Generating a 10-second MiniMax-H3 T2VA sample..."
curl -fsS --max-time 7200 "$API_URL" \
    -H 'Content-Type: application/json' \
    --data-binary "@$REQUEST_FILE" \
    --output "$RESPONSE_FILE" \
    --write-out '%{time_total}\n' > "$TIMING_FILE"
jq -e '.status == "completed" and (.data | length) >= 1' "$RESPONSE_FILE" >/dev/null

media_url="$(jq -er '.data[0].url' "$RESPONSE_FILE")"
media_path="${media_url#file://}"
if [[ ! -s "$media_path" ]]; then
    echo "Missing generated video: $media_path" >&2
    exit 1
fi
cp "$media_path" "$VIDEO_FILE"
printf '%s\n' "$media_url" > "$SOURCE_FILE"

ffprobe -v error \
    -show_entries stream=codec_type,codec_name,width,height,sample_rate,channels,r_frame_rate,start_time,duration \
    -show_entries format=duration,size \
    -of json "$VIDEO_FILE" > "$PROBE_FILE"

expected_size="$(jq -er '.size' "$REQUEST_FILE")"
expected_width="${expected_size%x*}"
expected_height="${expected_size#*x}"

jq -e \
    --argjson expected_width "$expected_width" \
    --argjson expected_height "$expected_height" '
    def abs: if . < 0 then -. else . end;
    ([.streams[] | select(.codec_type == "video")][0]) as $video |
    ([.streams[] | select(.codec_type == "audio")][0]) as $audio |
    ($video.codec_name == "h264") and
    ($video.r_frame_rate == "24/1") and
    ($video.width == $expected_width) and
    ($video.height == $expected_height) and
    ($audio.codec_name == "aac") and
    ($audio.sample_rate == "32000") and
    ($audio.channels == 2) and
    ((($video.start_time | tonumber) - ($audio.start_time | tonumber)) | abs) <= 0.10 and
    (((($video.start_time | tonumber) + ($video.duration | tonumber)) -
       (($audio.start_time | tonumber) + ($audio.duration | tonumber))) | abs) <= 0.25 and
    ((.format.duration | tonumber) >= 9.5 and (.format.duration | tonumber) <= 10.6) and
    ((.format.size | tonumber) > 0)
' "$PROBE_FILE" >/dev/null

ffmpeg -v error -i "$VIDEO_FILE" -map 0:v:0 -map 0:a:0 -f null -
ffmpeg -hide_banner -i "$VIDEO_FILE" -map 0:a:0 -af volumedetect -f null - \
    > /dev/null 2> "$AUDIO_STATS_FILE"
if grep -q 'mean_volume: -inf' "$AUDIO_STATS_FILE"; then
    echo "Generated audio is silent" >&2
    exit 1
fi

sha256sum "$VIDEO_FILE" > "$SHA_FILE"
export DYNAMO_REVISION IMAGE_REF MODEL MODEL_REVISION FASTH3_LORA_PATH FASTH3_VARIANT
export FASTH3_REVISION FASTH3_SHA256 ATTENTION_BACKEND FASTVIDEO_VSA_TOPK FASTVIDEO_VSA_SM100A
python3 - "$METADATA_FILE" "$HARDWARE_FILE" "$REQUEST_FILE" "$TIMING_FILE" <<'PY'
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone


def version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


with open(sys.argv[2], encoding="utf-8") as source:
    hardware = json.load(source)
with open(sys.argv[3], encoding="utf-8") as source:
    request = json.load(source)
with open(sys.argv[4], encoding="utf-8") as source:
    request_time_seconds = float(source.read().strip())

metadata = {
    "qualified_at": datetime.now(timezone.utc).isoformat(),
    "dynamo_revision": os.environ["DYNAMO_REVISION"],
    "container_image": os.environ["IMAGE_REF"],
    "model": os.environ["MODEL"],
    "model_revision": os.environ["MODEL_REVISION"],
    "request_time_seconds": request_time_seconds,
    "package_versions": {
        "ai-dynamo": version("ai-dynamo"),
        "vllm": version("vllm"),
        "vllm-omni": version("vllm-omni"),
        "fastvideo-kernel": version("fastvideo-kernel"),
        "av": version("av"),
    },
    "hardware": hardware,
    "request": request,
}
if os.environ["FASTH3_LORA_PATH"]:
    metadata["fasth3"] = {
        "adapter_path": os.environ["FASTH3_LORA_PATH"],
        "adapter_revision": os.environ["FASTH3_REVISION"],
        "adapter_sha256": os.environ["FASTH3_SHA256"],
        "variant": os.environ["FASTH3_VARIANT"],
        "attention_backend": os.environ["ATTENTION_BACKEND"],
    }
    if os.environ["FASTH3_VARIANT"].startswith("vsa-"):
        metadata["fasth3"]["fastvideo_vsa_topk"] = int(
            os.environ["FASTVIDEO_VSA_TOPK"]
        )
        metadata["fasth3"]["fastvideo_vsa_sm100a"] = (
            os.environ["FASTVIDEO_VSA_SM100A"] == "1"
        )
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(metadata, output, indent=2)
    output.write("\n")
PY
echo "MiniMax-H3 T2VA qualification passed: $VIDEO_FILE"
echo "Qualification metadata: $METADATA_FILE"
