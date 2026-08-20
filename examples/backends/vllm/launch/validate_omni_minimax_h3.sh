#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Qualify all MiniMax-H3 tasks against one already-running aggregated worker.
set -euo pipefail

API_URL="${DYN_H3_API_URL:-http://127.0.0.1:8000/v1/videos}"
MODEL="${DYN_H3_MODEL:-MiniMaxAI/MiniMax-H3}"
QUAL_DIR="${DYN_H3_QUAL_DIR:-/dynamo-tmp/logs/08-19/minimax-h3}"
ASSET_DIR="$QUAL_DIR/assets"
OUTPUT_DIR="$QUAL_DIR/outputs"

mkdir -p "$ASSET_DIR" "$OUTPUT_DIR"

ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i color=c=navy:s=448x256:d=1 \
    -frames:v 1 "$ASSET_DIR/first.png"
ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i color=c=gold:s=448x256:d=1 \
    -frames:v 1 "$ASSET_DIR/last.png"
ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i color=c=beige:s=448x256:d=1 \
    -frames:v 1 "$ASSET_DIR/reference.png"
ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i sine=frequency=440:sample_rate=32000:duration=4 \
    -ac 2 "$ASSET_DIR/reference.wav"
ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i testsrc=size=448x256:rate=24:duration=2 \
    -f lavfi -i sine=frequency=330:sample_rate=32000:duration=2 \
    -c:v libx264rgb -pix_fmt rgb24 -c:a aac -ac 2 -shortest \
    "$ASSET_DIR/reference-1.mp4"
ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i testsrc2=size=448x256:rate=24:duration=2 \
    -f lavfi -i sine=frequency=550:sample_rate=32000:duration=2 \
    -c:v libx264rgb -pix_fmt rgb24 -c:a aac -ac 2 -shortest \
    "$ASSET_DIR/reference-2.mp4"

validate_output() {
    local case_name="$1"
    local response_file="$2"
    local media_url
    local media_path
    local probe_file="$OUTPUT_DIR/${case_name}.ffprobe.json"

    media_url="$(jq -er '.data[0].url' "$response_file")"
    media_path="${media_url#file://}"
    if [[ ! -s "$media_path" ]]; then
        echo "Missing output for $case_name: $media_path" >&2
        exit 1
    fi
    cp "$media_path" "$OUTPUT_DIR/${case_name}.mp4"
    ffprobe -v error \
        -show_entries stream=codec_type,codec_name,sample_rate,channels,r_frame_rate \
        -of json "$media_path" > "$probe_file"

    jq -e '
        any(.streams[]; .codec_type == "video" and .codec_name == "h264" and .r_frame_rate == "24/1") and
        any(.streams[]; .codec_type == "audio" and .codec_name == "aac" and .sample_rate == "32000" and .channels == 2)
    ' "$probe_file" >/dev/null
    if ffmpeg -hide_banner -i "$media_path" -map 0:a:0 -af volumedetect -f null - \
        2>&1 | grep -q 'mean_volume: -inf'; then
        echo "Generated audio is silent for $case_name" >&2
        exit 1
    fi
}

run_case() {
    local case_name="$1"
    local request_json="$2"
    local response_file="$OUTPUT_DIR/${case_name}.response.json"

    echo "Running $case_name..."
    curl -fsS "$API_URL" \
        -H 'Content-Type: application/json' \
        --data-binary "$request_json" > "$response_file"
    jq -e '.status == "completed" and (.data | length) >= 1' "$response_file" >/dev/null
    validate_output "$case_name" "$response_file"
}

COMMON_NVEXT='{"duration":4.0,"fps":24,"num_inference_steps":50,"flow_shift":12.0,"audio_flow_shift":3.0,"seed":42}'

run_case t2va "$(jq -cn \
    --arg model "$MODEL" \
    --argjson common "$COMMON_NVEXT" \
    '{model:$model,prompt:"A quiet night scene with synchronized forest ambience.",size:"448x256",response_format:"url",nvext:($common + {task:"t2va",aspect_ratio:"16:9"})}')"

run_case fl2va "$(jq -cn \
    --arg model "$MODEL" \
    --arg first "$ASSET_DIR/first.png" \
    --arg last "$ASSET_DIR/last.png" \
    --argjson common "$COMMON_NVEXT" \
    '{model:$model,prompt:"Move naturally from the blue frame to the gold frame.",response_format:"url",input_references:[{type:"image",source:$first},{type:"image",source:$last}],nvext:($common + {task:"fl2va",frame_indices:[0,-1]})}')"

run_case ref2va_image_audio "$(jq -cn \
    --arg model "$MODEL" \
    --arg image "$ASSET_DIR/reference.png" \
    --arg audio "$ASSET_DIR/reference.wav" \
    --argjson common "$COMMON_NVEXT" \
    '{model:$model,prompt:"A subject speaks in sync with the reference tone.",size:"448x256",response_format:"url",input_references:[{type:"image",source:$image},{type:"audio",source:$audio}],nvext:($common + {task:"ref2va",aspect_ratio:"16:9"})}')"

run_case ref2va_two_videos "$(jq -cn \
    --arg model "$MODEL" \
    --arg video1 "$ASSET_DIR/reference-1.mp4" \
    --arg video2 "$ASSET_DIR/reference-2.mp4" \
    --argjson common "$COMMON_NVEXT" \
    '{model:$model,prompt:"Combine the subject motion from Video 1 with the scene from Video 2.",size:"448x256",response_format:"url",input_references:[{type:"video",source:$video1},{type:"video",source:$video2}],nvext:($common + {task:"ref2va",aspect_ratio:"16:9",start_time_seconds:[0.0,0.0]})}')"

echo "MiniMax-H3 qualification passed; outputs are in $OUTPUT_DIR"
