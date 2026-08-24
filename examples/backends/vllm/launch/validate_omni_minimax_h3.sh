#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Qualify MiniMax-H3 T2VA against one already-running aggregated worker.
set -euo pipefail

API_URL="${DYN_H3_API_URL:-http://127.0.0.1:8000/v1/videos}"
MODEL="${DYN_H3_MODEL:-MiniMaxAI/MiniMax-H3}"
QUAL_DIR="${DYN_H3_QUAL_DIR:-/tmp/dynamo_minimax_h3_qualification}"
OUTPUT_DIR="$QUAL_DIR/outputs"
CASE_NAME="cat-playing-canon-in-d-grand-piano"
REQUEST_FILE="$OUTPUT_DIR/${CASE_NAME}.request.json"
RESPONSE_FILE="$OUTPUT_DIR/${CASE_NAME}.response.json"
VIDEO_FILE="$OUTPUT_DIR/${CASE_NAME}.mp4"
PROBE_FILE="$OUTPUT_DIR/${CASE_NAME}.ffprobe.json"
SOURCE_FILE="$OUTPUT_DIR/${CASE_NAME}.source-url.txt"
SHA_FILE="$OUTPUT_DIR/${CASE_NAME}.sha256"

mkdir -p "$OUTPUT_DIR"

jq -n \
    --arg model "$MODEL" \
    --arg prompt "A photorealistic orange tabby cat seated at a polished black grand piano, visibly pressing the keys with both front paws while performing Pachelbel's Canon in D. Elegant concert hall, cinematic lighting, realistic paw and key motion, synchronized clear solo grand-piano audio playing the recognizable Canon in D melody, no speech, no other instruments." \
    '{
        model: $model,
        prompt: $prompt,
        size: "448x256",
        response_format: "url",
        output_format: "mp4",
        nvext: {
            num_inference_steps: 50,
            seed: 42
        },
        task: "t2va",
        duration: 10.0,
        aspect_ratio: "16:9",
        flow_shift: 12.0,
        audio_flow_shift: 3.0
    }' > "$REQUEST_FILE"

echo "Generating a 10-second MiniMax-H3 T2VA sample..."
curl -fsS --max-time 7200 "$API_URL" \
    -H 'Content-Type: application/json' \
    --data-binary "@$REQUEST_FILE" > "$RESPONSE_FILE"
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
    -show_entries stream=codec_type,codec_name,width,height,sample_rate,channels,r_frame_rate \
    -show_entries format=duration,size \
    -of json "$VIDEO_FILE" > "$PROBE_FILE"

expected_size="$(jq -er '.size' "$REQUEST_FILE")"
expected_width="${expected_size%x*}"
expected_height="${expected_size#*x}"

jq -e \
    --argjson expected_width "$expected_width" \
    --argjson expected_height "$expected_height" '
    any(.streams[];
        .codec_type == "video" and
        .codec_name == "h264" and
        .r_frame_rate == "24/1" and
        .width == $expected_width and
        .height == $expected_height
    ) and
    any(.streams[]; .codec_type == "audio" and .codec_name == "aac" and .sample_rate == "32000" and .channels == 2) and
    ((.format.duration | tonumber) >= 9.5 and (.format.duration | tonumber) <= 10.6) and
    ((.format.size | tonumber) > 0)
' "$PROBE_FILE" >/dev/null

if ffmpeg -hide_banner -i "$VIDEO_FILE" -map 0:a:0 -af volumedetect -f null - \
    2>&1 | grep -q 'mean_volume: -inf'; then
    echo "Generated audio is silent" >&2
    exit 1
fi

sha256sum "$VIDEO_FILE" > "$SHA_FILE"
echo "MiniMax-H3 T2VA qualification passed: $VIDEO_FILE"
