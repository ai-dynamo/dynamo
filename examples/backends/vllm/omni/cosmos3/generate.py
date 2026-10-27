# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Request a Cosmos3 image or short video and verify the returned media."""

import argparse
import base64
import io
import json
from pathlib import Path

import av
import requests
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="nvidia/Cosmos3-Nano")
    parser.add_argument("--mode", choices=("image", "video"), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-reference", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.input_reference is not None and args.mode != "video":
        parser.error("--input-reference requires --mode video")

    image_mode = args.mode == "image"
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "size": "1024x1024" if image_mode else "1280x720",
        "response_format": "b64_json",
        "nvext": {
            "seed": args.seed,
            "num_inference_steps": 50 if image_mode else 35,
            "guidance_scale": 7.0 if image_mode else 6.0,
            "negative_prompt": "blurry, distorted",
        },
    }
    if image_mode:
        payload["n"] = 1
    else:
        payload["output_format"] = "mp4"
        payload["nvext"].update(num_frames=33, fps=24)
    if args.input_reference is not None:
        with Image.open(args.input_reference) as source:
            reference = source.convert("RGB")
        buffer = io.BytesIO()
        reference.save(buffer, format="PNG")
        payload["input_reference"] = (
            "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        )

    endpoint = "/v1/images/generations" if image_mode else "/v1/videos"
    response = requests.post(
        args.url.rstrip("/") + endpoint, json=payload, timeout=1800
    )
    response.raise_for_status()
    result = response.json()
    if result.get("status", "completed") != "completed" or result.get("error"):
        raise RuntimeError(f"Generation failed: {result.get('error', result)}")
    if len(result.get("data", [])) != 1:
        raise RuntimeError("Expected exactly one generated output")
    media = base64.b64decode(result["data"][0]["b64_json"], validate=True)

    if image_mode:
        with Image.open(io.BytesIO(media)) as image:
            image.load()
            if image.format != "PNG" or image.size != (1024, 1024):
                raise RuntimeError(f"Unexpected image: {image.format}, {image.size}")
            metadata = {"format": image.format, "size": image.size}
    else:
        with av.open(io.BytesIO(media)) as container:
            stream = container.streams.video[0]
            frame_rate = float(stream.average_rate)
            frame_count = 0
            for frame in container.decode(stream):
                if (frame.width, frame.height) != (1280, 720):
                    raise RuntimeError("Unexpected video dimensions")
                frame_count += 1
            if frame_count != 33 or frame_rate != 24:
                raise RuntimeError(
                    f"Expected 33 frames at 24 FPS; got {frame_count} at {frame_rate}"
                )
            metadata = {
                "codec": stream.codec_context.name,
                "size": [1280, 720],
                "frames": frame_count,
                "fps": frame_rate,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(media)
    print(json.dumps({"output": str(args.output), "bytes": len(media), **metadata}))


if __name__ == "__main__":
    main()
