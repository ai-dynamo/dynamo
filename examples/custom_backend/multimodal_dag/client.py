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

"""Send and validate one text-plus-image request to the multimodal DAG."""

from __future__ import annotations

import argparse
import base64
import io
import json
from typing import Any

import requests
from PIL import Image, ImageDraw

from examples.custom_backend.multimodal_dag.protocol import PUBLIC_MODEL_NAME


def _image_data_uri() -> str:
    image = Image.new("RGB", (96, 64), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 47, 63), fill="red")
    draw.rectangle((48, 0, 95, 63), fill="blue")
    output = io.BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _validate_response(response: dict[str, Any]) -> None:
    choices = response.get("choices")
    assert isinstance(choices, list) and len(choices) == 1, response
    message = choices[0].get("message")
    assert isinstance(message, dict), response
    content = message.get("content")
    assert isinstance(content, str) and content.strip(), response

    classifier = response.get("nvext", {}).get("classifier")
    assert isinstance(classifier, dict), response
    assert classifier.get("label") in {"class_0", "class_1"}, classifier
    score = classifier.get("score")
    assert (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and 0.0 <= score <= 1.0
    ), classifier
    shape = classifier.get("embedding_shape")
    assert (
        isinstance(shape, list)
        and len(shape) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in shape
        )
    ), classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    payload = {
        "model": PUBLIC_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe the two colored regions in this synthetic image."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_uri()},
                    },
                ],
            }
        ],
        "stream": False,
        "max_tokens": 48,
        "temperature": 0.0,
        "seed": 7,
    }
    response = requests.post(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=args.timeout,
    )
    response.raise_for_status()
    result = response.json()
    _validate_response(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
