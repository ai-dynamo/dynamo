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

# [NOTE] this is keep as a reference in case we need to run a local media server to eliminate image server influence.
# However, this implementation is not used as it is actually slower than directly using public image URLs in our benchmark experiments.
#
# Example usage:
# python -m benchmarks.multimodal.local_media_server.main \
#     --image test.jpg:https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg \
#     --processing-time-mean-ms 200 \
#     --processing-time-variance-ms 400 &
# IMG_SERVER_PID=$!
# trap "kill $IMG_SERVER_PID" EXIT
#
# # Wait for the server to start
# for i in {1..10}; do
#     HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" localhost:8233/test.jpg)
#     if [[ "$HTTP_CODE" -eq 200 ]]; then
#         echo "Server is responding with HTTP 200."
#         break
#     else
#         echo "Server did not respond with HTTP 200. Response code: $HTTP_CODE. Retrying in 1 second..."
#         sleep 1
#     fi
#     if [[ $i -eq 10 ]]; then
#         echo "Server did not respond with HTTP 200 after 10 attempts. Exiting."
#         exit 1
#     fi
# done

import requests

from .args import parse_args, parse_images
from .local_media_server import run_server


def download(images: dict[str, str]) -> dict[str, bytes]:
    downloaded: dict[str, bytes] = {}
    for name, url in images.items():
        response = requests.get(url)
        response.raise_for_status()
        downloaded[name] = response.content
    return downloaded


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    name_to_url = parse_images(args.image)
    name_to_bytes = download(name_to_url)
    run_server(
        args.port,
        name_to_bytes,
        args.processing_time_mean_ms,
        args.processing_time_variance_ms,
    )


if __name__ == "__main__":
    main()
