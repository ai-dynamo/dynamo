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

import sys

import requests

from .args import parse_args, parse_images
from .local_media_server import run_server

DOWNLOAD_TIMEOUT_S = 60


def download(images: dict[str, str]) -> dict[str, bytes]:
    downloaded: dict[str, bytes] = {}
    for name, url in images.items():
        try:
            with requests.get(url, timeout=DOWNLOAD_TIMEOUT_S, stream=True) as response:
                response.raise_for_status()
                downloaded[name] = response.content
        except requests.RequestException as exc:
            print(f"Failed to download {name!r} from {url}: {exc}", file=sys.stderr)
            sys.exit(1)
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
