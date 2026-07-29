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

import argparse


def _image_pair(value: str) -> tuple[str, str]:
    try:
        file_name, url = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid format for image argument: {value!r}. "
            "Expected format is 'file_name:url'."
        ) from exc
    return file_name, url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start a local media server.")
    parser.add_argument(
        "--image",
        action="append",
        type=_image_pair,
        help='Specify images in the format "file_name:url". Can be used multiple times.',
        required=True,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8233,
        help="Specify the port number for the server. Default is 8233.",
    )
    parser.add_argument(
        "--processing-time-mean-ms",
        type=float,
        default=0.0,
        help="Mean per-request delay (ms) before the server responds, "
        "simulating origin-CDN latency. Default is 0.",
    )
    parser.add_argument(
        "--processing-time-variance-ms",
        type=float,
        default=0.0,
        help="Variance (sigma^2, ms^2) of the per-request delay. "
        "Each request samples N(mean, sqrt(variance)), clamped at 0. Default is 0.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def parse_images(image_args: list[tuple[str, str]]) -> dict[str, str]:
    return dict(image_args)
