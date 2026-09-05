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

"""Route Slime streaming rollout requests through the Dynamo frontend."""

import copy
import os
from argparse import Namespace
from urllib.parse import urlsplit


def _frontend_args(args: Namespace) -> Namespace:
    rollout_url = os.environ.get("DYNAMO_ROLLOUT_URL")
    if not rollout_url:
        raise ValueError("Set DYNAMO_ROLLOUT_URL to the Dynamo frontend base URL.")

    parsed = urlsplit(rollout_url)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise ValueError("DYNAMO_ROLLOUT_URL must use http:// and include a host.")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError(
            "DYNAMO_ROLLOUT_URL must not include a path, query, or fragment."
        )

    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("DYNAMO_ROLLOUT_URL contains an invalid port.") from error
    if port is None:
        raise ValueError("DYNAMO_ROLLOUT_URL must include a port.")

    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"

    frontend_args = copy.copy(args)
    frontend_args.sglang_router_ip = host
    frontend_args.sglang_router_port = port
    return frontend_args


async def generate_streaming(args: Namespace, sample, sampling_params):
    from slime.rollout.sglang_streaming_rollout import (
        generate_streaming as slime_generate_streaming,
    )

    return await slime_generate_streaming(_frontend_args(args), sample, sampling_params)


generate_streaming.abort_mode = "request"
