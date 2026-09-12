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

"""Connection configuration for Speech NIM gRPC services."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from dynamo.common.utils.namespace import get_worker_namespace

# Default Speech NIM gRPC target for a local deployment. Shared by the dataclass
# default and the CLI default so the two never drift.
DEFAULT_NIM_SERVER = "localhost:50051"
DEFAULT_NIM_STARTUP_TIMEOUT_S = 600.0


def resolve_dynamo_endpoint(endpoint: str | None, component: str) -> str:
    """Resolve an explicit endpoint or use the operator-provided namespace."""
    return endpoint or f"{get_worker_namespace()}.{component}.generate"


@dataclass
class NimConnectionConfig:
    """Connection settings for a Speech NIM gRPC endpoint.

    Attributes:
        server: ``host:port`` of the Speech NIM gRPC server. Defaults to a local
            deployment; override for a remote host or the NVCF endpoint
            (``grpc.nvcf.nvidia.com:443``).
        use_ssl: Whether to use a TLS channel. Required for NVCF.
        api_key: NVCF API key. When set, sent as an ``authorization: Bearer``
            gRPC metadata entry on every call.
        function_id: NVCF function id. When set, sent as a ``function-id``
            gRPC metadata entry on every call.
        ssl_root_cert: Optional path to a TLS root certificate.
    """

    server: str = DEFAULT_NIM_SERVER
    use_ssl: bool = False
    api_key: str | None = None
    function_id: str | None = None
    ssl_root_cert: str | None = None


def add_nim_connection_args(parser: argparse.ArgumentParser) -> None:
    """Add a "Speech NIM connection" argument group to ``parser``."""
    group = parser.add_argument_group("Speech NIM connection")
    group.add_argument(
        "--nim-server",
        default=DEFAULT_NIM_SERVER,
        help=f"host:port of the Speech NIM gRPC server (default: {DEFAULT_NIM_SERVER}).",
    )
    group.add_argument(
        "--nim-startup-timeout-s",
        type=float,
        default=DEFAULT_NIM_STARTUP_TIMEOUT_S,
        help="Seconds to wait for the Speech NIM gRPC server before registering the model.",
    )
    group.add_argument(
        "--nim-use-ssl",
        action="store_true",
        help="Use a TLS channel (required for NVCF).",
    )
    group.add_argument(
        "--nim-api-key",
        default=None,
        help="NVCF API key, sent as an authorization Bearer token.",
    )
    group.add_argument(
        "--nim-function-id",
        default=None,
        help="NVCF function id, sent as function-id metadata.",
    )
    group.add_argument(
        "--nim-ssl-root-cert",
        default=None,
        help="Path to a TLS root certificate.",
    )


def nim_connection_config_from_namespace(
    args: argparse.Namespace,
) -> NimConnectionConfig:
    """Build a :class:`NimConnectionConfig` from args parsed by :func:`add_nim_connection_args`."""
    return NimConnectionConfig(
        server=args.nim_server,
        use_ssl=args.nim_use_ssl,
        api_key=args.nim_api_key,
        function_id=args.nim_function_id,
        ssl_root_cert=args.nim_ssl_root_cert,
    )
