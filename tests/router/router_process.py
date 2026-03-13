# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from tests.utils.managed_process import ManagedProcess


class KVRouterProcess(ManagedProcess):
    """Manages the KV router process using dynamo.frontend"""

    def __init__(
        self,
        request,
        block_size: int,
        frontend_port: int,
        namespace: str,
        store_backend: str = "etcd",
        enforce_disagg: bool = False,
        blocks_threshold: float | None = None,
        tokens_threshold: float | None = None,
        tokens_threshold_frac: float | None = None,
        request_plane: str = "nats",
        durable_kv_events: bool = False,
    ):
        command = [
            "python3",
            "-m",
            "dynamo.frontend",
            "--kv-cache-block-size",
            str(block_size),
            "--router-mode",
            "kv",
            "--http-port",
            str(frontend_port),
            "--discovery-backend",
            store_backend,
            "--namespace",
            namespace,
        ]

        if enforce_disagg:
            command.append("--enforce-disagg")

        if blocks_threshold is not None:
            command.extend(["--active-decode-blocks-threshold", str(blocks_threshold)])

        if tokens_threshold is not None:
            command.extend(["--active-prefill-tokens-threshold", str(tokens_threshold)])

        if tokens_threshold_frac is not None:
            command.extend(
                ["--active-prefill-tokens-threshold-frac", str(tokens_threshold_frac)]
            )

        if durable_kv_events:
            command.append("--router-durable-kv-events")

        env = os.environ.copy()
        env["DYN_REQUEST_PLANE"] = request_plane

        super().__init__(
            command=command,
            env=env,
            timeout=60,
            display_output=True,
            health_check_ports=[frontend_port],
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._check_ready)
            ],
            log_dir=request.node.name,
            terminate_all_matching_process_names=False,
        )
        self.port = frontend_port

    def _check_ready(self, response):
        """Check if KV router is ready"""
        return response.status_code == 200

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
