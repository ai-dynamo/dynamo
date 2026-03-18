# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from tests.utils.managed_process import ManagedProcess


class FrontendRouterProcess(ManagedProcess):
    """Manages a dynamo.frontend process with configurable --router-mode.

    Supports all router modes (round-robin, random, kv, direct) and all
    KV-specific options (block size, thresholds, durable events, disagg).
    block_size is only sent to the CLI when router_mode is "kv".
    """

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
        router_mode: str = "kv",
    ):
        command = [
            "python3",
            "-m",
            "dynamo.frontend",
            "--router-mode",
            router_mode,
            "--http-port",
            str(frontend_port),
            "--discovery-backend",
            store_backend,
            "--namespace",
            namespace,
        ]

        if router_mode == "kv":
            command.extend(["--kv-cache-block-size", str(block_size)])

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
        self.router_mode = router_mode

    def _check_ready(self, response):
        """Check if KV, random, round-robin, or direct router is ready"""
        return response.status_code == 200

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)


# Backward-compatible alias so existing callers that import KVRouterProcess
# continue to work without changes.
KVRouterProcess = FrontendRouterProcess
