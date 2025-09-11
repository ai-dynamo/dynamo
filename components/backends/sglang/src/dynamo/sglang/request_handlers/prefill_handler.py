# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import random
import socket

import msgspec
import sglang as sgl
from sglang.srt.utils import get_ip

from dynamo._core import Component
from dynamo.sglang.args import Config
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


class PrefillWorkerHandler(BaseWorkerHandler):
    def __init__(self, component: Component, engine: sgl.Engine, config: Config):
        self.engine = engine
        self.bootstrap_host, self.bootstrap_port = self._get_bootstrap_info()
        super().__init__(component, engine, config, None, None, None)
        logging.info(
            f"Prefill worker handler initialized - bootstrap host: {self.bootstrap_host}, bootstrap port: {self.bootstrap_port}"
        )

    def _generate_bootstrap_room(self):
        return random.randint(0, 2**63 - 1)

    def cleanup(self):
        self.engine.shutdown()
        logging.info("Prefill engine shutdown")
        super().cleanup()

    def _get_bootstrap_info(self):
        """Bootstrap info from tokenizer manager"""
        inner_tm = self.engine.tokenizer_manager
        bootstrap_port = inner_tm.server_args.disaggregation_bootstrap_port

        if inner_tm.server_args.dist_init_addr:
            # IPv6-ready host extraction and resolution:
            # 1) Extract raw host from "host:port" or "[IPv6]:port"/"[IPv6]".
            # 2) Resolve via AF_UNSPEC to accept A/AAAA and literals.
            # 3) Bracket-wrap IPv6 for safe "{host}:{port}" URL formatting.
            addr = inner_tm.server_args.dist_init_addr.strip()
            if addr.startswith("["):
                end = addr.find("]")
                host_core = addr[1:end] if end != -1 else addr.strip("[]")
            else:
                # Only treat single ':' with numeric suffix as host:port; otherwise it's an IPv6/FQDN host.
                if addr.count(":") == 1:
                    host_candidate, maybe_port = addr.rsplit(":", 1)
                    host_core = host_candidate if maybe_port.isdigit() else addr
                else:
                    host_core = addr
            try:
                infos = socket.getaddrinfo(
                    host_core,
                    None,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                )
                resolved = infos[0][4][0]  # let OS policy pick v4/v6
                bootstrap_host = resolved
            except socket.gaierror:
                # Fallback: keep literal/FQDN as-is (still wrap IPv6 below)
                bootstrap_host = host_core
        else:
            bootstrap_host = get_ip()

        # Wrap IPv6 literal with brackets so f"{host}:{port}" stays valid.
        if ":" in bootstrap_host and not bootstrap_host.startswith("["):
            bootstrap_host = f"[{bootstrap_host}]"

        return bootstrap_host, bootstrap_port

    async def generate(self, request: str):
        req = msgspec.json.decode(request, type=dict)
        bootstrap_room = self._generate_bootstrap_room()

        bootstrap_info = {
            "bootstrap_host": self.bootstrap_host,
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": bootstrap_room,
        }

        yield bootstrap_info

        results = await self.engine.async_generate(
            input_ids=req["request"]["token_ids"],
            sampling_params=req["sampling_params"],
            stream=True,
            bootstrap_host=self.bootstrap_host,
            bootstrap_port=self.bootstrap_port,
            bootstrap_room=bootstrap_room,
        )

        asyncio.create_task(self._consume_results(results))

    async def _consume_results(self, results):
        async for _ in results:
            pass
