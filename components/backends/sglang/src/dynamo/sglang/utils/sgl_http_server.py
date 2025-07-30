# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging

import uvicorn
import uvloop
from fastapi import FastAPI
from fastapi.routing import APIRoute

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

FLUSH_CACHE_ENDPOINT = "flush_cache"

configure_dynamo_logging()


class SglangHttpServer:
    def __init__(self, port: int, runtime: DistributedRuntime, args):
        self.port = port
        self.app = FastAPI()
        self.runtime = runtime
        self.args = args
        self.setup_routes()

    async def _discover_endpoints(self, endpoint_name):
        """Discover endpoints that match the pattern"""
        etcd_client = self.runtime.etcd_client()
        if etcd_client is None:
            raise RuntimeError("Runtime has no etcd client; cannot discover endpoints")

        prefix = "instances/"
        kvs = await etcd_client.kv_get_prefix(prefix)

        # Collect (namespace, component) combos that expose the target endpoint
        discovered = set()
        for kv in kvs:
            key = kv["key"] if isinstance(kv, dict) else kv.key
            if isinstance(key, bytes):
                key = key.decode()
            if not key.startswith(prefix):
                continue

            segments = key.split("/")
            # Format: instances/<ns>/<comp>/<endpoint:lease>
            if len(segments) < 4:
                continue
            ns, comp, ep_with_lease = segments[1], segments[2], segments[3]

            if self.args.ns and ns != self.args.ns:
                continue
            if self.args.comp and comp != self.args.comp:
                continue

            ep_name = ep_with_lease.split(":", 1)[0]
            if ep_name == endpoint_name:
                discovered.add((ns, comp))
                logging.debug(f"Discovered endpoint: {ns}.{comp}")

        logging.debug(
            f"Endpoint discovery complete. Found {len(discovered)} matching endpoints"
        )
        return discovered

    async def _dispatch_command(
        self, endpoint_name: str, payload: dict | str = "{}", success_message: str = ""
    ):
        """Dispatches a command to all instances of a discovered endpoint."""
        discovered = await self._discover_endpoints(endpoint_name=endpoint_name)
        if not discovered:
            return {"message": "No matching endpoints found", "success": False}

        logging.debug(
            f"Found components: {', '.join([f'{ns}.{comp}' for ns, comp in discovered])}"
        )

        for ns, comp in discovered:
            ep = self.runtime.namespace(ns).component(comp).endpoint(endpoint_name)
            client = await ep.client()
            await client.wait_for_instances()
            ids = client.instance_ids()

            logging.debug(f"-- {ns}.{comp} : {len(ids)} instances --")

            for inst_id in ids:
                try:
                    stream = await client.direct(payload, inst_id)
                    async for stream_payload in stream:
                        logging.debug(f"[{ns}.{comp}][{inst_id}] -> {stream_payload}")
                except Exception as e:
                    logging.error(
                        f"[{ns}.{comp}][{inst_id}] {endpoint_name} error: {e}"
                    )

        return {"message": success_message, "success": True}

    def setup_routes(self):
        @self.app.post("/flush_cache")
        async def flush_cache():
            """Flush the radix cache."""
            endpoint_name = self.args.endpoint
            try:
                return await self._dispatch_command(
                    endpoint_name,
                    success_message="Cache flush initiated",
                )
            except Exception as e:
                logging.error(f"Cache flush error: {e}")
                return {"message": f"Cache flush failed: {str(e)}", "success": False}

        @self.app.post("/start_expert_distribution_record")
        async def start_expert_distribution_record():
            """Start recording expert distribution."""
            endpoint_name = "start_expert_distribution_record"
            try:
                return await self._dispatch_command(
                    endpoint_name,
                    success_message="Expert distribution recording started",
                )
            except Exception as e:
                logging.error(f"Start expert distribution error: {e}")
                return {
                    "message": f"Start expert distribution failed: {str(e)}",
                    "success": False,
                }

        @self.app.post("/stop_expert_distribution_record")
        async def stop_expert_distribution_record():
            """Stop recording expert distribution."""
            endpoint_name = "stop_expert_distribution_record"
            try:
                return await self._dispatch_command(
                    endpoint_name,
                    success_message="Expert distribution recording stopped",
                )
            except Exception as e:
                logging.error(f"Stop expert distribution error: {e}")
                return {
                    "message": f"Stop expert distribution failed: {str(e)}",
                    "success": False,
                }

        @self.app.post("/dump_expert_distribution_record")
        async def dump_expert_distribution_record(request: dict):
            """Dump expert distribution recording to specified directory."""
            endpoint_name = "dump_expert_distribution_record"
            try:
                return await self._dispatch_command(
                    endpoint_name,
                    success_message="Expert distribution recording dumped to directory",
                )
            except Exception as e:
                logging.error(f"Dump expert distribution error: {e}")
                return {
                    "message": f"Dump expert distribution failed: {str(e)}",
                    "success": False,
                }

    async def start_server(self):
        """Start the HTTP server"""
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
        )
        server = uvicorn.Server(config)

        # Debug: print all registered routes
        for route in self.app.routes:
            if isinstance(route, APIRoute):
                logging.debug(f"Registered route: {route.methods} {route.path}")

        await server.serve()


def parse_args():
    p = argparse.ArgumentParser(description="SGLang HTTP server for cache management")
    p.add_argument("--port", type=int, default=9001, help="Port to listen on")
    p.add_argument(
        "--ns",
        "--namespace",
        default="dynamo",
        help="Specify Dynamo namespace (default: discover all)",
    )
    p.add_argument(
        "--comp",
        "--component",
        default=None,
        help="Specify component name (default: discover all)",
    )
    return p.parse_args()


@dynamo_worker(static=False)
async def main(runtime: DistributedRuntime):
    args = parse_args()

    http_server = SglangHttpServer(args.port, runtime, args)
    await http_server.start_server()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())
