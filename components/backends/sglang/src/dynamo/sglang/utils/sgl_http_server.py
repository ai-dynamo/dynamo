# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging

import uvicorn
import uvloop
from fastapi import FastAPI

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

    async def _discover_endpoints(self):
        """Discover endpoints that match the pattern"""
        etcd_client = self.runtime.etcd_client()
        if etcd_client is None:
            raise RuntimeError("Runtime has no etcd client; cannot discover endpoints")

        prefix = "instances/"
        kvs = await etcd_client.kv_get_prefix(prefix)

        # Collect (namespace, component) combos that expose flush_cache
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
            if ep_name == self.args.endpoint:
                discovered.add((ns, comp))
                logging.debug(f"Discovered endpoint: {ns}.{comp}")

        logging.debug(
            f"Endpoint discovery complete. Found {len(discovered)} matching endpoints"
        )
        return discovered

    def setup_routes(self):
        @self.app.post("/flush_cache")
        async def flush_cache():
            """Flush the radix cache."""
            try:
                discovered = await self._discover_endpoints()

                if not discovered:
                    return {"message": "No matching endpoints found", "success": False}

                logging.debug(
                    f"Found components: {', '.join([f'{ns}.{comp}' for ns, comp in discovered])}"
                )

                for ns, comp in discovered:
                    ep = (
                        self.runtime.namespace(ns)
                        .component(comp)
                        .endpoint(self.args.endpoint)
                    )
                    client = await ep.client()
                    await client.wait_for_instances()
                    ids = client.instance_ids()

                    logging.debug(f"-- {ns}.{comp} : {len(ids)} instances --")

                    for inst_id in ids:
                        try:
                            stream = await client.direct("{}", inst_id)
                            async for payload in stream:
                                logging.debug(f"[{ns}.{comp}][{inst_id}] -> {payload}")
                        except Exception as e:
                            logging.error(f"[{ns}.{comp}][{inst_id}] flush error: {e}")

                return {"message": "Cache flush initiated", "success": True}
            except Exception as e:
                logging.error(f"Cache flush error: {e}")
                return {"message": f"Cache flush failed: {str(e)}", "success": False}

    async def start_server(self):
        """Start the HTTP server"""
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
        )
        server = uvicorn.Server(config)

        # Single nice log with available endpoints
        logging.info(
            f"🚀 SGL engine HTTP server running on http://0.0.0.0:{self.port} - Endpoints: POST /flush_cache"
        )

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
    p.add_argument(
        "--endpoint", default=FLUSH_CACHE_ENDPOINT, help="Specify endpoint name"
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
