# type: ignore  # Ignore all mypy errors in this file
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import logging
import threading
import time
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

from dynamo.sdk import async_on_start, dynamo_context, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


def start_server(server):
    # Setup stuff here...
    server.serve_forever()


class HealthServer(HTTPServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ready = False

    def set_ready(self, ready: bool):
        self.ready = ready


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.server.ready:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Ready.")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Not Ready")
            return


def parse_args(service_name, prefix) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total-workers",
        type=int,
        default=1,
        help="Total number of workers to be registered",
    )
    parser.add_argument(
        "--worker-components",
        nargs="+",
        default=["VllmWorker", "PrefillWorker"],
        help="Components that we are tracking worker readiness",
    )
    parser.add_argument(
        "--component-endpoints",
        nargs="+",
        default=["generate", "mock"],
        help="Components that we are tracking worker readiness",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout (seconds) for waiting for workers to be ready",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7001,
        help="port for readiness check",
    )
    config = ServiceConfig.get_instance()
    config_args = config.as_args(service_name, prefix=prefix)
    args = parser.parse_args(config_args)
    if len(args.worker_components) != len(args.component_endpoints):
        parser.error(
            "--worker-components and --component-endpoints must have the same number "
            f"of items, but got {args.worker_components} and {args.component_endpoints}"
        )
    return args


# Use dynamo style to have access to clients
@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"cpu": "1", "memory": "1Gi"},
    workers=1,
)
class Watcher:
    def __init__(self):
        self.args = parse_args(self.__class__.__name__, "")

    @async_on_start
    async def async_init(self):
        self.runtime = dynamo_context["runtime"]
        self.workers_clients = []
        for component, endpoint in zip(
            self.args.worker_components, self.args.component_endpoints
        ):
            self.workers_clients.append(
                await self.runtime.namespace("dynamo")
                .component(component)
                .endpoint(endpoint)
                .client()
            )
            logger.info(f"Component {component}/{endpoint} is registered")
        logger.info(f"Total number of workers to be waited: {self.args.total_workers}")
        logger.info(f"Timeout for waiting for workers to be ready: {self.args.timeout}")
        self.server = HealthServer(("0.0.0.0", self.args.port), RequestHandler)
        print(f"Serving on 0.0.0.0:{self.args.port}, listening to readiness check...")
        self._server_thread = threading.Thread(target=start_server, args=(self.server,))
        self._server_thread.start()
        await check_required_workers(
            self.workers_clients, self.args.total_workers, self.args.timeout
        )
        self.server.set_ready(True)
        logger.info("All workers are ready.")


async def check_required_workers(
    workers_clients, required_workers: int, timeout: int, poll_interval=1
):
    """Wait until the minimum number of workers are ready."""
    start_time = time.time()
    num_workers = 0
    while num_workers < required_workers and time.time() - start_time < timeout:
        num_workers = sum(map(lambda wc: len(wc.instance_ids()), workers_clients))
        if num_workers < required_workers:
            logger.info(
                f"Waiting for more workers to be ready.\n"
                f" Current: {num_workers},"
                f" Required: {required_workers}"
            )
            await asyncio.sleep(poll_interval)
    if num_workers < required_workers:
        raise TimeoutError(
            f"Timed out waiting for {required_workers} workers to be ready."
        )
