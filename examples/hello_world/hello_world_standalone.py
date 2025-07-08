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

import logging

from pydantic import BaseModel

from dynamo.sdk import DynamoContext, async_on_start, endpoint, serve, service

logger = logging.getLogger(__name__)


class RequestType(BaseModel):
    text: str


@service(
    dynamo={
        "namespace": "dynamo",
    }
)
class Server:
    def __init__(self, greeting: str) -> None:
        logger.info("Starting server")
        self.greeting = greeting

    @endpoint()
    async def generate(self, req: RequestType):
        yield f"{self.greeting} {req.text}"


@service(
    dynamo={
        "namespace": "dynamo",
    }
)
class Client:
    def __init__(self, dynamo_context: DynamoContext, name: str) -> None:
        logger.info("Starting client")
        self.runtime = dynamo_context.runtime
        self.name = name

    @async_on_start
    async def async_init(self):
        self.server = (
            await self.runtime.namespace("dynamo")
            .component("Server")
            .endpoint("generate")
            .client()
        )
        await self.server.wait_for_instances()

        stream = await self.server.generate(
            RequestType(text=self.name).model_dump_json()
        )
        async for word in stream:
            print(word.data())


if __name__ == "__main__":
    """
    Example of running Dynamo components with python command
    $ python hello_world_standalone.py server --greeting "Hello, World!"
    $ python hello_world_standalone.py client --name "Bob"
    """
    import argparse
    import asyncio

    import uvloop

    parser = argparse.ArgumentParser(description="Run Hello World server or client")
    parser.add_argument(
        "component", choices=["server", "client"], help="Which component to run"
    )
    parser.add_argument(
        "--greeting", default="Hello, World!", help="Greeting message (for server)"
    )
    parser.add_argument("--name", default="User", help="Name to use (for client)")
    args = parser.parse_args()

    uvloop.install()
    if args.component == "server":
        asyncio.run(serve(Server, greeting=args.greeting))
    else:
        asyncio.run(serve(Client, name=args.name))
