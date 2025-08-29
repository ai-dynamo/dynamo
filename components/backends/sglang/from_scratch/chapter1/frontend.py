# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time
from typing import List, Optional

import uvicorn
import uvloop
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "Qwen/Qwen2.5-0.5B-Instruct"
    messages: List[Message]
    max_tokens: Optional[int] = 25
    temperature: Optional[float] = 0.6


class FrontendRequestHandler:
    def __init__(self, runtime: DistributedRuntime):
        self.runtime = runtime
        self.processor_client = None
        self.app = None

    async def initialize(self):
        """Initialize the frontend service"""
        # Get processor client
        self.processor_client = (
            await self.runtime.namespace("inference")
            .component("processor")
            .endpoint("process")
            .client()
        )

        # Create FastAPI app
        self.app = FastAPI(title="Dynamo")

        # Add routes
        self.setup_routes()

        logging.info("Frontend initialized successfully")

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            try:
                # Send request to processor
                # generate under the hood is using round robin
                processor_response = await self.processor_client.generate(
                    request.model_dump()
                )

                # Collect response - expecting single response with content
                full_content = ""
                async for chunk in processor_response:
                    data = chunk.data()

                    if "error" in data:
                        raise HTTPException(status_code=500, detail=data["error"])

                    if "content" in data:
                        full_content = data["content"]
                        return {
                            "id": "chatcmpl-123",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "message": Message(
                                        role="assistant", content=full_content
                                    )
                                }
                            ],
                        }

                # If no content received, return empty response
                raise HTTPException(
                    status_code=500, detail="No content received from processor"
                )

            except Exception as e:
                logging.error(f"Error in chat completions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}

    async def run_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the FastAPI server"""
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)

        logging.info(f"Starting FastAPI server on {host}:{port}")
        await server.serve()


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    """Main worker function"""
    # Initialize frontend
    frontend = FrontendRequestHandler(runtime)
    await frontend.initialize()

    # Start FastAPI server
    await frontend.run_server()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
