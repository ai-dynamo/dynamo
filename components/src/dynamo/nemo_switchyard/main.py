#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.routellm_router [args]`
#
# Start a RouteLLM proxy that sits in front of the Dynamo Frontend.
# It receives OpenAI-compatible requests, uses RouteLLM to classify
# query complexity, rewrites the `model` field to select a strong or
# weak model, and forwards to the Dynamo Frontend for dispatch.

import argparse
import json
import logging
import os
import time
from typing import Any, Optional

import httpx
import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from routellm.controller import Controller

from . import __version__

logger = logging.getLogger(__name__)


class RouteLLMProxy:
    """Core proxy that classifies requests and forwards to the Dynamo Frontend."""

    def __init__(
        self,
        backend_url: str,
        strong_model: str,
        weak_model: str,
        router_type: str,
        threshold: float,
        checkpoint_path: Optional[str],
        fallback_model: Optional[str],
    ):
        self.backend_url = backend_url.rstrip("/")
        self.strong_model = strong_model
        self.weak_model = weak_model
        self.router_type = router_type
        self.threshold = threshold
        self.checkpoint_path = checkpoint_path
        self.fallback_model = fallback_model or strong_model

        self.controller: Optional[Controller] = None
        self.client: Optional[httpx.AsyncClient] = None

        # Routing statistics
        self.stats = {
            "total": 0,
            "strong_routes": 0,
            "weak_routes": 0,
            "fallback_routes": 0,
            "errors": 0,
        }

    async def initialize(self):
        """Create the RouteLLM controller and HTTP client."""
        logger.info(
            "Initializing RouteLLM proxy: router=%s, threshold=%.2f, "
            "strong=%s, weak=%s",
            self.router_type,
            self.threshold,
            self.strong_model,
            self.weak_model,
        )

        router_config = {self.router_type: self.checkpoint_path}
        self.controller = Controller(
            routers=[self.router_type],
            strong_model=self.strong_model,
            weak_model=self.weak_model,
            config=router_config,
        )

        self.client = httpx.AsyncClient(
            base_url=self.backend_url,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )

        logger.info("RouteLLM proxy initialized successfully")

    async def shutdown(self):
        """Clean up HTTP client."""
        if self.client:
            await self.client.aclose()

    def extract_prompt_from_messages(
        self, messages: list[dict[str, Any]]
    ) -> Optional[str]:
        """Extract the last user message text from a chat messages array."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                # Handle multimodal content arrays
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part.get("text")
                return None
        return None

    def extract_prompt_from_completion(self, prompt: Any) -> Optional[str]:
        """Extract string from a completions prompt field."""
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list) and prompt:
            first = prompt[0]
            if isinstance(first, str):
                return first
        return None

    def route_request(self, prompt: Optional[str]) -> str:
        """Use RouteLLM to select a model based on the prompt."""
        self.stats["total"] += 1

        if not prompt:
            logger.warning("Empty or unparseable prompt, falling back to %s",
                           self.fallback_model)
            self.stats["fallback_routes"] += 1
            return self.fallback_model

        try:
            routed_model = self.controller.route(
                prompt, self.router_type, self.threshold
            )
            if routed_model == self.strong_model:
                self.stats["strong_routes"] += 1
            elif routed_model == self.weak_model:
                self.stats["weak_routes"] += 1
            else:
                self.stats["fallback_routes"] += 1

            logger.debug("Routed to %s (prompt: %.60s...)", routed_model, prompt)
            return routed_model

        except Exception:
            logger.exception("RouteLLM routing failed, falling back to %s",
                             self.fallback_model)
            self.stats["fallback_routes"] += 1
            self.stats["errors"] += 1
            return self.fallback_model

    async def proxy_request(
        self,
        request: Request,
        body: dict[str, Any],
        target_model: str,
        is_stream: bool,
    ):
        """Rewrite the model field and forward to the backend."""
        body["model"] = target_model
        raw_body = json.dumps(body).encode("utf-8")

        # Forward headers, replacing content-length
        headers = dict(request.headers)
        headers.pop("host", None)
        headers["content-length"] = str(len(raw_body))
        headers["x-routellm-routed-model"] = target_model

        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"

        if is_stream:
            return await self._proxy_streaming(path, raw_body, headers)
        else:
            return await self._proxy_non_streaming(path, raw_body, headers)

    async def _proxy_streaming(
        self, path: str, body: bytes, headers: dict
    ) -> StreamingResponse:
        """Stream response bytes from the backend without SSE parsing."""
        backend_req = self.client.build_request(
            "POST", path, content=body, headers=headers
        )
        backend_resp = await self.client.send(backend_req, stream=True)

        return StreamingResponse(
            content=backend_resp.aiter_bytes(),
            status_code=backend_resp.status_code,
            headers=dict(backend_resp.headers),
            media_type=backend_resp.headers.get("content-type"),
        )

    async def _proxy_non_streaming(
        self, path: str, body: bytes, headers: dict
    ) -> JSONResponse:
        """Forward request and return the complete response."""
        backend_req = self.client.build_request(
            "POST", path, content=body, headers=headers
        )
        backend_resp = await self.client.send(backend_req)

        return JSONResponse(
            content=backend_resp.json(),
            status_code=backend_resp.status_code,
            headers={
                k: v
                for k, v in backend_resp.headers.items()
                if k.lower()
                not in ("content-length", "content-encoding", "transfer-encoding")
            },
        )


def create_app(proxy: RouteLLMProxy, virtual_model_name: str) -> FastAPI:
    """Create the FastAPI application with all routes."""

    app = FastAPI(
        title="Dynamo RouteLLM Proxy",
        version=__version__,
    )

    @app.on_event("startup")
    async def startup():
        await proxy.initialize()

    @app.on_event("shutdown")
    async def shutdown():
        await proxy.shutdown()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        prompt = proxy.extract_prompt_from_messages(messages)
        target_model = proxy.route_request(prompt)
        is_stream = body.get("stream", False)
        return await proxy.proxy_request(request, body, target_model, is_stream)

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        prompt_field = body.get("prompt")
        prompt = proxy.extract_prompt_from_completion(prompt_field)
        target_model = proxy.route_request(prompt)
        is_stream = body.get("stream", False)
        return await proxy.proxy_request(request, body, target_model, is_stream)

    @app.get("/v1/models")
    async def list_models():
        now = int(time.time())
        models = [
            {
                "id": virtual_model_name,
                "object": "model",
                "created": now,
                "owned_by": "dynamo-routellm",
            },
            {
                "id": proxy.strong_model,
                "object": "model",
                "created": now,
                "owned_by": "dynamo-routellm",
            },
            {
                "id": proxy.weak_model,
                "object": "model",
                "created": now,
                "owned_by": "dynamo-routellm",
            },
        ]
        return {"object": "list", "data": models}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return {
            "routing_stats": proxy.stats,
            "config": {
                "router_type": proxy.router_type,
                "threshold": proxy.threshold,
                "strong_model": proxy.strong_model,
                "weak_model": proxy.weak_model,
                "fallback_model": proxy.fallback_model,
                "backend_url": proxy.backend_url,
            },
        }

    return app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamo RouteLLM Proxy: Routes requests to strong/weak models based on query complexity",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"Dynamo RouteLLM Proxy {__version__}"
    )
    parser.add_argument(
        "--http-host",
        type=str,
        default=os.environ.get("ROUTELLM_HOST", "0.0.0.0"),
        help="HTTP host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=int(os.environ.get("ROUTELLM_PORT", "8080")),
        help="HTTP port (default: 8080)",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default=os.environ.get("ROUTELLM_BACKEND_URL", "http://localhost:8000"),
        help="Dynamo Frontend URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--strong-model",
        type=str,
        required=True,
        help="Strong model name (must match model registered in Dynamo)",
    )
    parser.add_argument(
        "--weak-model",
        type=str,
        required=True,
        help="Weak model name (must match model registered in Dynamo)",
    )
    parser.add_argument(
        "--router-type",
        type=str,
        choices=["mf", "causal_llm", "bert", "sw_ranking"],
        default="mf",
        help="RouteLLM router algorithm (default: mf)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Routing threshold 0-1 (default: 0.5). Higher values route more to the weak model.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="HuggingFace model ID or local path for the router checkpoint",
    )
    parser.add_argument(
        "--fallback-model",
        type=str,
        default=None,
        help="Model to use on routing failure (default: strong model)",
    )
    parser.add_argument(
        "--virtual-model-name",
        type=str,
        default="auto",
        help="Virtual model name clients send as 'model' field (default: auto)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    proxy = RouteLLMProxy(
        backend_url=args.backend_url,
        strong_model=args.strong_model,
        weak_model=args.weak_model,
        router_type=args.router_type,
        threshold=args.threshold,
        checkpoint_path=args.checkpoint_path,
        fallback_model=args.fallback_model,
    )

    app = create_app(proxy, args.virtual_model_name)

    uvloop.install()
    uvicorn.run(
        app,
        host=args.http_host,
        port=args.http_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
