# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MM Router Handler — routes multimodal requests via KV-cache-aware worker selection."""

import base64
import logging
import os
import time
from typing import Any, AsyncGenerator

from dynamo.llm import KvRouter
from dynamo.runtime.logging import configure_dynamo_logging

from .image_loader import RoutingImageLoader
from .mm_processor import build_block_mm_infos, extract_image_urls, process_multimodal

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class MMRouterHandler:
    """Routes requests to the vLLM worker with the best KV cache overlap."""

    def __init__(
        self,
        kv_router: KvRouter,
        tokenizer: Any,
        processor: Any,
        model: str,
        block_size: int,
    ):
        self.kv_router = kv_router
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.block_size = block_size
        self._image_loader = RoutingImageLoader()

        mode = os.getenv("MM_ROUTER_IMAGE_TRANSPORT_MODE", "url").strip().lower()
        if mode not in ("data_uri", "url"):
            logger.warning("Invalid MM_ROUTER_IMAGE_TRANSPORT_MODE=%r, using 'url'", mode)
            mode = "url"
        self.transport_mode = mode
        logger.info("MM image transport mode: %s", mode)

    async def generate(self, request: dict) -> AsyncGenerator[dict, None]:
        """Main entry point: process request, compute routing, forward to best worker."""
        t_start = time.perf_counter()

        messages = request.get("extra_args", {}).get("messages", [])
        image_urls = extract_image_urls(messages)
        n_images = len(image_urls)
        mm_perf: dict[str, float] | None = None

        if image_urls:
            routing_tokens, block_mm_infos, mm_perf = await self._process_mm_request(
                request, messages, image_urls
            )
        else:
            routing_tokens = request.get("token_ids")
            if not routing_tokens:
                raise ValueError("Missing token_ids in preprocessed request")
            n_blocks = (len(routing_tokens) + self.block_size - 1) // self.block_size
            block_mm_infos = [None] * n_blocks

        # Forward to backend via KvRouter
        t_pre_route = time.perf_counter()
        stream = await self.kv_router.generate(
            token_ids=request.get("token_ids"),
            model=request["model"],
            stop_conditions=request.get("stop_conditions"),
            sampling_options=request.get("sampling_options"),
            output_options=request.get("output_options"),
            router_config_override=request.get("router_config_override"),
            extra_args=request.get("extra_args"),
            multi_modal_data=request.get("multi_modal_data"),
            mm_routing_info={
                "routing_token_ids": routing_tokens,
                "block_mm_infos": block_mm_infos,
            },
        )
        t_stream_open = time.perf_counter()

        first_chunk = True
        async for response in stream:
            if first_chunk:
                t_first = time.perf_counter()
                perf_parts = [
                    f"total={1000*(t_first - t_start):.2f}ms",
                    f"pre_route={1000*(t_pre_route - t_start):.2f}ms",
                    f"kv_generate={1000*(t_stream_open - t_pre_route):.2f}ms",
                    f"first_chunk_wait={1000*(t_first - t_stream_open):.2f}ms",
                    f"images={n_images}",
                ]
                if mm_perf:
                    for k, v in mm_perf.items():
                        if isinstance(v, float):
                            perf_parts.append(f"{k}={v:.2f}ms")
                        else:
                            perf_parts.append(f"{k}={v}")
                logger.info("[perf][mm_handler] %s", " ".join(perf_parts))
                first_chunk = False
            yield response

    async def _process_mm_request(
        self,
        request: dict,
        messages: list[dict],
        image_urls: list[str],
    ) -> tuple[list[int], list[dict | None], dict[str, float]]:
        """Handle multimodal: load images, expand tokens, rewrite URLs, build routing info."""
        t0 = time.perf_counter()
        processed, raw_bytes_list = await process_multimodal(
            messages=messages,
            image_urls=image_urls,
            tokenizer=self.tokenizer,
            processor=self.processor,
            model=self.model,
            image_loader=self._image_loader,
        )
        t1 = time.perf_counter()

        # Strip image content from messages to reduce payload
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        part["image_url"]["url"] = "<stripped>"

        # Rewrite multi_modal_data URLs: Url → RawUrl (skips url::Url::parse in Rust)
        t2 = time.perf_counter()
        data_uri_count = url_count = 0
        mm_data = request.get("multi_modal_data", {})
        if isinstance(mm_data, dict):
            for i, item in enumerate(mm_data.get("image_url", [])):
                if not isinstance(item, dict) or "Url" not in item:
                    continue
                raw = raw_bytes_list[i] if i < len(raw_bytes_list) else None
                value = self._rewrite_url(item["Url"], raw)
                if value.startswith("data:"):
                    data_uri_count += 1
                else:
                    url_count += 1
                del item["Url"]
                item["RawUrl"] = value
        t3 = time.perf_counter()

        block_mm_infos = build_block_mm_infos(
            num_tokens=len(processed.tokens),
            block_size=self.block_size,
            mm_hashes=processed.mm_hashes,
            image_ranges=processed.image_ranges,
        )
        if block_mm_infos is None:
            raise ValueError("Failed to build block_mm_infos")

        perf = {
            "process_mm": 1000 * (t1 - t0),
            "rewrite": 1000 * (t3 - t2),
            "transport_mode": self.transport_mode,
            "data_uri_count": data_uri_count,
            "url_count": url_count,
        }
        return processed.tokens, block_mm_infos, perf

    def _rewrite_url(self, url: str, raw_bytes: bytes | None) -> str:
        """Rewrite URL for backend transport. Non-HTTP and 'url' mode pass through."""
        if self.transport_mode == "url" or not url.startswith(("http://", "https://")):
            return url
        if raw_bytes is None:
            logger.warning("Missing raw bytes for %s; keeping URL", url[:80])
            return url
        mime = _detect_mime(raw_bytes)
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        return f"data:{mime};base64,{b64}"


def _detect_mime(raw_bytes: bytes) -> str:
    """Detect MIME type from magic bytes."""
    if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"
