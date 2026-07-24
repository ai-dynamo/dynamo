# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HelloEngine — a hello-world Dynamo unified backend.

No GPU, no model weights, no downloads. Every response streams the
same hardcoded sentence, token by token. Even the tokenizer is a mock:
a bundled 256-token byte-level tokenizer (one token per byte, see
tokenizer/) that the frontend uses to tokenize requests and the engine
uses once to encode the hardcoded sentence. Pass --tokenizer-repo to
swap in a real Hugging Face tokenizer instead.

It also publishes a KV event for each prompt's blocks, so the Dynamo
KV router learns which worker "holds" which prefix: send the same
prompt twice and the router pins the second request to the worker
that served the first.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
from collections.abc import AsyncGenerator
from typing import Optional

from dynamo._core import Context
from dynamo.common.backend import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    LlmRegistration,
    WorkerConfig,
)
from dynamo.common.backend.publisher import KvEventSource, PushSource

logger = logging.getLogger(__name__)

BLOCK_SIZE = 16

HARDCODED_REPLY = (
    "Hello! I am a hand-written Dynamo engine. I run no model - every "
    "token you are reading was hardcoded by a human. I exist to show "
    "how a custom engine plugs into Dynamo's frontend, router, and "
    "KV-aware routing."
)


def _block_hash(token_block: list[int]) -> int:
    """Hash a block's tokens so identical prompt blocks always get the
    same ID — that's what lets the router match repeat prompts."""
    digest = hashlib.blake2b(
        b",".join(str(t).encode() for t in token_block), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") >> 1  # keep it a positive i64


class HelloEngine(LLMEngine):
    def __init__(self, tokenizer_repo: str, delay: float = 0.05):
        self.tokenizer_repo = tokenizer_repo
        self.delay = delay
        self._reply_token_ids: Optional[list[int]] = None  # set in start()
        self._publisher = None  # set when the framework hands us one
        self._published_blocks: set[int] = set()  # don't re-publish a block

    # ------------------------------------------------------------------
    # from_args: CLI -> (engine, WorkerConfig)   [called first]
    # ------------------------------------------------------------------
    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple["HelloEngine", WorkerConfig]:
        parser = argparse.ArgumentParser(prog="hello-engine")
        parser.add_argument(
            "--tokenizer-repo",
            # Default: the tiny byte-level mock tokenizer bundled with this
            # package (256 tokens, one per byte). Nothing is downloaded.
            # Pass an HF repo name to use a real tokenizer instead.
            default=os.path.join(os.path.dirname(__file__), "tokenizer"),
            help="Local tokenizer dir or HF repo. Default: bundled mock tokenizer.",
        )
        parser.add_argument(
            "--served-model-name",
            default="hello-engine",
            help="Public model name clients put in their requests.",
        )
        parser.add_argument("--delay", type=float, default=0.05)
        # Runtime flags. Defaults defer to the DYN_* env vars the
        # Kubernetes operator injects, so the same binary works locally
        # and in a DynamoGraphDeployment without flag changes.
        parser.add_argument(
            "--namespace", default=os.environ.get("DYN_NAMESPACE", "dynamo")
        )
        parser.add_argument("--component", default="hello")
        parser.add_argument("--endpoint", default="generate")
        parser.add_argument(
            "--discovery-backend",
            default=os.environ.get("DYN_DISCOVERY_BACKEND", "etcd"),
        )
        parser.add_argument(
            "--request-plane", default=os.environ.get("DYN_REQUEST_PLANE", "tcp")
        )
        parser.add_argument(
            "--event-plane", default=os.environ.get("DYN_EVENT_PLANE") or None
        )
        args = parser.parse_args(argv)

        engine = cls(tokenizer_repo=args.tokenizer_repo, delay=args.delay)
        worker_config = WorkerConfig(
            namespace=args.namespace,
            component=args.component,
            endpoint=args.endpoint,
            model_name=args.tokenizer_repo,  # where the tokenizer lives
            served_model_name=args.served_model_name,  # what clients call us
            discovery_backend=args.discovery_backend,
            request_plane=args.request_plane,
            event_plane=args.event_plane,
        )
        return engine, worker_config

    # ------------------------------------------------------------------
    # start: load "the model"   [called once at boot]
    # ------------------------------------------------------------------
    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        from tokenizers import Tokenizer

        local_file = os.path.join(self.tokenizer_repo, "tokenizer.json")
        if os.path.exists(local_file):
            tokenizer = Tokenizer.from_file(local_file)
        else:
            tokenizer = await asyncio.to_thread(
                Tokenizer.from_pretrained, self.tokenizer_repo
            )
        self._reply_token_ids = tokenizer.encode(
            HARDCODED_REPLY, add_special_tokens=False
        ).ids
        logger.info("HelloEngine ready: reply is %d tokens", len(self._reply_token_ids))

        return EngineConfig(
            model=self.tokenizer_repo,
            served_model_name=None,  # Worker uses WorkerConfig.served_model_name
            llm=LlmRegistration(
                context_length=4096,
                kv_cache_block_size=BLOCK_SIZE,  # REQUIRED for KV events to flow
                total_kv_blocks=4096,
                max_num_seqs=64,
                max_num_batched_tokens=4096,
            ),
        )

    # ------------------------------------------------------------------
    # generate: stream tokens   [called once per request]
    # ------------------------------------------------------------------
    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if self._reply_token_ids is None:
            raise RuntimeError("generate() called before start()")
        prompt_tokens = list(request.get("token_ids", []))

        # Tell the router we now "cache" this prompt's blocks.
        self._publish_prompt_blocks(prompt_tokens)

        stop_conditions = request.get("stop_conditions") or {}
        max_new = stop_conditions.get("max_tokens")
        if max_new is None:  # unset -> full reply (note: 0 is a valid cap)
            max_new = len(self._reply_token_ids)
        reply = self._reply_token_ids[:max_new]

        def _usage(completion_tokens: int) -> dict[str, int]:
            return {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": completion_tokens,
                "total_tokens": len(prompt_tokens) + completion_tokens,
            }

        for i, token_id in enumerate(reply):
            await asyncio.sleep(self.delay)
            if context.is_stopped():  # re-check after the await
                yield {
                    "token_ids": [],
                    "index": 0,
                    "finish_reason": "cancelled",
                    "completion_usage": _usage(i),
                }
                return
            yield {"token_ids": [token_id], "index": 0}

        # Always emit a terminal chunk carrying finish_reason — including
        # when reply is empty (max_tokens=0), where the loop never ran.
        yield {
            "token_ids": [],
            "index": 0,
            "finish_reason": "stop"
            if len(reply) == len(self._reply_token_ids)
            else "length",
            "completion_usage": _usage(len(reply)),
        }

    # ------------------------------------------------------------------
    # cleanup   [called at shutdown; must be idempotent + null-safe]
    # ------------------------------------------------------------------
    async def cleanup(self) -> None:
        self._publisher = None
        self._reply_token_ids = None

    # ------------------------------------------------------------------
    # KV events: opt in, receive the publisher, publish per prompt
    # ------------------------------------------------------------------
    async def kv_event_sources(self) -> list[KvEventSource]:
        # [called once after start()] "Build me a publisher and hand it
        # to the callback below."
        return [PushSource(on_ready=self._on_publisher_ready, dp_rank=0)]

    def _on_publisher_ready(self, publisher) -> None:
        # The framework hands us a live KvEventPublisher (wired to ZMQ
        # or NATS — we never know which). We just keep it.
        self._publisher = publisher

    def _publish_prompt_blocks(self, prompt_tokens: list[int]) -> None:
        """One publish_stored() call per new full 16-token block."""
        if self._publisher is None:  # KV routing disabled by operator
            return
        token_ids: list[int] = []
        hashes: list[int] = []
        for b in range(len(prompt_tokens) // BLOCK_SIZE):
            block = prompt_tokens[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
            h = _block_hash(block)
            if h not in self._published_blocks:
                self._published_blocks.add(h)
                token_ids.extend(block)
                hashes.append(h)
        if hashes:
            self._publisher.publish_stored(
                token_ids=token_ids,
                num_block_tokens=[BLOCK_SIZE] * len(hashes),
                block_hashes=hashes,
            )
            logger.info("published %d KV block(s) for prompt", len(hashes))
