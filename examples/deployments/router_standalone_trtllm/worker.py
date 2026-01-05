# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# Fix protobuf version conflict with etcd3
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

import msgpack
import zmq
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig

logger = logging.getLogger(__name__)

DEFAULT_KV_EVENT_BUFFER_MAX_SIZE = 1024

# Debug flag: set DYNAMO_DEBUG=1 to enable debug file dumps
DEBUG_ENABLED = os.environ.get("DYNAMO_DEBUG", "0") == "1"
DEBUG_WORKER_KV_FILE = "/tmp/debug_worker_kv.txt"

# Qwen2-VL specific token ID for image placeholders
IMAGE_TOKEN_ID = 151937


def dump_worker_kv_event(worker_id: int, event: dict, token_ids: list[int]):
    """Dump worker-side KV event to file for debugging."""
    if not DEBUG_ENABLED:
        return
    import datetime

    with open(DEBUG_WORKER_KV_FILE, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Worker ID: {worker_id}\n")
        f.write(f"Event: {event}\n")
        f.write(f"Tokens ({len(token_ids)}): {token_ids[:50]}...\n")
        f.write(f"{'='*60}\n")


def to_unsigned_u64(value: int | None) -> int | None:
    """Ensure value is in unsigned 64-bit range for Rust/msgpack."""
    if value is None:
        return None
    # Handle negative values (two's complement)
    return (1 << 64) + value if value < 0 else value


# -----------------------------------------------------------------------------
# ZMQ Publishers
# -----------------------------------------------------------------------------


class MetricsPublisher:
    """Publishes worker metrics over ZMQ."""

    def __init__(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")

    def publish(self, num_waiting_reqs: int, kv_cache_usage: float):
        self.socket.send_json(
            {
                "num_waiting_reqs": num_waiting_reqs,
                "kv_cache_usage": kv_cache_usage,
            }
        )

    def close(self):
        self.socket.close()
        self.context.term()


class KvEventsPublisher:
    """Publishes KV cache events over ZMQ."""

    def __init__(self, port: int, block_size: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.block_size = block_size
        self.partial_block_hashes: set[int] = set()
        self.sequence_number = 0

    def publish_stored(
        self,
        block_hashes: list[int],
        token_ids: list[int],
        parent_hash: int | None,
        block_mm_infos: list[dict | None] | None,
    ):
        """Publish a BlockStored event.

        Args:
            block_hashes: List of block hashes being stored.
            token_ids: All token IDs across the blocks.
            parent_hash: Hash of the parent block (if any).
            block_mm_infos: Per-block multimodal info list. Each element corresponds
                to a block and is either the mm_info dict (for blocks containing
                image tokens) or None (for text-only blocks).
        """
        event = {
            "type": "BlockStored",
            "block_hashes": [to_unsigned_u64(h) for h in block_hashes],
            "token_ids": token_ids,
            "block_size": self.block_size,
        }

        if parent_hash is not None:
            event["parent_block_hash"] = to_unsigned_u64(parent_hash)

        if block_mm_infos is not None:
            event["block_mm_infos"] = block_mm_infos

        self._send([event])

    def publish_removed(self, block_hashes: list[int]):
        """Publish a BlockRemoved event."""
        # Filter out partial blocks
        filtered = []
        for h in block_hashes:
            if h in self.partial_block_hashes:
                self.partial_block_hashes.remove(h)
            else:
                filtered.append(to_unsigned_u64(h))

        if filtered:
            self._send([{"type": "BlockRemoved", "block_hashes": filtered}])

    def _send(self, events: list[dict]):
        """Send events via ZMQ multipart message."""
        batch = [time.time(), events, 0]
        try:
            payload = msgpack.packb(batch, use_bin_type=True)
        except Exception as e:
            logger.error(f"msgpack error: {e}")
            return

        seq_bytes = self.sequence_number.to_bytes(8, byteorder="big")
        self.sequence_number += 1
        self.socket.send_multipart([b"", seq_bytes, payload])

    def close(self):
        self.socket.close()
        self.context.term()


# -----------------------------------------------------------------------------
# KV Event Processing Helpers
# -----------------------------------------------------------------------------


def extract_mm_info(
    blocks_data: list[dict], all_token_ids: list[int]
) -> tuple[list[int] | None, list[list[int]] | None]:
    """Extract multimodal hash info from TRTLLM block data.

    Handles multiple images by extracting all mm_hashes and matching them
    to their corresponding image token ranges.

    Returns:
        Tuple of (list of mm_hashes, list of offsets) or (None, None).
        Each offset is [start, end) for one image's token range.
    """
    # Collect all mm_hashes from blocks
    mm_hashes: list[int] = []
    for block in blocks_data:
        mm_keys = block.get("mm_keys", [])
        for mm_key in mm_keys:
            if mm_key.get("type") != "mm_key":
                continue
            hash_hex = mm_key.get("hash", "")
            if hash_hex:
                mm_hash = int(hash_hex[:16], 16)
                if mm_hash not in mm_hashes:  # Avoid duplicates
                    mm_hashes.append(mm_hash)

    if not mm_hashes:
        return None, None

    # Find all image token ranges
    image_offsets_list = find_all_image_token_ranges(all_token_ids)
    if not image_offsets_list:
        return None, None

    # Match mm_hashes to image_offsets by order
    # (assumes mm_hashes appear in same order as images in token sequence)
    return mm_hashes, image_offsets_list


def find_all_image_token_ranges(token_ids: list[int]) -> list[list[int]] | None:
    """Find all [start, end) ranges of contiguous image tokens.

    Returns:
        List of [start, end) ranges, one per contiguous image token sequence.
        Returns None if no image tokens found.
    """
    ranges: list[list[int]] = []
    current_start: int | None = None

    for i, tid in enumerate(token_ids):
        if tid == IMAGE_TOKEN_ID:
            if current_start is None:
                current_start = i
        elif current_start is not None:
            # End of contiguous sequence
            ranges.append([current_start, i])
            current_start = None

    # Handle sequence ending with image tokens
    if current_start is not None:
        ranges.append([current_start, len(token_ids)])

    return ranges if ranges else None


def build_per_block_mm_infos(
    num_blocks: int,
    block_size: int,
    mm_hashes: list[int] | None,
    image_offsets_list: list[list[int]] | None,
) -> list[dict | None] | None:
    """Build per-block mm_infos list for multiple images.

    Each block that overlaps with an image's token range gets the corresponding
    mm_info with that image's mm_hash.

    Args:
        num_blocks: Number of blocks in the stored event.
        block_size: Number of tokens per block.
        mm_hashes: List of mm_hash values, one per image.
        image_offsets_list: List of [start, end) token ranges, one per image.

    Returns:
        List of mm_info (one per block), with None for blocks without image tokens.
        Returns None if no mm_info is provided.
    """
    if mm_hashes is None or image_offsets_list is None:
        return None

    if not mm_hashes or not image_offsets_list:
        return None

    # Initialize result with None for all blocks
    result: list[dict | None] = [None] * num_blocks

    # Process each image
    for mm_hash, offsets in zip(mm_hashes, image_offsets_list):
        img_start, img_end = offsets

        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = block_start + block_size

            # Check if this block overlaps with this image's token range
            if block_end > img_start and block_start < img_end:
                if result[block_idx] is None:
                    result[block_idx] = {"mm_objects": []}
                # Add this image's mm_object to the block
                result[block_idx]["mm_objects"].append(
                    {"mm_hash": mm_hash, "offsets": [offsets]}
                )

    return result


def parse_stored_blocks(
    blocks_data: list[dict], block_size: int, partial_hashes: set[int]
) -> tuple[list[dict], list[int]]:
    """Parse stored blocks from TRTLLM event data.

    Returns:
        Tuple of (blocks list, all token_ids)
    """
    blocks = []
    all_token_ids = []

    for block in blocks_data:
        tokens = block["tokens"]
        num_tokens = len(tokens)
        block_hash = block["block_hash"]

        if num_tokens == block_size:
            token_ids = [int(t["token_id"]) for t in tokens]
            blocks.append(
                {
                    "block_hash": block_hash,
                    "token_ids": token_ids,
                    "num_tokens": num_tokens,
                }
            )
            all_token_ids.extend(token_ids)
        elif num_tokens < block_size:
            # Partial block - track but don't publish
            partial_hashes.add(block_hash)
            break
        else:
            logger.error(f"Block too large: {num_tokens} > {block_size}")
            break

    return blocks, all_token_ids


# -----------------------------------------------------------------------------
# TRT-LLM Worker
# -----------------------------------------------------------------------------


class TrtllmWorker:
    """Manages a single TensorRT-LLM worker with event/metrics publishing."""

    def __init__(
        self,
        worker_id: int,
        model: str,
        block_size: int,
        kv_events_port: int,
        metrics_port: int,
    ):
        self.worker_id = worker_id
        self.model = model
        self.block_size = block_size

        self.llm: Optional[LLM] = None
        self.metrics_publisher: Optional[MetricsPublisher] = None
        self.kv_events_publisher: Optional[KvEventsPublisher] = None

        self.background_tasks: list[asyncio.Task] = []
        self.max_window_size: int | None = None
        self.processing_initial_events = True
        self.kv_events_started = False

        self._initialize(kv_events_port, metrics_port)

    def _initialize(self, kv_events_port: int, metrics_port: int):
        """Initialize TensorRT-LLM engine and publishers."""
        logger.info(f"Worker {self.worker_id}: Initializing")

        self.llm = LLM(
            model=self.model,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,
                event_buffer_max_size=DEFAULT_KV_EVENT_BUFFER_MAX_SIZE,
            ),
        )

        self.metrics_publisher = MetricsPublisher(metrics_port)
        self.kv_events_publisher = KvEventsPublisher(kv_events_port, self.block_size)

        logger.info(f"Worker {self.worker_id}: Initialized")

    # -------------------------------------------------------------------------
    # Background Tasks
    # -------------------------------------------------------------------------

    async def start_background_tasks(self):
        """Start metrics publishing task."""
        self.background_tasks.append(asyncio.create_task(self._metrics_loop()))

    def _start_kv_events_task(self):
        """Lazily start KV events task on first request."""
        if self.kv_events_started:
            return
        self.kv_events_started = True
        logger.info(f"Worker {self.worker_id}: Starting KV events monitoring")
        self.background_tasks.append(asyncio.create_task(self._kv_events_loop()))

    async def _metrics_loop(self):
        """Continuously publish worker metrics."""
        await asyncio.sleep(1)

        try:
            async for stat in self.llm.get_stats_async(timeout=5):
                if not isinstance(stat, dict):
                    continue

                num_waiting = (
                    stat["numQueuedRequests"]
                    + stat["inflightBatchingStats"]["numPausedRequests"]
                )
                kv_stats = stat["kvCacheStats"]
                usage = (
                    kv_stats["allocTotalBlocks"] / kv_stats["maxNumBlocks"]
                    if kv_stats["maxNumBlocks"] > 0
                    else 0.0
                )

                self.metrics_publisher.publish(num_waiting, usage)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Worker {self.worker_id} metrics error: {e}")

    async def _kv_events_loop(self):
        """Continuously process and publish KV cache events."""
        await asyncio.sleep(2)

        try:
            events = self.llm.get_kv_cache_events_async(timeout=None)
            logger.info(f"Worker {self.worker_id}: KV events iterator obtained")

            async for event in events:
                self._process_kv_event(event)

        except asyncio.CancelledError:
            pass
        except RuntimeError as e:
            if "IterationResult is not properly instantiated" in str(e):
                logger.warning(f"Worker {self.worker_id}: KV events not available")
            else:
                logger.error(f"Worker {self.worker_id} KV events error: {e}")
        except Exception as e:
            logger.error(f"Worker {self.worker_id} KV events error: {e}")

        logger.warning(f"Worker {self.worker_id}: KV events loop exited")

    def _process_kv_event(self, event: dict):
        """Process a single KV cache event."""
        if not isinstance(event, dict):
            return
        if "event_id" not in event or "data" not in event:
            return

        data = event["data"]
        event_type = data.get("type")

        if self._should_drop_event(event):
            return

        if event_type == "stored":
            self._handle_stored_event(data)
        elif event_type == "removed":
            self._handle_removed_event(data)
        elif event_type == "created" and self.processing_initial_events:
            self._update_window_size(event)

    def _should_drop_event(self, event: dict) -> bool:
        """Check if event should be dropped (non-global attention)."""
        if self.processing_initial_events:
            return False
        window_size = event.get("window_size")
        if window_size is None:
            return False
        return window_size != self.max_window_size

    def _update_window_size(self, event: dict):
        """Update max window size from created events."""
        window_size = event.get("window_size")
        if window_size and (
            self.max_window_size is None or window_size > self.max_window_size
        ):
            self.max_window_size = window_size

    def _handle_stored_event(self, data: dict):
        """Handle a stored block event."""
        self.processing_initial_events = False

        blocks, all_token_ids = parse_stored_blocks(
            data["blocks"],
            self.block_size,
            self.kv_events_publisher.partial_block_hashes,
        )

        if not blocks:
            return

        parent_hash = data.get("parent_hash")
        mm_hashes, image_offsets_list = extract_mm_info(data["blocks"], all_token_ids)

        block_hashes = [b["block_hash"] for b in blocks]

        # Build per-block mm_infos (only blocks with image tokens get mm_info)
        block_mm_infos = build_per_block_mm_infos(
            len(blocks), self.block_size, mm_hashes, image_offsets_list
        )

        # Debug dump
        dump_worker_kv_event(
            self.worker_id,
            {"type": "stored", "blocks": len(blocks), "mm_hashes": mm_hashes},
            all_token_ids,
        )

        self.kv_events_publisher.publish_stored(
            block_hashes, all_token_ids, parent_hash, block_mm_infos
        )

    def _handle_removed_event(self, data: dict):
        """Handle a removed block event."""
        self.processing_initial_events = False

        block_hashes = data.get("block_hashes", [])
        self.kv_events_publisher.publish_removed(block_hashes)

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    async def generate(
        self,
        prompt_input,  # list[int] (tokens) or dict (MM input)
        sampling_params: dict,
    ) -> AsyncGenerator[dict, None]:
        """Generate tokens for a request."""
        from tensorrt_llm.llmapi.llm import SamplingParams

        # Start KV events on first request
        self._start_kv_events_task()

        trtllm_params = SamplingParams(
            max_tokens=sampling_params.get("max_tokens", 100),
            temperature=sampling_params.get("temperature", 1.0),
            top_p=sampling_params.get("top_p", 1.0),
            top_k=max(0, sampling_params.get("top_k", 0)),
        )

        outputs = self.llm.generate_async(
            prompt_input, sampling_params=trtllm_params, streaming=False
        )

        async for output in outputs:
            yield self._format_output(output)

    def _format_output(self, request_output) -> dict:
        """Format TRTLLM output to standard response dict."""
        if not hasattr(request_output, "outputs") or not request_output.outputs:
            return {"text": "", "text_diff": "", "token_ids": [], "finish_reason": None}

        completion = request_output.outputs[0]
        text = getattr(completion, "text_diff", None) or getattr(completion, "text", "")

        return {
            "text": text,
            "text_diff": getattr(completion, "text_diff", text),
            "token_ids": getattr(completion, "token_ids", []),
            "finish_reason": getattr(completion, "finish_reason", None),
        }

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def shutdown(self):
        """Shutdown worker and cleanup resources."""
        logger.info(f"Worker {self.worker_id}: Shutting down")

        for task in self.background_tasks:
            task.cancel()

        if self.llm:
            self.llm.shutdown()
        if self.metrics_publisher:
            self.metrics_publisher.close()
        if self.kv_events_publisher:
            self.kv_events_publisher.close()


# -----------------------------------------------------------------------------
# Worker Manager
# -----------------------------------------------------------------------------


class TrtllmWorkers:
    """Manages multiple TensorRT-LLM workers.

    Warning: Creating multiple workers in the same process causes them to share
    the same GPU(s).
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2-VL-2B-Instruct",
        block_size: int = 32,
        base_kv_events_port: int = 5557,
        base_metrics_port: int = 5657,
        num_workers: int = 1,
    ):
        self.workers = []

        if num_workers > 1:
            logger.warning(
                f"Creating {num_workers} workers in the same process. "
                "All workers will share the same GPU(s). For multi-GPU isolation, "
                "start each worker in a separate process with CUDA_VISIBLE_DEVICES set."
            )

        logger.info(f"Initializing {num_workers} workers for {model}")

        for i in range(num_workers):
            self.workers.append(
                TrtllmWorker(
                    worker_id=i,
                    model=model,
                    block_size=block_size,
                    kv_events_port=base_kv_events_port + i,
                    metrics_port=base_metrics_port + i,
                )
            )

        logger.info(f"All {num_workers} workers initialized")

    async def start_all(self):
        """Start background tasks for all workers."""
        for worker in self.workers:
            await worker.start_background_tasks()

    async def direct(
        self, prompt_input, worker_id: int, sampling_params: dict
    ) -> AsyncGenerator[dict, None]:
        """Send request to a specific worker."""
        async for output in self.workers[worker_id].generate(
            prompt_input, sampling_params
        ):
            yield output

    def shutdown_all(self):
        """Shutdown all workers."""
        logger.info("Shutting down all workers")
        for worker in self.workers:
            worker.shutdown()
