# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def dump_worker_kv_event(worker_id: int, event: dict, token_ids: list[int]):
    """Dump worker-side KV event with token_ids to file for debugging."""
    if not DEBUG_ENABLED:
        return
    import datetime

    with open(DEBUG_WORKER_KV_FILE, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Worker ID: {worker_id}\n")
        f.write(f"Event type: {event.get('type')}\n")
        f.write(f"num_tokens: {len(token_ids)}\n")
        f.write(f"token_ids (first 50): {token_ids[:50]}\n")
        f.write(f"token_ids (last 50): {token_ids[-50:]}\n")
        f.write(f"block_hashes: {event.get('block_hashes', [])}\n")
        f.write(f"parent_hash: {event.get('parent_hash')}\n")
        f.write(f"{'='*60}\n")


def _to_signed_i64(value: int | None) -> int | None:
    """Convert a Python int to signed 64-bit range by two's complement."""
    if value is None:
        return None
    if value >= 2**63:
        return value - 2**64
    if value < -(2**63):
        return ((value + 2**63) % 2**64) - 2**63
    return value


class MetricsPublisher:
    """Publishes worker metrics over ZMQ."""

    def __init__(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        logger.info(f"Metrics publisher initialized on port {port}")

    def publish(self, num_waiting_reqs: int, kv_cache_usage: float):
        metrics_data = {
            "num_waiting_reqs": num_waiting_reqs,
            "kv_cache_usage": kv_cache_usage,
        }
        self.socket.send_json(metrics_data)

    def close(self):
        self.socket.close()
        self.context.term()


class KvEventsPublisher:
    """Publishes KV cache events over ZMQ in ZmqKvEventListener format."""

    def __init__(self, port: int, block_size: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.block_size = block_size
        self.partial_block_hashes = set()
        self.sequence_number = 0
        logger.info(f"KV events publisher initialized on port {port}")

    def publish_event(self, event: dict):
        """Publish KV event in ZmqKvEventListener format.
        
        Format: 3 ZMQ frames
        1. Topic (empty bytes)
        2. Sequence number (8 bytes, big-endian u64)
        3. MessagePack payload: [timestamp, [events], data_parallel_rank]
        """
        # Convert to Router expected format
        event_type = event.get("type")
        converted_events = []
        
        if event_type == "stored":
            # Extract all block_hashes and token_ids
            # Convert negative hashes to unsigned (u64) and ensure token_ids are i32
            block_hashes = []
            for b in event.get("blocks", []):
                hash_val = b["block_hash"]
                # Convert signed to unsigned u64
                if hash_val < 0:
                    hash_val = (1 << 64) + hash_val
                block_hashes.append(hash_val)
            
            token_ids = []
            for b in event.get("blocks", []):
                # Ensure token_ids are integers (not dicts)
                token_ids.extend([int(tid) for tid in b["token_ids"]])
            
            # Handle parent_block_hash (could be None or negative)
            parent_hash = event.get("parent_hash")
            if parent_hash is not None and parent_hash < 0:
                parent_hash = (1 << 64) + parent_hash
            
            # Build event dict
            # Don't include optional fields if they are None
            stored_event = {
                "type": "BlockStored",
                "block_hashes": block_hashes,
                "token_ids": token_ids,
                "block_size": int(self.block_size),
            }
            
            # Only add these if they have actual values
            if parent_hash is not None:
                stored_event["parent_block_hash"] = int(parent_hash)

            # Inject block_mm_infos if present in the event
            # Rust expects: block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>
            # where BlockExtraInfo has mm_objects: Vec<MmObject>
            mm_extra_info = event.get("mm_extra_info")
            if mm_extra_info is not None:
                # Create per-block mm_infos array (one entry per block)
                num_blocks = len(block_hashes)
                block_mm_infos = [mm_extra_info for _ in range(num_blocks)]
                stored_event["block_mm_infos"] = block_mm_infos
            
            converted_events.append(stored_event)
        elif event_type == "removed":
            # Convert negative hashes to unsigned u64
            block_hashes = []
            for hash_val in event.get("block_hashes", []):
                if hash_val < 0:
                    hash_val = (1 << 64) + hash_val
                block_hashes.append(hash_val)
            
            converted_events.append({
                "type": "BlockRemoved",
                "block_hashes": block_hashes,
            })
        
        if not converted_events:
            return
        
        # Create batch: [timestamp, events_list, data_parallel_rank]
        # data_parallel_rank can be None (not 0)
        batch = [
            time.time(),
            converted_events,
            0  # data_parallel_rank as 0 (was None, but Rust expects u32)
        ]
        
        # Encode with MessagePack
        try:
            payload = msgpack.packb(batch, use_bin_type=True)
        except Exception as e:
            logger.error(f"Failed to pack batch with msgpack: {e}")
            return
        
        # Frame 1: Topic (empty)
        topic = b""
        
        # Frame 2: Sequence number (8 bytes, big-endian)
        seq_bytes = self.sequence_number.to_bytes(8, byteorder='big')
        self.sequence_number += 1
        
        # Frame 3: Payload
        self.socket.send_multipart([topic, seq_bytes, payload])

    def close(self):
        self.socket.close()
        self.context.term()


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
        self.kv_events_port = kv_events_port
        self.metrics_port = metrics_port

        self.llm: Optional[LLM] = None
        self.metrics_publisher: Optional[MetricsPublisher] = None
        self.kv_events_publisher: Optional[KvEventsPublisher] = None
        self.background_tasks = []
        self.max_window_size = None
        self.processing_initial_created_events = True
        self.kv_events_started = False
        self.first_request_processed = False

        self._initialize()

    def _initialize(self):
        """Initialize the TensorRT-LLM engine and publishers."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.worker_id)

        logger.info(
            f"Initializing worker {self.worker_id} with model {self.model} on GPU {self.worker_id}"
        )

        logger.info(f"Worker {self.worker_id}: Creating KV cache config...")
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=True,
            event_buffer_max_size=DEFAULT_KV_EVENT_BUFFER_MAX_SIZE,
        )

        logger.info(f"Worker {self.worker_id}: Initializing TensorRT-LLM engine (this may take 5-15 minutes on first run)...")
        self.llm = LLM(
            model=self.model,
            kv_cache_config=kv_cache_config,
        )
        logger.info(f"Worker {self.worker_id}: TensorRT-LLM engine initialized!")

        self.metrics_publisher = MetricsPublisher(self.metrics_port)
        self.kv_events_publisher = KvEventsPublisher(self.kv_events_port, self.block_size)

        logger.info(f"Worker {self.worker_id} initialized successfully")

    async def start_background_tasks(self):
        """Start background tasks for publishing metrics and KV events."""
        logger.info(f"Starting background tasks for worker {self.worker_id}")
        self.background_tasks.append(asyncio.create_task(self._publish_metrics_loop()))
        # KV events will be started lazily after first request
        logger.info(f"Worker {self.worker_id}: KV events will start after first request")

    async def _publish_metrics_loop(self):
        """Continuously publish metrics."""
        # Wait for engine to warm up
        await asyncio.sleep(1)
        
        try:
            stats = self.llm.get_stats_async(timeout=5)
            async for stat in stats:
                if not isinstance(stat, dict):
                    logger.debug(f"Worker {self.worker_id} skipping non-dict stat: {type(stat)}")
                    continue
                num_waiting_reqs = stat["numQueuedRequests"] + stat[
                    "inflightBatchingStats"
                ]["numPausedRequests"]
                alloc_total_blocks = stat["kvCacheStats"]["allocTotalBlocks"]
                max_num_blocks = stat["kvCacheStats"]["maxNumBlocks"]
                kv_cache_usage = (
                    alloc_total_blocks / max_num_blocks if max_num_blocks > 0 else 0.0
                )

                self.metrics_publisher.publish(num_waiting_reqs, kv_cache_usage)
                logger.debug(
                    f"Worker {self.worker_id} metrics: waiting={num_waiting_reqs}, usage={kv_cache_usage:.2%}"
                )
        except asyncio.CancelledError:
            logger.info(f"Worker {self.worker_id} metrics loop cancelled")
        except Exception as e:
            logger.error(f"Worker {self.worker_id} metrics loop error: {e}")

    def _should_drop_event(self, event: dict) -> bool:
        """Determine if we should drop this event (non-global attention layer)."""
        if "window_size" not in event or self.processing_initial_created_events:
            return False
        if event["window_size"] != self.max_window_size:
            return True
        return False

    def _update_max_window_size(self, event: dict):
        """Update max window size from created events."""
        if "window_size" in event:
            window_size = event["window_size"]
            if self.max_window_size is None or window_size > self.max_window_size:
                self.max_window_size = window_size
                logger.debug(
                    f"Worker {self.worker_id} max_window_size updated to {self.max_window_size}"
                )

    async def _publish_kv_events_loop(self):
        """Continuously publish KV cache events."""
        # Wait for engine to warm up
        await asyncio.sleep(2)
        
        try:
            # Use longer timeout or None for infinite wait
            events = self.llm.get_kv_cache_events_async(timeout=None)
            logger.info(f"Worker {self.worker_id}: KV events iterator obtained")

            event_count = 0
            async for event in events:
                event_count += 1
                logger.debug(f"Worker {self.worker_id}: KV event #{event_count}: {event}")
                
                try:
                    # Validate event structure
                    if not isinstance(event, dict):
                        logger.warning(f"Worker {self.worker_id}: Event is not dict, type={type(event)}")
                        continue
                    
                    if "event_id" not in event or "data" not in event:
                        logger.warning(f"Worker {self.worker_id}: Missing event_id or data")
                        continue
                    
                    if self._should_drop_event(event):
                        logger.info(f"Worker {self.worker_id}: Dropped event (non-global attention)")
                        continue

                    event_id = event["event_id"]
                    data = event["data"]
                    event_type = data.get("type")
                    
                    logger.debug(f"Worker {self.worker_id}: Processing event_type={event_type}")
                    
                    if not event_type:
                        logger.warning(f"Worker {self.worker_id}: No event_type in data")
                        continue
                    
                    # Process different event types
                    if event_type == "stored":
                        num_blocks = len(data.get('blocks', []))
                        logger.debug(f"Worker {self.worker_id}: STORED event with {num_blocks} blocks")
                        self.processing_initial_created_events = False
                        parent_hash = _to_signed_i64(data.get("parent_hash"))

                        blocks = []
                        for block in data["blocks"]:
                            token_num_in_block = len(block["tokens"])
                            block_hash = _to_signed_i64(block["block_hash"])

                            if token_num_in_block == self.block_size:
                                token_ids = [int(token["token_id"]) for token in block["tokens"]]
                                blocks.append({
                                    "block_hash": block_hash,
                                    "token_ids": token_ids,
                                    "num_tokens": token_num_in_block,
                                })
                            elif token_num_in_block < self.block_size:
                                self.kv_events_publisher.partial_block_hashes.add(block_hash)
                                break
                            else:
                                logger.error(f"Worker {self.worker_id}: Block too large: {token_num_in_block}")
                                break

                        if blocks:
                            # Collect all token_ids from blocks for debugging
                            all_token_ids = []
                            for b in blocks:
                                all_token_ids.extend(b["token_ids"])

                            # Extract mm_extra_info from TRTLLM's mm_keys in blocks
                            # TRTLLM format: 'mm_keys': [{'type': 'mm_key', 'hash': 'hex_string', 'start_offset': 0}]
                            mm_extra_info = None
                            for block in data["blocks"]:
                                mm_keys = block.get("mm_keys")
                                if mm_keys:
                                    for mm_key in mm_keys:
                                        if mm_key.get("type") == "mm_key":
                                            hash_hex = mm_key.get("hash", "")
                                            # Convert hex hash to int (take first 16 chars)
                                            mm_hash = int(hash_hex[:16], 16) if hash_hex else 0
                                            # Find image token range for offsets
                                            IMAGE_TOKEN_ID = 151937
                                            image_start = None
                                            image_end = None
                                            for i, tid in enumerate(all_token_ids):
                                                if tid == IMAGE_TOKEN_ID:
                                                    if image_start is None:
                                                        image_start = i
                                                    image_end = i + 1
                                            if image_start is not None:
                                                mm_extra_info = {
                                                    "mm_objects": [{
                                                        "mm_hash": mm_hash,
                                                        "offsets": [[image_start, image_end]]
                                                    }]
                                                }
                                                logger.debug(
                                                    f"Worker {self.worker_id}: mm_hash={mm_hash}, "
                                                    f"offsets=[{image_start}, {image_end}]"
                                                )
                                            break
                                    break  # Only need first mm_key

                            kv_event = {
                                "event_id": event_id,
                                "type": "stored",
                                "parent_hash": parent_hash,
                                "blocks": blocks,
                            }
                            if mm_extra_info is not None:
                                kv_event["mm_extra_info"] = mm_extra_info

                            dump_worker_kv_event(self.worker_id, kv_event, all_token_ids)

                            self.kv_events_publisher.publish_event(kv_event)
                            logger.info(
                                f"Worker {self.worker_id} published stored event: {len(blocks)} blocks, "
                                f"block_hashes={[b['block_hash'] for b in blocks]}"
                            )

                    elif event_type == "removed":
                        self.processing_initial_created_events = False
                        block_hashes = []
                        for block_hash in data["block_hashes"]:
                            block_hash = _to_signed_i64(block_hash)
                            if block_hash in self.kv_events_publisher.partial_block_hashes:
                                logger.debug(
                                    f"Skipping partial block hash {block_hash} from removal"
                                )
                                self.kv_events_publisher.partial_block_hashes.remove(block_hash)
                                continue
                            block_hashes.append(block_hash)

                        if block_hashes:
                            kv_event = {
                                "event_id": event_id,
                                "type": "removed",
                                "block_hashes": block_hashes,
                            }
                            self.kv_events_publisher.publish_event(kv_event)
                            logger.debug(
                                f"Worker {self.worker_id} published removed event: {len(block_hashes)} blocks"
                            )

                    elif event_type == "created" and self.processing_initial_created_events:
                        self._update_max_window_size(event)
                
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(f"Worker {self.worker_id}: Error processing event: {e}")
                    continue

        except asyncio.CancelledError:
            logger.info(f"Worker {self.worker_id} KV events loop cancelled")
        except RuntimeError as e:
            # KV events might not be properly initialized in some TensorRT-LLM versions
            if "IterationResult is not properly instantiated" in str(e):
                logger.warning(
                    f"Worker {self.worker_id}: KV cache events not available. "
                    f"Router will work with reduced functionality (no cache overlap tracking). "
                    f"This may happen if the TensorRT-LLM version doesn't fully support KV events."
                )
            else:
                logger.error(f"Worker {self.worker_id} KV events loop error: {e}")
        except Exception as e:
            logger.error(f"Worker {self.worker_id} KV events loop error: {e}")

        logger.warning(f"Worker {self.worker_id}: KV events loop exited unexpectedly")

    async def generate(
        self,
        prompt_input,  # Can be list[int] (tokens) or dict (MM input from default_multimodal_input_loader)
        sampling_params: dict,
    ) -> AsyncGenerator[dict, None]:
        """Generate tokens for a request."""
        from tensorrt_llm.llmapi.llm import SamplingParams

        # Start KV events monitoring after first request (lazy initialization)
        if not self.first_request_processed:
            self.first_request_processed = True
            if not self.kv_events_started:
                logger.info(f"Worker {self.worker_id}: Starting KV events monitoring after first request")
                self.background_tasks.append(asyncio.create_task(self._publish_kv_events_loop()))
                self.kv_events_started = True

        # TensorRT-LLM requires top_k >= 0, use 0 to disable top_k sampling
        top_k = sampling_params.get("top_k", 0)
        if top_k < 0:
            top_k = 0

        trtllm_sampling_params = SamplingParams(
            max_tokens=sampling_params.get("max_tokens", 100),
            temperature=sampling_params.get("temperature", 1.0),
            top_p=sampling_params.get("top_p", 1.0),
            top_k=top_k,
        )

        # Log input type
        if isinstance(prompt_input, dict):
            logger.debug(f"Worker {self.worker_id}: MM request with keys: {prompt_input.keys()}")
        else:
            logger.debug(f"Worker {self.worker_id}: Text request with {len(prompt_input)} tokens")

        # Pass prompt_input directly to generate_async
        # TRTLLM accepts both token list and dict (from default_multimodal_input_loader)
        outputs = self.llm.generate_async(
            prompt_input,
            sampling_params=trtllm_sampling_params,
            streaming=True,  # Must be True for KV events and incremental output
        )

        # Extract text from TensorRT-LLM RequestOutput objects
        async for request_output in outputs:
            # TensorRT-LLM returns RequestOutput objects with outputs list
            if hasattr(request_output, 'outputs') and request_output.outputs:
                completion_output = request_output.outputs[0]
                
                # Extract text_diff for streaming (incremental text)
                if hasattr(completion_output, 'text_diff'):
                    text = completion_output.text_diff
                elif hasattr(completion_output, 'text'):
                    text = completion_output.text
                else:
                    text = ""
                
                # Create a response dict similar to vLLM format
                response = {
                    'text': text,
                    'text_diff': getattr(completion_output, 'text_diff', text),
                    'token_ids': getattr(completion_output, 'token_ids', []),
                    'finish_reason': getattr(completion_output, 'finish_reason', None),
                }
                yield response
            else:
                # Fallback: yield as-is if structure is unexpected
                yield request_output

    def shutdown(self):
        """Shutdown the worker and cleanup resources."""
        logger.info(f"Shutting down worker {self.worker_id}")

        for task in self.background_tasks:
            task.cancel()

        if self.llm:
            self.llm.shutdown()

        if self.metrics_publisher:
            self.metrics_publisher.close()

        if self.kv_events_publisher:
            self.kv_events_publisher.close()


class TrtllmWorkers:
    """Manages multiple TensorRT-LLM workers."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        block_size: int = 32,
        base_kv_events_port: int = 5557,
        base_metrics_port: int = 5657,
        num_workers: int = 1,
    ):
        self.model = model
        self.block_size = block_size
        self.num_workers = num_workers
        self.workers = []

        logger.info(f"Initializing {num_workers} TensorRT-LLM workers for model {model}")
        logger.info("NOTE: First-time initialization compiles the model and takes 5-15 minutes")

        for worker_id in range(num_workers):
            logger.info(f"Creating worker {worker_id}/{num_workers}...")
            worker = TrtllmWorker(
                worker_id=worker_id,
                model=model,
                block_size=block_size,
                kv_events_port=base_kv_events_port + worker_id,
                metrics_port=base_metrics_port + worker_id,
            )
            self.workers.append(worker)
            logger.info(f"Worker {worker_id} created successfully")

        logger.info("All workers initialized successfully!")

    async def start_all(self):
        """Start background tasks for all workers."""
        for worker in self.workers:
            await worker.start_background_tasks()

    async def direct(
        self,
        prompt_input,  # Can be list[int] (tokens) or dict (MM input)
        worker_id: int,
        sampling_params: dict,
    ) -> AsyncGenerator[dict, None]:
        """Send request directly to a specific worker."""
        worker = self.workers[worker_id]
        async for output in worker.generate(prompt_input, sampling_params):
            yield output

    def shutdown_all(self):
        """Shutdown all workers."""
        logger.info("Shutting down all workers")
        for worker in self.workers:
            worker.shutdown()

