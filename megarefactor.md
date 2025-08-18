# SGLang Backend Unified Entry Point Mega Refactor

## Overview

This document contains all the code and refactoring steps needed to unify the SGLang backend on a single `dynamo.sglang` entry point, bringing it in line with vLLM and TRT-LLM patterns.

## Current State Analysis

**Current Issues:**
- SGLang uses separate entry points: `dynamo.sglang.worker` (prefill) and `dynamo.sglang.decode_worker` (decode)
- Unlike vLLM and TRT-LLM which have unified entry points (`dynamo.vllm`, `dynamo.trtllm`)
- Inconsistent with the overall Dynamo architecture pattern

**Current File Structure:**
```
dynamo/sglang/
├── __init__.py (empty)
├── worker/
│   ├── __init__.py (empty)  
│   ├── __main__.py
│   └── main.py (378 lines - handles aggregated + prefill)
├── decode_worker/
│   ├── __init__.py (empty)
│   ├── __main__.py  
│   └── main.py (95 lines - handles decode only)
└── common/
    ├── base_handlers.py
    ├── protocol.py
    └── sgl_utils.py
```

## Target Architecture

**New File Structure:**
```
dynamo/sglang/
├── __init__.py          # Minimal package init
├── __main__.py          # CLI entry point  
├── main.py              # Core initialization logic
├── args.py              # Configuration management
├── handlers.py          # ALL handler logic (consolidated)
└── common/
    ├── protocol.py       # Communication protocols
    └── sgl_utils.py     # Utility functions
```

## Implementation Steps

### Step 1: Create New Core Files

#### File: `components/backends/sglang/src/dynamo/sglang/main.py`

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import signal
import socket
import sys
from typing import Optional

import sglang as sgl
import uvloop
import zmq
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_ip, get_zmq_socket

from dynamo.llm import (
    ModelType,
    register_llm,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
    WorkerMetricsPublisher,
    WorkerStats,
    KvStats,
    ForwardPassMetrics
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from .args import parse_sglang_args, SGLangConfig
from .handlers import RequestHandlerFactory
from .common.sgl_utils import graceful_shutdown, setup_native_endpoints

configure_dynamo_logging()
logger = logging.getLogger(__name__)

@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()
    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    logger.info("Signal handlers set up for graceful shutdown")
    
    config = parse_sglang_args(sys.argv[1:])
    await init(runtime, config)

async def init(runtime: DistributedRuntime, config: SGLangConfig):
    """Initialize SGLang worker based on disaggregation mode"""
    logger.info(f"Initializing SGLang worker in {config.mode} mode")
    
    engine = sgl.Engine(server_args=config.server_args)
    
    # Component naming based on mode
    component_name = _get_component_name(config.mode)
    component = runtime.namespace("dynamo").component(component_name)
    await component.create_service()
    
    # Setup endpoints
    generate_endpoint = component.endpoint("generate")
    
    # Setup mode-specific dependencies
    decode_client = None
    
    # Prefill mode setup - only decode client connection
    if config.mode == "prefill":
        decode_client = (
            await runtime.namespace("dynamo")
            .component("decode")
            .endpoint("generate")
            .client()
        )
    
    # Setup KV publisher
    zmq_config = ZmqKvEventPublisherConfig(
        worker_id=generate_endpoint.lease_id(),
        kv_block_size=config.server_args.page_size,
    )
    kv_publisher = ZmqKvEventPublisher(component=component, config=zmq_config)
    
    # Register model
    if _should_register_model(config):
        await register_llm(
            ModelType.Backend,
            generate_endpoint,
            config.server_args.model_path,
            config.server_args.served_model_name,
            kv_cache_block_size=config.server_args.page_size,
            migration_limit=config.migration_limit,
        )
    
    # Create handler with shared dependencies (metrics and bootstrap setup moved back to handlers)
    handler = RequestHandlerFactory.create_handler(
        mode=config.mode,
        engine=engine,
        server_args=config.server_args,
        component=component,
        endpoint=generate_endpoint,
        decode_client=decode_client,
        config=config
    )
    
    # Setup metrics endpoint for handlers that have metrics
    if hasattr(handler, 'setup_metrics_endpoint'):
        await handler.setup_metrics_endpoint(component)
    
    # Setup native endpoints
    tasks = [generate_endpoint.serve_endpoint(handler.generate)]
    tasks.extend(setup_native_endpoints(config.server_args, component, handler))
    
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise


def _should_register_model(config: SGLangConfig) -> bool:
    """Determine if this worker should register the model"""
    # Only register for aggregated mode or prefill mode (not decode)
    return config.mode in ["null", "prefill"]

def _get_component_name(mode: str) -> str:
    """Map disaggregation mode to component name"""
    mode_mapping = {
        "null": "worker",      # Aggregated mode
        "prefill": "worker",   # Prefill worker  
        "decode": "decode"     # Decode worker
    }
    return mode_mapping.get(mode, "worker")

def main():
    uvloop.install()
    asyncio.run(worker())

if __name__ == "__main__":
    main()
```

#### File: `components/backends/sglang/src/dynamo/sglang/__main__.py`

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.sglang.main import main

if __name__ == "__main__":
    main()
```

#### File: `components/backends/sglang/src/dynamo/sglang/__init__.py`

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

#### File: `components/backends/sglang/src/dynamo/sglang/args.py`

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from dataclasses import dataclass
from typing import Optional
from sglang.srt.server_args import ServerArgs
from .common.sgl_utils import parse_sglang_args_inc

@dataclass
class SGLangConfig:
    """Unified configuration for SGLang workers"""
    server_args: ServerArgs
    migration_limit: int = 0
    mode: str = "null"  # null, prefill, decode
    
    @property
    def is_prefill_worker(self) -> bool:
        return self.mode == "prefill"
    
    @property
    def is_decode_worker(self) -> bool:
        return self.mode == "decode"
    
    @property
    def is_aggregated(self) -> bool:
        return self.mode == "null"
    
    @property
    def needs_decode_client(self) -> bool:
        """Check if this worker needs a decode client connection"""
        return self.is_prefill_worker and self.server_args.disaggregation_mode == "prefill"

def parse_sglang_args(args: list[str]) -> SGLangConfig:
    """Parse arguments and create unified config"""
    # Extract migration-limit before passing to SGLang parser
    migration_limit = 0
    filtered_args = []
    i = 0
    while i < len(args):
        if args[i] == "--migration-limit" and i + 1 < len(args):
            migration_limit = int(args[i + 1])
            i += 2  # Skip both --migration-limit and its value
        else:
            filtered_args.append(args[i])
            i += 1
    
    # Parse SGLang arguments
    server_args = parse_sglang_args_inc(filtered_args)
    
    # Determine mode from disaggregation_mode
    mode = getattr(server_args, 'disaggregation_mode', 'null')
    
    return SGLangConfig(
        server_args=server_args,
        migration_limit=migration_limit,
        mode=mode
    )
```

#### File: `components/backends/sglang/src/dynamo/sglang/handlers.py`

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import random
import socket
from typing import Any, Dict, Optional, Union

import msgspec
import sglang as sgl
import zmq
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_zmq_socket

from dynamo.llm import (
    WorkerMetricsPublisher,
    WorkerStats,
    KvStats,
    ForwardPassMetrics
)
from .args import SGLangConfig
from .common.protocol import DisaggPreprocessedRequest

logger = logging.getLogger(__name__)

class BaseSGLangHandler:
    """Base handler for all SGLang worker types"""
    
    def __init__(self, engine: sgl.Engine, server_args: ServerArgs, component, 
                 endpoint, config: SGLangConfig, decode_client: Optional[Any] = None):
        self.engine = engine
        self.server_args = server_args
        self.component = component
        self.endpoint = endpoint
        self.config = config
        self.decode_client = decode_client

    async def generate(self, request):
        """Default generate implementation for aggregated and prefill modes"""
        # Check if this is decode mode (different signature)
        if isinstance(request, str):
            # This should only happen for DecodeWorkerHandler
            raise NotImplementedError("DecodeWorkerHandler must override this method")
        
        # Standard aggregated/prefill generation logic
        is_batch = self._is_batch_request(request)
        batch_size = self._get_request_batch_size(request)
        sampling_params = self._build_sampling_params(request)

        # Check if disaggregated mode (prefill)
        if self.server_args.disaggregation_mode != "null":
            # Prefill mode - need decode_client
            if self.decode_client is None:
                raise ValueError("decode_client required for disaggregated mode")
            
            # Setup bootstrap info
            if is_batch:
                bootstrap_room = [self._generate_bootstrap_room() for _ in range(batch_size)]
                bootstrap_host = [self.bootstrap_host] * batch_size
                bootstrap_port = [self.bootstrap_port] * batch_size
            else:
                bootstrap_host = self.bootstrap_host
                bootstrap_port = self.bootstrap_port
                bootstrap_room = self._generate_bootstrap_room()

            # Create disaggregated request
            disagg_request = DisaggPreprocessedRequest(
                request=request,
                sampling_params=sampling_params,
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                bootstrap_room=bootstrap_room,
            )

            # Start prefill (result not used)
            prefill = await self.engine.async_generate(
                input_ids=request["token_ids"] if not is_batch else request["batch_token_ids"],
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                bootstrap_room=bootstrap_room,
            )
            prefill_task = asyncio.create_task(self._prefill_generator(prefill))

            # Get decode results
            decode = await self.decode_client.generate(disagg_request.model_dump_json())

            async for out in self._process_stream(decode, unpack=True, is_batch=is_batch):
                yield out

            await prefill_task
        else:
            # Aggregated mode - direct generation
            g = await self.engine.async_generate(
                input_ids=request["token_ids"] if not is_batch else request["batch_token_ids"],
                sampling_params=sampling_params,
                stream=True,
            )

            async for out in self._process_stream(g, unpack=False, is_batch=is_batch):
                yield out

    # Helper methods for generation
    def _build_sampling_params(self, request: dict) -> dict:
        """Build sampling parameters from request"""
        sampling_params = {}
        if request["sampling_options"]["temperature"]:
            sampling_params["temperature"] = request["sampling_options"]["temperature"]
        if request["sampling_options"]["top_p"]:
            sampling_params["top_p"] = request["sampling_options"]["top_p"]
        if request["sampling_options"]["top_k"]:
            sampling_params["top_k"] = request["sampling_options"]["top_k"]
        sampling_params["max_new_tokens"] = request["stop_conditions"]["max_tokens"]
        if request["stop_conditions"]["ignore_eos"]:
            sampling_params["ignore_eos"] = request["stop_conditions"]["ignore_eos"]
        return sampling_params

    def _get_request_batch_size(self, request: dict):
        """Get batch size from request, returns None for single requests"""
        if request["batch_token_ids"] is not None:
            return len(request["batch_token_ids"])
        return None

    def _is_batch_request(self, request: dict):
        """Check if request is in batch mode"""
        return request["batch_token_ids"] is not None

    def _generate_bootstrap_room(self):
        """Generate random bootstrap room ID"""
        return random.randint(0, 2**63 - 1)

    async def _process_stream(self, stream_source, unpack: bool, is_batch: bool):
        """Process streaming results from engine"""
        # Initialize based on batch mode
        num_output_tokens_so_far: Union[Dict[int, int], int]
        if is_batch:
            num_output_tokens_so_far = {}
        else:
            num_output_tokens_so_far = 0

        async for res in stream_source:
            data = res.data() if unpack else res
            finish_reason = data["meta_info"]["finish_reason"]

            if is_batch:
                # Handle batch response
                assert isinstance(num_output_tokens_so_far, dict)
                index = data.get("index", 0)
                if index not in num_output_tokens_so_far:
                    num_output_tokens_so_far[index] = 0

                if finish_reason:
                    out = {
                        "token_ids": [],
                        "finish_reason": finish_reason["type"],
                        "index": index,
                    }
                else:
                    next_total_toks = len(data["output_ids"])
                    new_tokens = data["output_ids"][num_output_tokens_so_far[index] :]
                    out = {
                        "token_ids": new_tokens,
                        "index": index,
                    }
                    num_output_tokens_so_far[index] = next_total_toks
            else:
                # Handle single response
                assert isinstance(num_output_tokens_so_far, int)
                if finish_reason:
                    out = {"token_ids": [], "finish_reason": finish_reason["type"]}
                else:
                    next_total_toks = len(data["output_ids"])
                    out = {"token_ids": data["output_ids"][num_output_tokens_so_far:]}
                    num_output_tokens_so_far = next_total_toks

            yield out

    async def _prefill_generator(self, prefill):
        """Consume prefill generator without using results"""
        async for _ in prefill:
            pass

    # SGLang native endpoints
    async def flush_cache(self, request: dict):
        """Flush KV cache for each worker"""
        _ = request
        await self.engine.tokenizer_manager.flush_cache()
        yield True

    async def start_expert_distribution_record(self, request: dict):
        """Start recording expert distribution."""
        _ = request
        await self.engine.tokenizer_manager.start_expert_distribution_record()
        yield True

    async def stop_expert_distribution_record(self, request: dict):
        """Stop recording expert distribution."""
        _ = request
        await self.engine.tokenizer_manager.stop_expert_distribution_record()
        yield True

    async def dump_expert_distribution_record(self, request: dict):
        """Dumps the expert distribution record to the directory specified in the environment variable `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR`."""
        _ = request
        await self.engine.tokenizer_manager.dump_expert_distribution_record()
        yield True

class AggregatedWorkerHandler(BaseSGLangHandler):
    """Handler for aggregated mode (replaces current worker/main.py)"""
    
    def __init__(self, engine: sgl.Engine, server_args: ServerArgs, component, 
                 endpoint, config: SGLangConfig, decode_client: Optional[Any] = None):
        super().__init__(engine, server_args, component, endpoint, config, decode_client)
        
        # Setup metrics publishing for aggregated mode
        self.metrics_publisher = WorkerMetricsPublisher()
        
        # Setup ZMQ metrics
        zmq_context = zmq.asyncio.Context()
        self.receive_metrics_from_scheduler = get_zmq_socket(
            zmq_context, zmq.PULL, engine.port_args.metrics_ipc_name, True
        )
        
        # Initial metrics publish
        worker_stats = WorkerStats(
            request_active_slots=0, request_total_slots=1024, 
            num_requests_waiting=0, data_parallel_rank=0
        )
        kv_stats = KvStats(
            kv_active_blocks=0, kv_total_blocks=1024,
            gpu_cache_usage_perc=0, gpu_prefix_cache_hit_rate=0
        )
        initial_metrics = ForwardPassMetrics(
            worker_stats=worker_stats, kv_stats=kv_stats, spec_decode_stats=None
        )
        self.metrics_publisher.publish(initial_metrics)
        
        logger.info("Aggregated handler metrics publisher initialized")
    
    async def setup_metrics_endpoint(self, component):
        """Setup metrics publisher endpoint - called after handler creation"""
        await self.metrics_publisher.create_endpoint(component)
        
        # Start metrics loop
        asyncio.create_task(self._receive_and_publish_metrics_loop())
    
    async def _receive_and_publish_metrics_loop(self):
        """Receive metrics from SGL scheduler and publish them"""
        while True:
            try:
                kv_metrics = await self.receive_metrics_from_scheduler.recv_pyobj()
                worker_stats = WorkerStats(
                    request_active_slots=kv_metrics.request_active_slots,
                    request_total_slots=kv_metrics.request_total_slots,
                    num_requests_waiting=kv_metrics.num_requests_waiting,
                    data_parallel_rank=kv_metrics.data_parallel_rank,
                )
                kv_stats = KvStats(
                    kv_active_blocks=kv_metrics.kv_active_blocks,
                    kv_total_blocks=kv_metrics.kv_total_blocks,
                    gpu_cache_usage_perc=kv_metrics.gpu_cache_usage_perc,
                    gpu_prefix_cache_hit_rate=kv_metrics.gpu_prefix_cache_hit_rate,
                )
                metrics = ForwardPassMetrics(
                    worker_stats=worker_stats,
                    kv_stats=kv_stats,
                    spec_decode_stats=None,
                )
                self.metrics_publisher.publish(metrics)
            except Exception:
                logger.exception("Failed to receive or publish metrics")

class PrefillWorkerHandler(BaseSGLangHandler):
    """Handler for prefill mode (enhanced version of current worker in prefill mode)"""
    
    def __init__(self, engine: sgl.Engine, server_args: ServerArgs, component, 
                 endpoint, config: SGLangConfig, decode_client: Optional[Any] = None):
        super().__init__(engine, server_args, component, endpoint, config, decode_client)
        
        # Setup bootstrap info for disaggregation (moved back from main.py)
        self.bootstrap_host = None
        self.bootstrap_port = None
        
        if self.server_args.disaggregation_mode == "prefill":
            inner_tm = engine.tokenizer_manager
            self.bootstrap_port = inner_tm.server_args.disaggregation_bootstrap_port
            if inner_tm.server_args.dist_init_addr:
                self.bootstrap_host = socket.gethostbyname(
                    inner_tm.server_args.dist_init_addr.split(":")[0]
                )
            else:
                from sglang.srt.utils import get_ip
                self.bootstrap_host = get_ip()
            
            logger.info(f"Prefill handler disaggregation setup - host: {self.bootstrap_host}, port: {self.bootstrap_port}")
        
        # Setup metrics publishing for prefill mode
        self.metrics_publisher = WorkerMetricsPublisher()
        
        # Setup ZMQ metrics
        zmq_context = zmq.asyncio.Context()
        self.receive_metrics_from_scheduler = get_zmq_socket(
            zmq_context, zmq.PULL, engine.port_args.metrics_ipc_name, True
        )
        
        # Initial metrics publish
        worker_stats = WorkerStats(
            request_active_slots=0, request_total_slots=1024, 
            num_requests_waiting=0, data_parallel_rank=0
        )
        kv_stats = KvStats(
            kv_active_blocks=0, kv_total_blocks=1024,
            gpu_cache_usage_perc=0, gpu_prefix_cache_hit_rate=0
        )
        initial_metrics = ForwardPassMetrics(
            worker_stats=worker_stats, kv_stats=kv_stats, spec_decode_stats=None
        )
        self.metrics_publisher.publish(initial_metrics)
        
        logger.info("Prefill handler metrics publisher initialized")
    
    async def setup_metrics_endpoint(self, component):
        """Setup metrics publisher endpoint - called after handler creation"""
        await self.metrics_publisher.create_endpoint(component)
        
        # Start metrics loop
        asyncio.create_task(self._receive_and_publish_metrics_loop())
    
    async def _receive_and_publish_metrics_loop(self):
        """Receive metrics from SGL scheduler and publish them"""
        while True:
            try:
                kv_metrics = await self.receive_metrics_from_scheduler.recv_pyobj()
                worker_stats = WorkerStats(
                    request_active_slots=kv_metrics.request_active_slots,
                    request_total_slots=kv_metrics.request_total_slots,
                    num_requests_waiting=kv_metrics.num_requests_waiting,
                    data_parallel_rank=kv_metrics.data_parallel_rank,
                )
                kv_stats = KvStats(
                    kv_active_blocks=kv_metrics.kv_active_blocks,
                    kv_total_blocks=kv_metrics.kv_total_blocks,
                    gpu_cache_usage_perc=kv_metrics.gpu_cache_usage_perc,
                    gpu_prefix_cache_hit_rate=kv_metrics.gpu_prefix_cache_hit_rate,
                )
                metrics = ForwardPassMetrics(
                    worker_stats=worker_stats,
                    kv_stats=kv_stats,
                    spec_decode_stats=None,
                )
                self.metrics_publisher.publish(metrics)
            except Exception:
                logger.exception("Failed to receive or publish metrics")

class DecodeWorkerHandler(BaseSGLangHandler):
    """Handler for decode mode (enhanced version of current decode_worker/main.py)"""
    
    async def generate(self, request: str):
        """Generate tokens for decode mode - overrides base class"""
        try:
            # Parse the JSON request
            req = msgspec.json.decode(request, type=dict)

            # Enhanced with better error handling and logging
            logger.debug(f"Processing decode request for {len(req.get('request', {}).get('token_ids', []))} tokens")
            
            # Generate using the engine
            results = await self.engine.async_generate(
                input_ids=req["request"]["token_ids"]
                if req["request"]["batch_token_ids"] is None
                else req["request"]["batch_token_ids"],
                sampling_params=req["sampling_params"],
                stream=True,
                bootstrap_host=req["bootstrap_host"],
                bootstrap_port=req["bootstrap_port"],
                bootstrap_room=req["bootstrap_room"],
            )

            # Stream results back
            async for result in results:
                yield result
                
        except Exception as e:
            logger.error(f"Decode generation failed: {e}")
            raise

class RequestHandlerFactory:
    """Factory for creating appropriate request handlers"""
    
    @staticmethod
    def create_handler(
        mode: str,
        engine: sgl.Engine,
        server_args: ServerArgs,
        component,
        endpoint,
        decode_client: Optional[Any] = None,
        config: Optional[SGLangConfig] = None
    ) -> BaseSGLangHandler:
        """Create handler based on disaggregation mode"""
        
        handler_map = {
            "null": AggregatedWorkerHandler,
            "prefill": PrefillWorkerHandler,
            "decode": DecodeWorkerHandler,
        }
        
        handler_class = handler_map.get(mode)
        if not handler_class:
            raise ValueError(f"Unsupported disaggregation mode: {mode}")
        
        logger.info(f"Creating {handler_class.__name__} for mode: {mode}")
        return handler_class(
            engine=engine,
            server_args=server_args,
            component=component,
            endpoint=endpoint,
            config=config,
            decode_client=decode_client
        )
```

### Step 2: Create Backward Compatibility Shims

#### File: `components/backends/sglang/src/dynamo/sglang/worker/__main__.py`

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# BACKWARD COMPATIBILITY SHIM
# This file provides backward compatibility for existing deployments using dynamo.sglang.worker
# New deployments should use dynamo.sglang directly

import sys
import warnings

def main():
    warnings.warn(
        "dynamo.sglang.worker is deprecated. Use 'dynamo.sglang' instead. "
        "This compatibility shim will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import and delegate to new unified entry point
    from ..main import main as unified_main
    unified_main()

if __name__ == "__main__":
    main()
```

#### File: `components/backends/sglang/src/dynamo/sglang/decode_worker/__main__.py`

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# BACKWARD COMPATIBILITY SHIM
# This file provides backward compatibility for existing deployments using dynamo.sglang.decode_worker
# New deployments should use dynamo.sglang directly

import sys
import warnings

def main():
    warnings.warn(
        "dynamo.sglang.decode_worker is deprecated. Use 'dynamo.sglang' instead. "
        "This compatibility shim will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import and delegate to new unified entry point
    from ..main import main as unified_main
    unified_main()

if __name__ == "__main__":
    main()
```

### Step 3: Update Deployment Configurations

#### File: `components/backends/sglang/deploy/agg.yaml`

**CHANGE LINES 96-98 FROM:**
```yaml
args:
  - "python3"
  - "-m"
  - "dynamo.sglang.worker"
```

**TO:**
```yaml
args:
  - "python3"
  - "-m"
  - "dynamo.sglang"
```

#### File: `components/backends/sglang/deploy/disagg.yaml`

**CHANGE LINES 96-98 FROM:**
```yaml
args:
  - "python3"
  - "-m"
  - "dynamo.sglang.decode_worker"
```

**TO:**
```yaml
args:
  - "python3"
  - "-m"
  - "dynamo.sglang"
```

**CHANGE LINES 163-165 FROM:**
```yaml
args:
  - "python3"
  - "-m"
  - "dynamo.sglang.worker"
```

**TO:**
```yaml
args:
  - "python3"
  - "-m"
  - "dynamo.sglang"
```

### Step 4: Update Common Files (if needed)

#### File: `components/backends/sglang/src/dynamo/sglang/common/protocol.py`

Ensure this file exists and contains the `DisaggPreprocessedRequest` class:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel
from typing import Any, Dict, Union, List

class DisaggPreprocessedRequest(BaseModel):
    """Protocol for disaggregated preprocessing requests"""
    request: Dict[str, Any]
    sampling_params: Dict[str, Any]
    bootstrap_host: Union[str, List[str]]
    bootstrap_port: Union[int, List[int]]
    bootstrap_room: Union[int, List[int]]
```

### Step 5: Remove Old Files (Future PR)

**After validation and migration period, these files can be removed:**

- `components/backends/sglang/src/dynamo/sglang/worker/main.py` (378 lines)
- `components/backends/sglang/src/dynamo/sglang/decode_worker/main.py` (95 lines)
- `components/backends/sglang/src/dynamo/sglang/common/base_handlers.py` (functionality moved to `BaseSGLangHandler`)

**Keep the `__main__.py` shim files for backward compatibility until they can be deprecated.**

## Testing Strategy

### Unit Tests to Add

1. **Configuration Tests** (`test_args.py`):
```python
def test_parse_sglang_args_aggregated():
    args = ["--model-path", "test-model"]
    config = parse_sglang_args(args)
    assert config.mode == "null"
    assert config.is_aggregated

def test_parse_sglang_args_prefill():
    args = ["--model-path", "test-model", "--disaggregation-mode", "prefill"]
    config = parse_sglang_args(args)
    assert config.mode == "prefill"
    assert config.is_prefill_worker

def test_parse_sglang_args_decode():
    args = ["--model-path", "test-model", "--disaggregation-mode", "decode"]
    config = parse_sglang_args(args)
    assert config.mode == "decode"
    assert config.is_decode_worker

def test_parse_migration_limit():
    args = ["--model-path", "test-model", "--migration-limit", "10"]
    config = parse_sglang_args(args)
    assert config.migration_limit == 10
```

2. **Handler Factory Tests** (`test_handlers.py`):
```python
import pytest
from unittest.mock import Mock, AsyncMock
from dynamo.sglang.handlers import RequestHandlerFactory, AggregatedWorkerHandler, PrefillWorkerHandler, DecodeWorkerHandler, BaseSGLangHandler
from dynamo.sglang.args import SGLangConfig
from sglang.srt.server_args import ServerArgs

def test_handler_factory_aggregated():
    handler = RequestHandlerFactory.create_handler("null", engine, server_args, component, endpoint, config=config)
    assert isinstance(handler, AggregatedWorkerHandler)
    assert isinstance(handler, BaseSGLangHandler)

def test_handler_factory_prefill():
    handler = RequestHandlerFactory.create_handler("prefill", engine, server_args, component, endpoint, config=config)
    assert isinstance(handler, PrefillWorkerHandler)
    assert isinstance(handler, BaseSGLangHandler)

def test_handler_factory_decode():
    handler = RequestHandlerFactory.create_handler("decode", engine, server_args, component, endpoint, config=config)
    assert isinstance(handler, DecodeWorkerHandler)
    assert isinstance(handler, BaseSGLangHandler)

def test_handler_factory_invalid_mode():
    with pytest.raises(ValueError):
        RequestHandlerFactory.create_handler("invalid", engine, server_args, component, endpoint, config=config)

@pytest.mark.asyncio
async def test_handler_native_endpoints():
    """Test that all handlers support native SGLang endpoints"""
    handler = AggregatedWorkerHandler(engine, server_args, component, endpoint, config)
    
    # Test flush_cache endpoint
    result = [item async for item in handler.flush_cache({})]
    assert result == [True]
    
    # Test expert distribution endpoints if available
    if hasattr(handler.server_args, 'expert_distribution_recorder_mode'):
        result = [item async for item in handler.start_expert_distribution_record({})]
        assert result == [True]
        
        result = [item async for item in handler.stop_expert_distribution_record({})]
        assert result == [True]
        
        result = [item async for item in handler.dump_expert_distribution_record({})]
        assert result == [True]
```

### Integration Tests

1. **End-to-End Mode Tests**: Test all three modes with actual SGLang engines
2. **Deployment Config Tests**: Validate YAML configurations work with new entry point
3. **Backward Compatibility Tests**: Ensure old entry points still work via shims

### Performance Tests

1. **Regression Tests**: Ensure no performance degradation
2. **Memory Tests**: Verify memory usage patterns remain consistent
3. **Throughput Tests**: Validate request handling capacity

## Migration Timeline

### Phase 1: Implementation (This PR)
- [ ] Create new unified architecture files
- [ ] Add backward compatibility shims
- [ ] Update deployment configurations
- [ ] Add comprehensive tests
- [ ] Update documentation

### Phase 2: Validation (Next Sprint)
- [ ] Deploy in staging environments
- [ ] Run performance regression tests
- [ ] Validate all three modes work correctly
- [ ] Gather feedback from users

### Phase 3: Migration (Following Sprint)
- [ ] Update production deployments to use unified entry point
- [ ] Monitor for issues and performance
- [ ] Document migration process for external users

### Phase 4: Cleanup (Future PR)
- [ ] Remove old implementation files (worker/main.py, decode_worker/main.py)
- [ ] Add deprecation warnings to shim files
- [ ] Plan removal of backward compatibility shims

## Benefits Summary

1. **Consistency**: Aligns SGLang with vLLM and TRT-LLM backend patterns
2. **Simplified Deployment**: Single entry point (`dynamo.sglang`) for all modes
3. **Better Maintainability**: Centralized configuration and shared code
4. **Improved Extensibility**: Easier to add new modes or features
5. **Enhanced User Experience**: Unified interface for all SGLang use cases
6. **Backward Compatibility**: Existing deployments continue to work during transition

## Architecture Benefits

1. **Follows vLLM/TRT-LLM Pattern**: All setup done in `main.py:init()` before handler creation
2. **No Complex Initialization**: Handlers receive all dependencies via constructor
3. **Factory Pattern**: Keeps mode-specific logic organized and testable
4. **Performance Optimized**: Single file with consolidated logic, no function call overhead
5. **Extensible**: Easy to add new modes or override behavior in subclasses

## Risk Mitigation

1. **Backward Compatibility Shims**: Old entry points continue to work
2. **Gradual Migration**: Can deploy new architecture alongside existing one
3. **Comprehensive Testing**: Unit, integration, and performance tests
4. **Monitoring**: Track metrics during migration to catch issues early
5. **Rollback Plan**: Can revert to old implementation if needed

---

This mega refactor document contains all the code and steps needed to successfully unify the SGLang backend architecture while maintaining full backward compatibility and following established Dynamo patterns.