# LoRA Support Design for Dynamo (v2)

## Executive Summary

This document outlines a pragmatic, phase-based approach to adding LoRA support to Dynamo, leveraging vLLM's native dynamic LoRA capabilities and Dynamo's existing service discovery infrastructure.

**Key Principles:**
1. **Leverage vLLM Native Features**: Use vLLM's built-in dynamic LoRA loading rather than building custom solutions
2. **Metadata-Driven Discovery**: Extend existing `/metadata` endpoint pattern for LoRA state advertisement
3. **Incremental Rollout**: Phase A focuses on basic serving, Phase B adds intelligent orchestration
4. **KV Cache Safety**: Ensure LoRA-specific block hashes prevent cache pollution

## Current Model Handling Analysis

### How Dynamo Currently Handles Models

Based on `Service-discovery.md`, Dynamo's current model discovery works as follows:

```python
# Worker Registration (Backend)
decode_worker = service_registry.register_instance("dynamo", "decode")
decode_worker.set_metadata({
    "model": {
        "name": "Qwen3-32B",
        "type": "Completions",
        "runtime_config": {
            "total_kv_blocks": 24064,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 2048
        }
    },
    "transport": {...},
    "mdc": {...}  # Model Deployment Card
})
decode_worker.set_ready("ready")

# Frontend Discovery (Router)
workers = service_discovery.list_instances("dynamo", "decode")
for worker in workers:
    metadata = worker.metadata()
    model = metadata["model"]
    # Register model in in-memory registry
    # Map instance to model
```

**Current Flow:**
1. Worker starts, loads base model
2. Worker registers with service discovery
3. Worker publishes metadata via `/metadata` endpoint
4. Frontend discovers workers via EndpointSlices
5. Frontend fetches metadata from each worker
6. Frontend builds model → worker mapping

**Key Insight:** The metadata mechanism is perfect for advertising LoRA state!

---

## Phase A: Simple LoRA Serving

**Goal:** Enable serving requests with pre-loaded LoRA adapters without dynamic loading/unloading.

### A.1 LoRA Metadata Schema

Extend the metadata structure to include LoRA information:

```python
# Enhanced metadata structure
metadata = {
    "model": {
        "name": "Qwen/Qwen2.5-Coder-7B",
        "type": "Completions",
        "base_model": "Qwen/Qwen2.5-Coder-7B",  # NEW: explicit base model
        "runtime_config": {...}
    },
    "transport": {...},
    "mdc": {...},

    # NEW: LoRA capabilities and state
    "lora": {
        "enabled": true,
        "max_loras": 4,
        "max_lora_rank": 64,

        # Available LoRAs on disk (discovered at startup)
        "available_loras": [
            {
                "lora_id": "sql-expert/v1",
                "path": "/shared/loras/sql-expert/v1",
                "base_model": "Qwen/Qwen2.5-Coder-7B",
                "rank": 32,
                "state": "on_disk"  # on_disk, loading, ready, failed
            },
            {
                "lora_id": "python-expert/v1",
                "path": "/shared/loras/python-expert/v1",
                "base_model": "Qwen/Qwen2.5-Coder-7B",
                "rank": 64,
                "state": "on_disk"
            }
        ],

        # Currently loaded LoRAs (ready for inference)
        "loaded_loras": [
            {
                "lora_id": "sql-expert/v1",
                "state": "ready",
                "loaded_at": "2025-10-17T10:30:00Z",
                "memory_mb": 256,
                "last_used": "2025-10-17T11:45:23Z"
            }
        ],

        "capacity": {
            "loaded_count": 1,
            "available_slots": 3
        }
    }
}
```

### A.2 LoRA Discovery at Startup

Workers discover available LoRAs from shared storage at startup:

```python
# components/src/dynamo/common/lora_discovery.py

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class LoRAAdapter:
    lora_id: str
    path: str
    base_model: str
    rank: int
    metadata: Dict

class LoRADiscovery:
    """Discovers LoRA adapters from shared storage"""

    def __init__(self, lora_root: str):
        self.lora_root = Path(lora_root)

    def discover_loras(self) -> List[LoRAAdapter]:
        """
        Scan shared storage for LoRA adapters.

        Expected structure:
        /shared/loras/
        ├── sql-expert/
        │   └── v1/
        │       ├── adapter_config.json
        │       ├── adapter_model.safetensors
        │       └── metadata.json (optional)
        ├── python-expert/
        │   └── v1/
        │       ├── adapter_config.json
        │       └── adapter_model.safetensors
        """
        adapters = []

        if not self.lora_root.exists():
            return adapters

        # Walk directory structure
        for lora_dir in self.lora_root.rglob("adapter_config.json"):
            adapter_path = lora_dir.parent

            # Read adapter config
            with open(lora_dir) as f:
                config = json.load(f)

            # Construct lora_id from path relative to root
            rel_path = adapter_path.relative_to(self.lora_root)
            lora_id = str(rel_path)

            # Read optional metadata
            metadata_file = adapter_path / "metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

            adapter = LoRAAdapter(
                lora_id=lora_id,
                path=str(adapter_path),
                base_model=config.get("base_model_name_or_path", ""),
                rank=config.get("r", 0),
                metadata=metadata
            )

            adapters.append(adapter)

        return adapters
```

### A.3 Backend Worker with Static LoRA Loading

Workers pre-load configured LoRAs at startup:

```python
# components/src/dynamo/vllm/lora_worker.py

import os
import logging
from typing import List, Dict, Optional
from vllm import AsyncLLMEngine
from vllm.lora.request import LoRARequest

from dynamo.common.lora_discovery import LoRADiscovery

logger = logging.getLogger(__name__)

class LoRAWorker:
    """Worker with LoRA support"""

    def __init__(self, config):
        self.config = config
        self.engine: Optional[AsyncLLMEngine] = None
        self.lora_discovery = LoRADiscovery(
            os.environ.get("DYNAMO_LORA_PATH", "/shared/loras")
        )

        # State tracking
        self.available_loras: Dict[str, Dict] = {}  # lora_id -> info
        self.loaded_loras: Dict[str, LoRARequest] = {}  # lora_id -> LoRARequest

    async def initialize(self):
        """Initialize vLLM engine with LoRA support"""

        # Create vLLM engine with LoRA enabled
        engine_args = {
            "model": self.config.model,
            "enable_lora": True,
            "max_loras": self.config.max_loras,
            "max_lora_rank": self.config.max_lora_rank,
            # ... other vLLM config
        }

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Discover available LoRAs
        await self._discover_available_loras()

        # Pre-load configured LoRAs
        await self._preload_loras()

    async def _discover_available_loras(self):
        """Discover LoRAs from shared storage"""
        adapters = self.lora_discovery.discover_loras()

        for adapter in adapters:
            self.available_loras[adapter.lora_id] = {
                "path": adapter.path,
                "base_model": adapter.base_model,
                "rank": adapter.rank,
                "state": "on_disk",
                "metadata": adapter.metadata
            }

        logger.info(f"Discovered {len(self.available_loras)} LoRA adapters")

    async def _preload_loras(self):
        """Pre-load LoRAs specified in config"""
        preload_loras = self.config.preload_loras or []

        for lora_id in preload_loras:
            if lora_id in self.available_loras:
                await self._load_lora(lora_id)
            else:
                logger.warning(f"LoRA {lora_id} not found for preloading")

    async def _load_lora(self, lora_id: str) -> bool:
        """Load a LoRA adapter into vLLM"""
        if lora_id in self.loaded_loras:
            logger.info(f"LoRA {lora_id} already loaded")
            return True

        if lora_id not in self.available_loras:
            logger.error(f"LoRA {lora_id} not available")
            return False

        try:
            lora_info = self.available_loras[lora_id]
            lora_info["state"] = "loading"

            # Create LoRARequest for vLLM
            lora_request = LoRARequest(
                lora_name=lora_id,
                lora_int_id=self._generate_lora_int_id(lora_id),
                lora_path=lora_info["path"]
            )

            # vLLM will load the LoRA when it receives first request
            # For Phase A, we just track it as available
            self.loaded_loras[lora_id] = lora_request
            lora_info["state"] = "ready"

            logger.info(f"Loaded LoRA {lora_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load LoRA {lora_id}: {e}")
            self.available_loras[lora_id]["state"] = "failed"
            return False

    def _generate_lora_int_id(self, lora_id: str) -> int:
        """Generate stable integer ID for LoRA"""
        return abs(hash(lora_id)) % 1_000_000

    def get_lora_request(self, lora_id: str) -> Optional[LoRARequest]:
        """Get LoRARequest for inference"""
        return self.loaded_loras.get(lora_id)

    def get_metadata(self) -> Dict:
        """Generate metadata for service discovery"""
        return {
            "model": {
                "name": self.config.model,
                "type": "Completions",
                "base_model": self.config.model,
                "runtime_config": {...}
            },
            "transport": {...},
            "lora": {
                "enabled": True,
                "max_loras": self.config.max_loras,
                "max_lora_rank": self.config.max_lora_rank,

                "available_loras": [
                    {
                        "lora_id": lora_id,
                        "path": info["path"],
                        "base_model": info["base_model"],
                        "rank": info["rank"],
                        "state": info["state"]
                    }
                    for lora_id, info in self.available_loras.items()
                ],

                "loaded_loras": [
                    {
                        "lora_id": lora_id,
                        "state": "ready"
                    }
                    for lora_id in self.loaded_loras.keys()
                ],

                "capacity": {
                    "loaded_count": len(self.loaded_loras),
                    "available_slots": self.config.max_loras - len(self.loaded_loras)
                }
            }
        }
```

### A.4 New Endpoint: `/v1/metadata/loras`

Dedicated endpoint for detailed LoRA information:

```python
# components/src/dynamo/vllm/lora_metadata_endpoint.py

from fastapi import FastAPI, HTTPException
from typing import Dict, List

class LoRAMetadataEndpoint:
    """Dedicated endpoint for LoRA metadata"""

    def __init__(self, lora_worker):
        self.lora_worker = lora_worker
        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self):

        @self.app.get("/v1/metadata/loras")
        async def list_loras():
            """List all LoRAs (available and loaded)"""
            return {
                "available_loras": [
                    {
                        "lora_id": lora_id,
                        "path": info["path"],
                        "base_model": info["base_model"],
                        "rank": info["rank"],
                        "state": info["state"],
                        "metadata": info.get("metadata", {})
                    }
                    for lora_id, info in self.lora_worker.available_loras.items()
                ],
                "loaded_loras": [
                    {
                        "lora_id": lora_id,
                        "state": "ready",
                        "lora_int_id": req.lora_int_id
                    }
                    for lora_id, req in self.lora_worker.loaded_loras.items()
                ],
                "capacity": {
                    "max_loras": self.lora_worker.config.max_loras,
                    "loaded_count": len(self.lora_worker.loaded_loras),
                    "available_slots": (
                        self.lora_worker.config.max_loras -
                        len(self.lora_worker.loaded_loras)
                    )
                }
            }

        @self.app.get("/v1/metadata/loras/{lora_id}")
        async def get_lora_details(lora_id: str):
            """Get details for specific LoRA"""
            if lora_id not in self.lora_worker.available_loras:
                raise HTTPException(status_code=404, detail="LoRA not found")

            info = self.lora_worker.available_loras[lora_id]
            is_loaded = lora_id in self.lora_worker.loaded_loras

            return {
                "lora_id": lora_id,
                "path": info["path"],
                "base_model": info["base_model"],
                "rank": info["rank"],
                "state": info["state"],
                "loaded": is_loaded,
                "metadata": info.get("metadata", {}),
                "lora_int_id": (
                    self.lora_worker.loaded_loras[lora_id].lora_int_id
                    if is_loaded else None
                )
            }
```

### A.5 Enhanced Request Handler

Handle requests with LoRA specification:

```python
# components/src/dynamo/vllm/handlers.py

from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams

async def generate(self, request, context):
    """Generate with optional LoRA"""

    # Extract LoRA ID from request
    lora_id = request.get("lora_id")
    lora_request = None

    if lora_id:
        lora_request = self.lora_worker.get_lora_request(lora_id)
        if not lora_request:
            yield {
                "error": f"LoRA '{lora_id}' not loaded on this worker",
                "available_loras": list(self.lora_worker.loaded_loras.keys())
            }
            return

    # Create prompt and sampling params
    prompt = TokensPrompt(prompt_token_ids=request["token_ids"])
    sampling_params = SamplingParams(**self.default_sampling_params)

    # Apply request-specific sampling params
    for key, value in request.get("sampling_options", {}).items():
        if value is not None and hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    # Generate with LoRA
    request_id = str(uuid.uuid4().hex)

    async for output in self.engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
        lora_request=lora_request  # Pass LoRA to vLLM
    ):
        yield self._format_output(output)
```

### A.6 Router Changes: LoRA-Aware Block Hashing

**Critical:** Block hashes MUST include LoRA ID to prevent cache pollution.

```python
# lib/llm/src/router/lora_kv_router.py (conceptual - would be Rust)

import hashlib
from typing import Optional, List

class LoRAKVRouter:
    """KV Router with LoRA-aware block hashing"""

    def compute_block_hash(
        self,
        token_ids: List[int],
        lora_id: Optional[str] = None
    ) -> str:
        """
        Compute KV block hash including LoRA ID.

        This prevents cache pollution between base model and LoRA requests.
        E.g., same prompt with different LoRAs should have different hashes.
        """
        hasher = hashlib.sha256()

        # Hash token IDs
        for token_id in token_ids:
            hasher.update(token_id.to_bytes(4, byteorder='little'))

        # Include LoRA ID in hash
        if lora_id:
            hasher.update(b"__lora__")
            hasher.update(lora_id.encode('utf-8'))
        else:
            hasher.update(b"__base_model__")

        return hasher.hexdigest()

    def route_request(self, request: dict) -> int:
        """
        Route request to worker with LoRA loaded.

        Phase A: Simple routing
        - If lora_id specified, route to worker with that LoRA loaded
        - Use KV-aware routing among eligible workers
        """
        lora_id = request.get("lora_id")
        token_ids = request.get("token_ids", [])

        # Compute LoRA-aware block hash
        block_hash = self.compute_block_hash(token_ids, lora_id)

        # Get eligible workers
        if lora_id:
            # Filter to workers with this LoRA loaded
            eligible_workers = self._get_workers_with_lora(lora_id)
            if not eligible_workers:
                raise ValueError(f"No workers available with LoRA {lora_id}")
        else:
            # Any worker can handle base model request
            eligible_workers = self._get_all_workers()

        # Use KV routing logic among eligible workers
        best_worker = self._select_best_worker_for_hash(
            block_hash,
            eligible_workers
        )

        return best_worker

    def _get_workers_with_lora(self, lora_id: str) -> List[int]:
        """Get workers that have specific LoRA loaded"""
        workers = []

        for worker_id, metadata in self.worker_metadata.items():
            lora_info = metadata.get("lora", {})
            loaded_loras = lora_info.get("loaded_loras", [])

            # Check if this worker has the LoRA loaded
            for lora in loaded_loras:
                if lora["lora_id"] == lora_id and lora["state"] == "ready":
                    workers.append(worker_id)
                    break

        return workers
```

### A.7 Configuration

```yaml
# config/lora_worker.yaml

# vLLM configuration
model: "Qwen/Qwen2.5-Coder-7B"
enable_lora: true
max_loras: 4
max_lora_rank: 64

# LoRA storage
lora_path: "/shared/loras"

# Pre-load these LoRAs at startup
preload_loras:
  - "sql-expert/v1"
  - "python-expert/v1"

# Service discovery
namespace: "dynamo"
component: "worker"
```

### A.8 Deployment (Kubernetes)

```yaml
# deploy/lora-worker.yaml

apiVersion: v1
kind: Pod
metadata:
  name: dynamo-worker-lora
  labels:
    nvidia.com/dynamo-namespace: dynamo
    nvidia.com/dynamo-component: worker
    nvidia.com/dynamo-lora-enabled: "true"
spec:
  containers:
  - name: vllm-worker
    image: dynamo/vllm-worker:latest
    env:
    - name: DYNAMO_LORA_PATH
      value: "/shared/loras"
    - name: VLLM_ENABLE_LORA
      value: "true"
    - name: VLLM_MAX_LORAS
      value: "4"
    - name: VLLM_MAX_LORA_RANK
      value: "64"
    - name: VLLM_PRELOAD_LORAS
      value: "sql-expert/v1,python-expert/v1"

    volumeMounts:
    - name: lora-storage
      mountPath: /shared/loras
      readOnly: true

    readinessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10

  volumes:
  - name: lora-storage
    persistentVolumeClaim:
      claimName: dynamo-lora-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dynamo-lora-pvc
spec:
  accessModes:
  - ReadOnlyMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: nfs-client
```

### A.9 Phase A Testing

```python
# Test basic LoRA serving

import requests

# 1. Check worker metadata
response = requests.get("http://worker:8000/metadata")
metadata = response.json()
print(f"Available LoRAs: {metadata['lora']['available_loras']}")
print(f"Loaded LoRAs: {metadata['lora']['loaded_loras']}")

# 2. Check dedicated LoRA endpoint
response = requests.get("http://worker:8000/v1/metadata/loras")
loras = response.json()
print(f"Total available: {len(loras['available_loras'])}")
print(f"Total loaded: {len(loras['loaded_loras'])}")

# 3. Send request with LoRA
response = requests.post(
    "http://router:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-Coder-7B",
        "lora_id": "sql-expert/v1",
        "messages": [
            {
                "role": "user",
                "content": "Write SQL to get top 10 customers by revenue"
            }
        ],
        "max_tokens": 200
    }
)
print(response.json())

# 4. Send request without LoRA (base model)
response = requests.post(
    "http://router:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-Coder-7B",
        "messages": [
            {
                "role": "user",
                "content": "Explain quantum computing"
            }
        ],
        "max_tokens": 200
    }
)
print(response.json())
```

### A.10 Phase A Limitations

1. **Static Loading**: LoRAs must be pre-configured and loaded at worker startup
2. **No Dynamic Loading**: Cannot load new LoRAs without restarting worker
3. **Manual Distribution**: Operators must manually ensure LoRAs are loaded on enough workers
4. **No Load Balancing**: No intelligent distribution of LoRAs across fleet
5. **No Auto-Scaling**: Cannot automatically adjust LoRA placement based on demand

**Phase A is sufficient for:**
- Fixed set of LoRAs known at deployment time
- Small number of LoRAs (2-4 per worker)
- Predictable workload patterns
- Initial LoRA feature rollout and validation

---

## Phase B: Dynamic LoRA Management

**Goal:** Enable dynamic LoRA loading/unloading with intelligent orchestration across the fleet.

### B.1 vLLM Dynamic Loading Integration

Leverage vLLM's runtime LoRA capabilities:

```python
# components/src/dynamo/vllm/dynamic_lora_manager.py

import asyncio
import logging
import time
import aiohttp
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DynamicLoRAManager:
    """
    Manages dynamic LoRA loading/unloading using vLLM's native APIs.

    References:
    - https://docs.vllm.ai/en/stable/features/lora.html#dynamically-serving-lora-adapters
    """

    def __init__(self, vllm_server_url: str, max_loras: int = 4):
        self.vllm_server_url = vllm_server_url
        self.max_loras = max_loras

        # State tracking
        self.loaded_loras: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()

    async def load_lora(
        self,
        lora_id: str,
        lora_path: str,
        base_model: Optional[str] = None
    ) -> bool:
        """
        Load LoRA dynamically using vLLM API.

        POST /v1/load_lora_adapter
        {
            "lora_name": "sql_adapter",
            "lora_path": "/shared/loras/sql-expert/v1"
        }
        """
        async with self._lock:
            try:
                # Check if already loaded
                if lora_id in self.loaded_loras:
                    logger.info(f"LoRA {lora_id} already loaded")
                    return True

                # Check capacity
                if len(self.loaded_loras) >= self.max_loras:
                    # Need to evict LRU LoRA
                    evicted = await self._evict_lru_lora()
                    if not evicted:
                        logger.error("Failed to evict LoRA for capacity")
                        return False

                # Call vLLM load API
                start_time = time.time()

                async with aiohttp.ClientSession() as session:
                    payload = {
                        "lora_name": lora_id,
                        "lora_path": lora_path
                    }

                    async with session.post(
                        f"{self.vllm_server_url}/v1/load_lora_adapter",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)  # 5 min timeout
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.text()

                            load_time = time.time() - start_time

                            # Track loaded LoRA
                            self.loaded_loras[lora_id] = {
                                "lora_path": lora_path,
                                "base_model": base_model,
                                "loaded_at": datetime.utcnow().isoformat(),
                                "load_time_sec": load_time,
                                "last_used": time.time(),
                                "request_count": 0
                            }

                            logger.info(
                                f"Successfully loaded LoRA {lora_id} "
                                f"in {load_time:.2f}s"
                            )
                            return True
                        else:
                            error = await resp.text()
                            logger.error(
                                f"Failed to load LoRA {lora_id}: "
                                f"{resp.status} - {error}"
                            )
                            return False

            except Exception as e:
                logger.error(f"Error loading LoRA {lora_id}: {e}")
                return False

    async def unload_lora(self, lora_id: str) -> bool:
        """
        Unload LoRA using vLLM API.

        POST /v1/unload_lora_adapter
        {
            "lora_name": "sql_adapter"
        }
        """
        async with self._lock:
            try:
                if lora_id not in self.loaded_loras:
                    logger.info(f"LoRA {lora_id} not loaded")
                    return True

                # Call vLLM unload API
                async with aiohttp.ClientSession() as session:
                    payload = {"lora_name": lora_id}

                    async with session.post(
                        f"{self.vllm_server_url}/v1/unload_lora_adapter",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as resp:
                        if resp.status == 200:
                            del self.loaded_loras[lora_id]
                            logger.info(f"Successfully unloaded LoRA {lora_id}")
                            return True
                        else:
                            error = await resp.text()
                            logger.error(
                                f"Failed to unload LoRA {lora_id}: "
                                f"{resp.status} - {error}"
                            )
                            return False

            except Exception as e:
                logger.error(f"Error unloading LoRA {lora_id}: {e}")
                return False

    async def _evict_lru_lora(self) -> bool:
        """Evict least recently used LoRA"""
        if not self.loaded_loras:
            return False

        # Find LRU
        lru_lora_id = min(
            self.loaded_loras.keys(),
            key=lambda x: self.loaded_loras[x]["last_used"]
        )

        logger.info(f"Evicting LRU LoRA: {lru_lora_id}")
        return await self.unload_lora(lru_lora_id)

    def mark_lora_used(self, lora_id: str):
        """Update last used time for LoRA"""
        if lora_id in self.loaded_loras:
            self.loaded_loras[lora_id]["last_used"] = time.time()
            self.loaded_loras[lora_id]["request_count"] += 1

    def get_loaded_loras(self) -> Dict[str, Dict]:
        """Get currently loaded LoRAs"""
        return dict(self.loaded_loras)
```

### B.2 Enhanced Worker Endpoints

Add management endpoints to workers:

```python
# components/src/dynamo/vllm/management_endpoints.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

class LoadLoRARequest(BaseModel):
    lora_id: str
    lora_path: str
    base_model: Optional[str] = None

class UnloadLoRARequest(BaseModel):
    lora_id: str

class LoRAManagementEndpoints:
    """Management endpoints for dynamic LoRA operations"""

    def __init__(self, dynamic_lora_manager):
        self.manager = dynamic_lora_manager
        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self):

        @self.app.post("/v1/admin/load_lora")
        async def load_lora(
            request: LoadLoRARequest,
            background_tasks: BackgroundTasks
        ):
            """Load LoRA adapter"""
            success = await self.manager.load_lora(
                lora_id=request.lora_id,
                lora_path=request.lora_path,
                base_model=request.base_model
            )

            if success:
                return {
                    "status": "success",
                    "lora_id": request.lora_id,
                    "message": "LoRA loaded successfully"
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to load LoRA"
                )

        @self.app.post("/v1/admin/unload_lora")
        async def unload_lora(request: UnloadLoRARequest):
            """Unload LoRA adapter"""
            success = await self.manager.unload_lora(request.lora_id)

            if success:
                return {
                    "status": "success",
                    "lora_id": request.lora_id,
                    "message": "LoRA unloaded successfully"
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to unload LoRA"
                )

        @self.app.get("/v1/admin/loras")
        async def list_loaded_loras():
            """List currently loaded LoRAs"""
            return {
                "loaded_loras": self.manager.get_loaded_loras(),
                "capacity": {
                    "max_loras": self.manager.max_loras,
                    "loaded_count": len(self.manager.loaded_loras),
                    "available_slots": (
                        self.manager.max_loras -
                        len(self.manager.loaded_loras)
                    )
                }
            }
```

### B.3 Centralized LoRA Manager

Orchestrates LoRA placement across the fleet:

```python
# components/src/dynamo/lora_manager/orchestrator.py

import asyncio
import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LoadingPolicy(Enum):
    LAZY = "lazy"  # Load on-demand
    EAGER = "eager"  # Pre-load popular LoRAs
    WEIGHTED = "weighted"  # Load based on priority

@dataclass
class LoRAPlacement:
    lora_id: str
    worker_ids: List[str]
    target_replicas: int

class LoRAOrchestrator:
    """
    Centralized LoRA orchestrator for the fleet.

    Responsibilities:
    1. Monitor LoRA usage patterns
    2. Decide LoRA placement across workers
    3. Trigger load/unload operations
    4. Rebalance LoRAs based on demand
    """

    def __init__(
        self,
        service_discovery,
        loading_policy: LoadingPolicy = LoadingPolicy.LAZY
    ):
        self.service_discovery = service_discovery
        self.loading_policy = loading_policy

        # State
        self.lora_placements: Dict[str, LoRAPlacement] = {}
        self.worker_states: Dict[str, Dict] = {}  # worker_id -> state
        self.lora_usage: Dict[str, int] = defaultdict(int)  # lora_id -> request_count

        # Configuration
        self.min_replicas_per_lora = 1
        self.target_replicas_per_lora = 2
        self.rebalance_interval_sec = 300  # 5 minutes

    async def start(self):
        """Start orchestrator"""
        logger.info("Starting LoRA Orchestrator")

        # Discover initial worker state
        await self._sync_worker_states()

        # Start background tasks
        asyncio.create_task(self._watch_workers())
        asyncio.create_task(self._periodic_rebalance())

    async def request_lora(self, lora_id: str) -> List[str]:
        """
        Request a LoRA to be available.
        Returns list of worker IDs where LoRA is loaded.
        """
        # Track usage
        self.lora_usage[lora_id] += 1

        # Check if already loaded somewhere
        if lora_id in self.lora_placements:
            placement = self.lora_placements[lora_id]
            if placement.worker_ids:
                return placement.worker_ids

        # Need to load LoRA
        logger.info(f"Requesting LoRA {lora_id} - not currently loaded")

        # Select target worker(s)
        target_workers = await self._select_workers_for_lora(
            lora_id,
            replicas=self.min_replicas_per_lora
        )

        # Issue load commands
        loaded_workers = []
        for worker_id in target_workers:
            success = await self._load_lora_on_worker(lora_id, worker_id)
            if success:
                loaded_workers.append(worker_id)

        # Update placement
        self.lora_placements[lora_id] = LoRAPlacement(
            lora_id=lora_id,
            worker_ids=loaded_workers,
            target_replicas=self.target_replicas_per_lora
        )

        return loaded_workers

    async def _select_workers_for_lora(
        self,
        lora_id: str,
        replicas: int
    ) -> List[str]:
        """Select best workers to load LoRA"""
        candidates = []

        for worker_id, state in self.worker_states.items():
            lora_info = state.get("metadata", {}).get("lora", {})
            capacity = lora_info.get("capacity", {})
            available_slots = capacity.get("available_slots", 0)

            if available_slots > 0:
                # Check if LoRA is available on worker's disk
                available_loras = lora_info.get("available_loras", [])
                lora_available = any(
                    lora["lora_id"] == lora_id
                    for lora in available_loras
                )

                if lora_available:
                    candidates.append((worker_id, available_slots))

        # Sort by available capacity (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Select top N workers
        selected = [worker_id for worker_id, _ in candidates[:replicas]]

        return selected

    async def _load_lora_on_worker(
        self,
        lora_id: str,
        worker_id: str
    ) -> bool:
        """Issue load command to specific worker"""
        try:
            worker_state = self.worker_states.get(worker_id)
            if not worker_state:
                logger.error(f"Worker {worker_id} not found")
                return False

            # Get LoRA path from worker's available LoRAs
            metadata = worker_state.get("metadata", {})
            lora_info = metadata.get("lora", {})
            available_loras = lora_info.get("available_loras", [])

            lora_data = next(
                (l for l in available_loras if l["lora_id"] == lora_id),
                None
            )

            if not lora_data:
                logger.error(f"LoRA {lora_id} not available on worker {worker_id}")
                return False

            lora_path = lora_data["path"]
            worker_address = worker_state["address"]

            # Call worker's load endpoint
            async with aiohttp.ClientSession() as session:
                payload = {
                    "lora_id": lora_id,
                    "lora_path": lora_path,
                    "base_model": lora_data.get("base_model")
                }

                async with session.post(
                    f"{worker_address}/v1/admin/load_lora",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    if resp.status == 200:
                        logger.info(
                            f"Successfully loaded LoRA {lora_id} "
                            f"on worker {worker_id}"
                        )
                        return True
                    else:
                        error = await resp.text()
                        logger.error(
                            f"Failed to load LoRA {lora_id} "
                            f"on worker {worker_id}: {error}"
                        )
                        return False

        except Exception as e:
            logger.error(
                f"Error loading LoRA {lora_id} on worker {worker_id}: {e}"
            )
            return False

    async def _sync_worker_states(self):
        """Sync worker states from service discovery"""
        workers = self.service_discovery.list_instances("dynamo", "worker")

        for worker in workers:
            try:
                metadata = await worker.metadata()
                self.worker_states[worker.instance_id] = {
                    "address": worker.address,
                    "metadata": metadata
                }
            except Exception as e:
                logger.error(
                    f"Failed to get metadata from worker {worker.instance_id}: {e}"
                )

    async def _watch_workers(self):
        """Watch for worker changes"""
        event_stream = self.service_discovery.watch("dynamo", "worker")

        async for event in event_stream:
            try:
                if hasattr(event, 'instance_added'):
                    await self._handle_worker_added(event.instance_added)
                elif hasattr(event, 'instance_removed'):
                    await self._handle_worker_removed(event.instance_removed)
            except Exception as e:
                logger.error(f"Error handling worker event: {e}")

    async def _handle_worker_added(self, worker):
        """Handle new worker"""
        logger.info(f"Worker added: {worker.instance_id}")
        metadata = await worker.metadata()
        self.worker_states[worker.instance_id] = {
            "address": worker.address,
            "metadata": metadata
        }

    async def _handle_worker_removed(self, worker):
        """Handle worker removal"""
        logger.info(f"Worker removed: {worker.instance_id}")

        worker_id = worker.instance_id

        # Remove from worker states
        if worker_id in self.worker_states:
            del self.worker_states[worker_id]

        # Update LoRA placements
        for lora_id, placement in self.lora_placements.items():
            if worker_id in placement.worker_ids:
                placement.worker_ids.remove(worker_id)

                # Trigger re-placement if below target
                if len(placement.worker_ids) < placement.target_replicas:
                    asyncio.create_task(
                        self._ensure_lora_replicas(lora_id)
                    )

    async def _ensure_lora_replicas(self, lora_id: str):
        """Ensure LoRA has target number of replicas"""
        placement = self.lora_placements.get(lora_id)
        if not placement:
            return

        current_replicas = len(placement.worker_ids)
        needed_replicas = placement.target_replicas - current_replicas

        if needed_replicas > 0:
            logger.info(
                f"LoRA {lora_id} needs {needed_replicas} more replicas"
            )

            # Select and load on additional workers
            new_workers = await self._select_workers_for_lora(
                lora_id,
                replicas=needed_replicas
            )

            for worker_id in new_workers:
                if worker_id not in placement.worker_ids:
                    success = await self._load_lora_on_worker(lora_id, worker_id)
                    if success:
                        placement.worker_ids.append(worker_id)

    async def _periodic_rebalance(self):
        """Periodic rebalancing based on usage"""
        while True:
            try:
                await asyncio.sleep(self.rebalance_interval_sec)
                await self._rebalance_loras()
            except Exception as e:
                logger.error(f"Error in rebalance: {e}")

    async def _rebalance_loras(self):
        """Rebalance LoRAs based on usage patterns"""
        logger.info("Starting LoRA rebalancing")

        # Get top LoRAs by usage
        sorted_loras = sorted(
            self.lora_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Ensure hot LoRAs have more replicas
        for lora_id, usage_count in sorted_loras[:10]:  # Top 10
            placement = self.lora_placements.get(lora_id)
            if placement:
                # Increase replicas for hot LoRAs
                desired_replicas = min(
                    len(self.worker_states),
                    self.target_replicas_per_lora * 2
                )

                if len(placement.worker_ids) < desired_replicas:
                    logger.info(
                        f"Increasing replicas for hot LoRA {lora_id} "
                        f"(usage: {usage_count})"
                    )
                    placement.target_replicas = desired_replicas
                    await self._ensure_lora_replicas(lora_id)

        # Reset usage counters
        self.lora_usage.clear()
```

### B.4 Router Integration with Orchestrator

Router queries orchestrator for LoRA availability:

```python
# lib/llm/src/router/lora_aware_router.py

class LoRARouter:
    """Router with dynamic LoRA support"""

    def __init__(self, lora_orchestrator):
        self.orchestrator = lora_orchestrator
        self.kv_router = LoRAKVRouter()

    async def route_request(self, request: dict) -> str:
        """Route request with dynamic LoRA loading"""
        lora_id = request.get("lora_id")
        token_ids = request.get("token_ids", [])

        # Compute LoRA-aware block hash
        block_hash = self.kv_router.compute_block_hash(token_ids, lora_id)

        if lora_id:
            # Ensure LoRA is loaded somewhere
            worker_ids = await self.orchestrator.request_lora(lora_id)

            if not worker_ids:
                raise ValueError(
                    f"LoRA {lora_id} could not be loaded on any worker"
                )

            # Route among workers with LoRA
            best_worker = self.kv_router.select_best_worker_for_hash(
                block_hash,
                worker_ids
            )
        else:
            # Route to any worker
            best_worker = self.kv_router.select_best_worker_for_hash(
                block_hash,
                self.kv_router.get_all_workers()
            )

        return best_worker
```

### B.5 Phase B Configuration

```yaml
# config/lora_orchestrator.yaml

lora_orchestrator:
  # Loading policy
  policy: "lazy"  # lazy, eager, weighted

  # Replica targets
  min_replicas_per_lora: 1
  target_replicas_per_lora: 2
  max_replicas_per_lora: 4

  # Rebalancing
  rebalance_interval_sec: 300
  enable_auto_scaling: true

  # Eviction
  lru_eviction_enabled: true
  ttl_eviction_enabled: false
  ttl_seconds: 3600

# Worker configuration
worker:
  enable_dynamic_loading: true
  vllm_server_url: "http://localhost:8000"
  max_loras: 4
  max_lora_rank: 64
```

### B.6 Phase B Deployment

```yaml
# deploy/lora-orchestrator.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamo-lora-orchestrator
spec:
  replicas: 1  # Single orchestrator for now
  selector:
    matchLabels:
      app: dynamo-lora-orchestrator
  template:
    metadata:
      labels:
        app: dynamo-lora-orchestrator
    spec:
      containers:
      - name: orchestrator
        image: dynamo/lora-orchestrator:latest
        env:
        - name: DYNAMO_NAMESPACE
          value: "dynamo"
        - name: DYNAMO_COMPONENT
          value: "worker"
        - name: LORA_POLICY
          value: "lazy"
        - name: REBALANCE_INTERVAL_SEC
          value: "300"
        ports:
        - containerPort: 8080
          name: http
---
apiVersion: v1
kind: Service
metadata:
  name: dynamo-lora-orchestrator
spec:
  selector:
    app: dynamo-lora-orchestrator
  ports:
  - port: 8080
    targetPort: 8080
    name: http
```

---

## Implementation Roadmap

### Phase A: Simple LoRA Serving (4-6 weeks)

**Week 1-2: Core Infrastructure**
- [ ] Implement LoRA discovery system
- [ ] Add LoRA metadata to service discovery
- [ ] Create `/v1/metadata/loras` endpoint
- [ ] Update worker initialization with LoRA support

**Week 3-4: Router Integration**
- [ ] Implement LoRA-aware block hashing (Rust)
- [ ] Update router to filter workers by loaded LoRAs
- [ ] Add LoRA request handling in frontend
- [ ] Update request format to include `lora_id`

**Week 5-6: Testing & Documentation**
- [ ] End-to-end testing with 2-3 LoRAs
- [ ] Performance benchmarking (latency, throughput)
- [ ] Cache hit rate validation
- [ ] Documentation and deployment guides

**Success Criteria:**
- ✅ Workers can discover and serve pre-loaded LoRAs
- ✅ Router correctly routes requests to workers with specific LoRAs
- ✅ KV cache does not pollute across different LoRAs
- ✅ Base model requests continue to work normally

### Phase B: Dynamic Management (6-8 weeks)

**Week 1-2: Dynamic Loading**
- [ ] Integrate vLLM dynamic loading APIs
- [ ] Implement DynamicLoRAManager
- [ ] Add worker management endpoints
- [ ] LRU eviction logic

**Week 3-4: Orchestrator**
- [ ] Implement central LoRA orchestrator
- [ ] Worker state synchronization
- [ ] LoRA placement logic
- [ ] Replica management

**Week 5-6: Intelligence**
- [ ] Usage tracking and analytics
- [ ] Automatic rebalancing
- [ ] Hot LoRA detection
- [ ] Capacity-aware placement

**Week 7-8: Production Readiness**
- [ ] Monitoring and metrics
- [ ] Health checks and reconciliation
- [ ] Load testing and optimization
- [ ] Operational runbooks

**Success Criteria:**
- ✅ LoRAs can be loaded/unloaded dynamically
- ✅ Orchestrator automatically places LoRAs across fleet
- ✅ Hot LoRAs are replicated across multiple workers
- ✅ System gracefully handles worker failures
- ✅ Clear metrics and observability

---

## Key Design Decisions

### 1. Metadata-Driven Discovery

**Decision:** Use the existing `/metadata` endpoint pattern to advertise LoRA state.

**Rationale:**
- Consistent with Dynamo's service discovery architecture
- No new infrastructure needed
- Kubernetes-native with EndpointSlices
- Real-time updates via watch API

### 2. LoRA-Aware Block Hashing

**Decision:** Include LoRA ID in KV block hash computation.

**Rationale:**
- Prevents cache pollution between base model and LoRA requests
- Same prompt with different LoRAs should have different cached states
- Simple to implement: just concatenate LoRA ID to token sequence

### 3. Phase-Based Rollout

**Decision:** Phase A (static) → Phase B (dynamic)

**Rationale:**
- De-risks implementation
- Provides value early (Phase A is sufficient for many use cases)
- Allows validation of core concepts before adding complexity
- Easier to test and debug

### 4. Leverage vLLM Native APIs

**Decision:** Use vLLM's `/v1/load_lora_adapter` and `/v1/unload_lora_adapter`.

**Rationale:**
- vLLM team has already solved the hard problems
- Tested in production by other users
- Reduces our maintenance burden
- Stays compatible with vLLM upgrades

### 5. Centralized Orchestration

**Decision:** Single orchestrator for Phase B (can be made HA later).

**Rationale:**
- Simplifies coordination logic
- Easier to reason about state
- Single source of truth for placement decisions
- Can add Raft/consensus later if needed

---

## Monitoring and Observability

### Metrics

```python
# Key metrics to track

# Worker metrics
lora_load_duration_seconds{lora_id, worker_id}
lora_unload_duration_seconds{lora_id, worker_id}
lora_memory_usage_bytes{lora_id, worker_id}
lora_request_count{lora_id, worker_id}
lora_load_failures_total{lora_id, worker_id}

# Orchestrator metrics
lora_placement_operations_total{operation}  # load, unload, rebalance
lora_replicas_current{lora_id}
lora_replicas_target{lora_id}
lora_usage_requests_total{lora_id}
lora_evictions_total{lora_id, reason}  # lru, capacity, ttl

# Router metrics
lora_routing_decisions_total{lora_id, outcome}  # success, not_loaded, error
lora_cache_hit_rate{lora_id}
lora_request_latency_seconds{lora_id}
```

### Logging

```python
# Structured logging format

{
    "timestamp": "2025-10-17T12:00:00Z",
    "level": "INFO",
    "component": "lora_orchestrator",
    "event": "lora_loaded",
    "lora_id": "sql-expert/v1",
    "worker_id": "worker-abc123",
    "duration_sec": 2.34,
    "memory_mb": 256
}
```

---

## Security Considerations

1. **LoRA Path Validation**: Ensure LoRA paths are within allowed directories
2. **Access Control**: Restrict management endpoints to admin users
3. **Resource Limits**: Enforce max LoRAs per worker to prevent DoS
4. **Audit Logging**: Log all LoRA load/unload operations

---

## Open Questions

1. **Multi-tenant LoRAs**: How do we handle namespace isolation?
2. **LoRA Versioning**: How do we handle multiple versions of the same LoRA?
3. **Cross-region LoRAs**: How do we sync LoRAs across regions?
4. **LoRA Authentication**: Do we need authentication for specific LoRAs?

---

## References

- [vLLM Dynamic LoRA Documentation](https://docs.vllm.ai/en/stable/features/lora.html#dynamically-serving-lora-adapters)
- [AIBrix LoRA Dynamic Loading](https://aibrix.readthedocs.io/latest/features/lora-dynamic-loading.html)
- [Dynamo Service Discovery](./Service-discovery.md)
- [PEFT Paper](https://arxiv.org/abs/2407.04656)

---

## Conclusion

This design provides a pragmatic path to LoRA support in Dynamo:

- **Phase A** enables basic LoRA serving with minimal changes, suitable for fixed workloads
- **Phase B** adds intelligent orchestration for dynamic, production environments
- Both phases leverage existing infrastructure (service discovery, KV routing)
- Design is aligned with industry best practices (vLLM, AIBrix)

The phased approach allows us to deliver value quickly while building toward a comprehensive solution.

