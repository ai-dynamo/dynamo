# Multi-LoRA Support Design for Dynamo

https://arxiv.org/pdf/2407.04656


## Architecture Overview

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DYNAMO MULTI-LORA ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   ROUTER    │───►│ LORA MANAGER │───►│       DRT       │───►│   SERVICE   │ │
│  │             │    │              │    │                 │    │  DISCOVERY  │ │
│  │ KV + LoRA   │    │ Orchestrator │    │ Distributed     │    │   (etcd)    │ │
│  │ Routing     │    │              │    │ Runtime         │    │             │ │
│  └─────────────┘    └──────────────┘    └─────────────────┘    └─────────────┘ │
│         │                   │                      │                      │     │
│         ▼                   ▼                      ▼                      ▼     │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │ Request     │    │ Policy       │    │ Backend         │    │ LoRA        │ │
│  │ Analysis    │    │ Engine       │    │ Coordination    │    │ Registry    │ │
│  │             │    │              │    │                 │    │             │ │
│  │ - Model ID  │    │ - Eager      │    │ - Load/Unload   │    │ - Available │ │
│  │ - LoRA ID   │    │ - Lazy       │    │ - Health Check  │    │ - Loaded    │ │
│  │ - KV Hash   │    │ - Weighted   │    │ - Capacity Mgmt │    │ - Metadata  │ │
│  └─────────────┘    └──────────────┘    └─────────────────┘    └─────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND WORKER PODS                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │    POD 1    │    │    POD 2    │    │    POD 3    │    │    POD N    │      │
│  │             │    │             │    │             │    │             │      │
│  │ Base Model  │    │ Base Model  │    │ Base Model  │    │ Base Model  │      │
│  │ + LoRA-A    │    │ + LoRA-B    │    │ + LoRA-C    │    │ + LoRA-X    │      │
│  │ + LoRA-D    │    │ + LoRA-E    │    │ + LoRA-F    │    │ + LoRA-Y    │      │
│  │             │    │             │    │             │    │             │      │
│  │ vLLM/SGLang │    │ vLLM/SGLang │    │ vLLM/SGLang │    │ vLLM/SGLang │      │
│  │ /TensorRT   │    │ /TensorRT   │    │ /TensorRT   │    │ /TensorRT   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                   │                   │                   │          │
│         ▼                   ▼                   ▼                   ▼         │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                    SHARED LORA STORAGE (PVC)                           │   │
│  │                                                                        │   │
│  │  /loras/                                                               │   │
│  │  ├── namespace1/                                                       │   │
│  │  │   ├── model1/                                                       │   │
│  │  │   │   ├── revision1/                                                │   │
│  │  │   │   │   ├── lora-a/                                               │   │
│  │  │   │   │   ├── lora-b/                                               │   │
│  │  │   │   │   └── lora-c/                                               │   │
│  │  │   │   └── revision2/                                                │   │
│  │  │   └── model2/                                                       │   │
│  │  └── namespace2/                                                       │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Industry Best Practices Integration

### Learnings from AIBrix LoRA Dynamic Loading

AIBrix provides a production-ready approach to LoRA management with several key insights:

1. **ModelAdapter Controller Pattern**: AIBrix uses a Kubernetes controller to manage LoRA lifecycle with clear phase transitions:
   - `Pending` → `Scheduled` → `Loading` → `Bound` → `Running`
   - Automatic retry mechanisms with exponential backoff
   - Pod switching and failover capabilities

2. **Service Discovery Innovation**: AIBrix solves the "multiple LoRAs per pod" challenge by customizing endpoints, allowing a single pod to belong to multiple services - one per LoRA adapter.

3. **Reliability Features**:
   - Up to 5 retry attempts per pod with exponential backoff
   - Automatic pod switching if loading fails
   - Pod health validation before scheduling
   - Connection error handling with graceful recovery

### vLLM Dynamic LoRA Capabilities

vLLM provides runtime LoRA management through:

1. **Runtime Environment Flag**: `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`
2. **API Endpoints**:
   - `/v1/load_lora_adapter` - Load LoRA at runtime
   - `/v1/unload_lora_adapter` - Unload LoRA at runtime
   - `/v1/list_lora_adapters` - List loaded LoRAs
3. **Configuration Parameters**:
   - `--enable-lora` - Enable LoRA support
   - `--max-loras` - Maximum concurrent LoRAs
   - `--max-lora-rank` - Maximum LoRA rank
   - `--lora-extra-vocab-size` - Additional vocabulary size

### NVIDIA NIM PEFT Architecture

NVIDIA NIM provides enterprise-grade LoRA serving with:

1. **Adapter Store Pattern**: Centralized storage for LoRA adapters
2. **Dynamic Multi-LoRA Inference**: Simultaneous requests with different LoRAs
3. **Framework Compatibility**: Support for NeMo and Hugging Face trained adapters
4. **Production Features**: Metrics, monitoring, and enterprise security

### Dynamo Integration Strategy

Our design incorporates these proven patterns:

1. **AIBrix-inspired Controller**: Kubernetes-native LoRA lifecycle management
2. **vLLM Runtime Integration**: Direct API integration for dynamic loading
3. **NIM-style Adapter Store**: Shared storage with organized hierarchy
4. **Enhanced Service Discovery**: Multi-LoRA per pod support

## Phase 1: POC Implementation

### 1. Environment Configuration

**Contract: `DYNAMO_LORA_PATH` Environment Variable**
```bash
# Points to shared PVC mount path
export DYNAMO_LORA_PATH="/shared/loras"
```

**Directory Structure:**
```
/shared/loras/
├── namespace1/
│   ├── model1/
│   │   ├── revision1/
│   │   │   ├── lora-adapter-1/
│   │   │   │   ├── adapter_config.json
│   │   │   │   ├── adapter_model.safetensors
│   │   │   │   └── metadata.json
│   │   │   └── lora-adapter-2/
│   │   └── revision2/
│   └── model2/
└── namespace2/
```

### 2. Backend Wrapper Extensions

#### 2.1 LoRA Management Interface

```python
# components/src/dynamo/common/lora_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class LoRAStatus(Enum):
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    FAILED = "failed"

@dataclass
class LoRAInfo:
    lora_id: str
    lora_path: str
    base_model: str
    namespace: str
    revision: str
    status: LoRAStatus
    memory_usage: Optional[int] = None
    load_time: Optional[float] = None
    last_used: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class LoRAManagerInterface(ABC):
    """Interface that all backend LoRA managers must implement"""

    @abstractmethod
    async def load_lora(self, lora_id: str, lora_path: str, **kwargs) -> bool:
        """Load a LoRA adapter"""
        pass

    @abstractmethod
    async def unload_lora(self, lora_id: str) -> bool:
        """Unload a LoRA adapter"""
        pass

    @abstractmethod
    async def list_loras(self) -> List[LoRAInfo]:
        """List all loaded LoRA adapters"""
        pass

    @abstractmethod
    async def get_lora_info(self, lora_id: str) -> Optional[LoRAInfo]:
        """Get information about a specific LoRA"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check for LoRA management system"""
        pass
```

#### 2.2 vLLM LoRA Manager Implementation

```python
# components/src/dynamo/vllm/lora_manager.py
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from vllm.lora.request import LoRARequest
from dynamo.common.lora_interface import LoRAManagerInterface, LoRAInfo, LoRAStatus

logger = logging.getLogger(__name__)

class VLLMLoRAManager(LoRAManagerInterface):
    """vLLM-specific LoRA management implementation"""

    def __init__(self, engine, max_loras: int = 4):
        self.engine = engine
        self.max_loras = max_loras
        self.loaded_loras: Dict[str, LoRAInfo] = {}
        self.lora_requests: Dict[str, LoRARequest] = {}
        self._lock = asyncio.Lock()

    async def load_lora(self, lora_id: str, lora_path: str, **kwargs) -> bool:
        """Load LoRA adapter into vLLM engine"""
        async with self._lock:
            try:
                if lora_id in self.loaded_loras:
                    logger.warning(f"LoRA {lora_id} already loaded")
                    return True

                if len(self.loaded_loras) >= self.max_loras:
                    # Implement LRU eviction
                    await self._evict_lru_lora()

                # Update status to loading
                lora_info = LoRAInfo(
                    lora_id=lora_id,
                    lora_path=lora_path,
                    base_model=kwargs.get('base_model', ''),
                    namespace=kwargs.get('namespace', ''),
                    revision=kwargs.get('revision', ''),
                    status=LoRAStatus.LOADING
                )
                self.loaded_loras[lora_id] = lora_info

                start_time = time.time()

                # Create LoRA request for vLLM
                lora_request = LoRARequest(
                    lora_name=lora_id,
                    lora_int_id=hash(lora_id) % 1000000,  # Generate unique int ID
                    lora_path=lora_path
                )

                # Load into vLLM engine (this is engine-specific)
                # For vLLM v1, we need to add the LoRA to the engine
                await self._load_lora_into_engine(lora_request)

                # Update status and timing
                load_time = time.time() - start_time
                lora_info.status = LoRAStatus.LOADED
                lora_info.load_time = load_time
                lora_info.last_used = time.time()

                self.lora_requests[lora_id] = lora_request

                logger.info(f"Successfully loaded LoRA {lora_id} in {load_time:.2f}s")
                return True

            except Exception as e:
                logger.error(f"Failed to load LoRA {lora_id}: {e}")
                if lora_id in self.loaded_loras:
                    self.loaded_loras[lora_id].status = LoRAStatus.FAILED
                return False

    async def unload_lora(self, lora_id: str) -> bool:
        """Unload LoRA adapter from vLLM engine"""
        async with self._lock:
            try:
                if lora_id not in self.loaded_loras:
                    logger.warning(f"LoRA {lora_id} not loaded")
                    return True

                self.loaded_loras[lora_id].status = LoRAStatus.UNLOADING

                # Remove from vLLM engine
                if lora_id in self.lora_requests:
                    await self._unload_lora_from_engine(self.lora_requests[lora_id])
                    del self.lora_requests[lora_id]

                del self.loaded_loras[lora_id]

                logger.info(f"Successfully unloaded LoRA {lora_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to unload LoRA {lora_id}: {e}")
                return False

    async def list_loras(self) -> List[LoRAInfo]:
        """List all loaded LoRA adapters"""
        return list(self.loaded_loras.values())

    async def get_lora_info(self, lora_id: str) -> Optional[LoRAInfo]:
        """Get information about a specific LoRA"""
        return self.loaded_loras.get(lora_id)

    async def health_check(self) -> Dict[str, Any]:
        """Health check for LoRA management system"""
        return {
            "loaded_loras": len(self.loaded_loras),
            "max_loras": self.max_loras,
            "available_slots": self.max_loras - len(self.loaded_loras),
            "lora_ids": list(self.loaded_loras.keys())
        }

    async def _evict_lru_lora(self):
        """Evict least recently used LoRA"""
        if not self.loaded_loras:
            return

        # Find LRU LoRA
        lru_lora_id = min(
            self.loaded_loras.keys(),
            key=lambda x: self.loaded_loras[x].last_used or 0
        )

        logger.info(f"Evicting LRU LoRA: {lru_lora_id}")
        await self.unload_lora(lru_lora_id)

    async def _load_lora_into_engine(self, lora_request: LoRARequest):
        """Engine-specific LoRA loading logic"""
        # This will depend on vLLM version and API
        # For vLLM v1, this might involve calling engine methods
        pass

    async def _unload_lora_from_engine(self, lora_request: LoRARequest):
        """Engine-specific LoRA unloading logic"""
        # This will depend on vLLM version and API
        pass

    def get_lora_request(self, lora_id: str) -> Optional[LoRARequest]:
        """Get vLLM LoRA request object for inference"""
        if lora_id in self.loaded_loras and self.loaded_loras[lora_id].status == LoRAStatus.LOADED:
            self.loaded_loras[lora_id].last_used = time.time()
            return self.lora_requests.get(lora_id)
        return None
```

#### 2.3 Enhanced Backend Handlers

```python
# components/src/dynamo/vllm/lora_handlers.py
import logging
from typing import AsyncGenerator, Dict, Any
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams

from .handlers import BaseWorkerHandler
from .lora_manager import VLLMLoRAManager
from dynamo.common.lora_interface import LoRAInfo

logger = logging.getLogger(__name__)

class LoRAWorkerHandler(BaseWorkerHandler):
    """Enhanced worker handler with LoRA support"""

    def __init__(self, runtime, component, engine, default_sampling_params, **kwargs):
        super().__init__(runtime, component, engine, default_sampling_params)
        self.lora_manager = VLLMLoRAManager(engine)

    async def generate(self, request, context) -> AsyncGenerator[dict, None]:
        """Generate with LoRA support"""
        request_id = str(uuid.uuid4().hex)
        logger.debug(f"New LoRA Request ID: {request_id}")

        # Extract LoRA information from request
        lora_id = request.get("lora_id")
        lora_request = None

        if lora_id:
            lora_request = self.lora_manager.get_lora_request(lora_id)
            if not lora_request:
                yield {"error": f"LoRA {lora_id} not loaded or failed to load"}
                return

        prompt = TokensPrompt(prompt_token_ids=request["token_ids"])
        sampling_params = SamplingParams(**self.default_sampling_params)

        # Apply sampling parameters from request
        for key, value in request.get("sampling_options", {}).items():
            if value is not None and hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        # Set LoRA request in sampling params if available
        if lora_request:
            sampling_params.lora_request = lora_request

        dp_rank = request.get("dp_rank", None)

        async with self._abort_monitor(context, request_id):
            try:
                async for tok in self.generate_tokens(
                    prompt, sampling_params, request_id, data_parallel_rank=dp_rank
                ):
                    yield tok
            except Exception as e:
                logger.error(f"Generation error: {e}")
                yield {"error": str(e)}

    async def load_lora(self, request=None):
        """Load LoRA endpoint"""
        try:
            lora_id = request["lora_id"]
            lora_path = request["lora_path"]

            success = await self.lora_manager.load_lora(
                lora_id=lora_id,
                lora_path=lora_path,
                **request.get("metadata", {})
            )

            if success:
                yield {"status": "success", "lora_id": lora_id, "message": "LoRA loaded successfully"}
            else:
                yield {"status": "error", "lora_id": lora_id, "message": "Failed to load LoRA"}

        except Exception as e:
            yield {"status": "error", "message": str(e)}

    async def unload_lora(self, request=None):
        """Unload LoRA endpoint"""
        try:
            lora_id = request["lora_id"]

            success = await self.lora_manager.unload_lora(lora_id)

            if success:
                yield {"status": "success", "lora_id": lora_id, "message": "LoRA unloaded successfully"}
            else:
                yield {"status": "error", "lora_id": lora_id, "message": "Failed to unload LoRA"}

        except Exception as e:
            yield {"status": "error", "message": str(e)}

    async def list_loras(self, request=None):
        """List LoRAs endpoint"""
        try:
            loras = await self.lora_manager.list_loras()
            lora_list = [
                {
                    "lora_id": lora.lora_id,
                    "status": lora.status.value,
                    "base_model": lora.base_model,
                    "namespace": lora.namespace,
                    "revision": lora.revision,
                    "memory_usage": lora.memory_usage,
                    "last_used": lora.last_used
                }
                for lora in loras
            ]
            yield {"status": "success", "loras": lora_list}

        except Exception as e:
            yield {"status": "error", "message": str(e)}
```

### 3. LoRA Discovery System

```python
# components/src/dynamo/common/lora_discovery.py
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DiscoveredLoRA:
    lora_id: str
    namespace: str
    model: str
    revision: str
    path: str
    metadata: Dict

class LoRADiscovery:
    """Discovers available LoRAs from the shared storage"""

    def __init__(self, lora_root_path: str):
        self.lora_root_path = Path(lora_root_path)
        self._discovered_loras: Dict[str, DiscoveredLoRA] = {}

    def discover_loras(self, namespace_filter: Optional[str] = None) -> List[DiscoveredLoRA]:
        """Discover all available LoRAs in the storage"""
        discovered = []

        if not self.lora_root_path.exists():
            logger.warning(f"LoRA root path does not exist: {self.lora_root_path}")
            return discovered

        # Walk through namespace/model/revision/lora structure
        for namespace_dir in self.lora_root_path.iterdir():
            if not namespace_dir.is_dir():
                continue

            namespace = namespace_dir.name
            if namespace_filter and namespace != namespace_filter:
                continue

            for model_dir in namespace_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                model = model_dir.name

                for revision_dir in model_dir.iterdir():
                    if not revision_dir.is_dir():
                        continue

                    revision = revision_dir.name

                    for lora_dir in revision_dir.iterdir():
                        if not lora_dir.is_dir():
                            continue

                        lora_name = lora_dir.name
                        lora_id = f"{namespace}/{model}/{revision}/{lora_name}"

                        # Check for required files
                        if self._validate_lora_directory(lora_dir):
                            metadata = self._load_lora_metadata(lora_dir)

                            lora = DiscoveredLoRA(
                                lora_id=lora_id,
                                namespace=namespace,
                                model=model,
                                revision=revision,
                                path=str(lora_dir),
                                metadata=metadata
                            )

                            discovered.append(lora)
                            self._discovered_loras[lora_id] = lora

        logger.info(f"Discovered {len(discovered)} LoRAs")
        return discovered

    def _validate_lora_directory(self, lora_dir: Path) -> bool:
        """Validate that LoRA directory contains required files"""
        required_files = ["adapter_config.json"]
        adapter_files = ["adapter_model.safetensors", "adapter_model.bin"]

        # Check for config file
        if not (lora_dir / "adapter_config.json").exists():
            return False

        # Check for at least one adapter file
        if not any((lora_dir / f).exists() for f in adapter_files):
            return False

        return True

    def _load_lora_metadata(self, lora_dir: Path) -> Dict:
        """Load LoRA metadata from directory"""
        metadata = {}

        # Load adapter config
        config_path = lora_dir / "adapter_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    metadata["adapter_config"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load adapter config from {config_path}: {e}")

        # Load custom metadata if available
        metadata_path = lora_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    custom_metadata = json.load(f)
                    metadata.update(custom_metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_path}: {e}")

        return metadata

    def get_lora(self, lora_id: str) -> Optional[DiscoveredLoRA]:
        """Get discovered LoRA by ID"""
        return self._discovered_loras.get(lora_id)

    def get_loras_for_model(self, namespace: str, model: str, revision: str = None) -> List[DiscoveredLoRA]:
        """Get all LoRAs for a specific model"""
        matching = []
        for lora in self._discovered_loras.values():
            if lora.namespace == namespace and lora.model == model:
                if revision is None or lora.revision == revision:
                    matching.append(lora)
        return matching
```

### 4. Dynamo Service Discovery Integration

Based on the [Dynamo Service Discovery Enhancement](https://raw.githubusercontent.com/ai-dynamo/enhancements/981844c8e1ca48552c0cc6a9ff34bb86b61c23c2/deps/etcd-k8s.md), we integrate LoRA management with the new Kubernetes-native service discovery system.

#### 4.1 LoRA Metadata in Service Discovery

```python
# components/src/dynamo/vllm/service_discovery_integration.py
import logging
from typing import Dict, List, Any, Optional
from dynamo.common.lora_interface import LoRAInfo, LoRAStatus

logger = logging.getLogger(__name__)

class LoRAServiceDiscoveryIntegration:
    """Integrates LoRA information with Dynamo's new service discovery system"""

    def __init__(self, instance_handle, lora_manager):
        self.instance_handle = instance_handle
        self.lora_manager = lora_manager
        self._last_metadata = {}

    async def update_lora_metadata(self):
        """Update service discovery metadata with current LoRA state"""
        try:
            # Get current LoRA state from manager
            loaded_loras = await self.lora_manager.list_loras()

            # Build LoRA metadata for service discovery
            lora_metadata = {
                "loaded_loras": {
                    lora.lora_id: {
                        "status": lora.status.value,
                        "base_model": lora.base_model,
                        "namespace": lora.namespace,
                        "revision": lora.revision,
                        "memory_usage": lora.memory_usage,
                        "last_used": lora.last_used,
                        "load_time": lora.load_time
                    }
                    for lora in loaded_loras
                },
                "lora_capacity": {
                    "max_loras": self.lora_manager.max_loras,
                    "loaded_count": len(loaded_loras),
                    "available_slots": self.lora_manager.max_loras - len(loaded_loras)
                },
                "lora_endpoints": {
                    "load_lora": "/v1/load_lora_adapter",
                    "unload_lora": "/v1/unload_lora_adapter",
                    "list_loras": "/v1/list_lora_adapters"
                }
            }

            # Get existing metadata and merge with LoRA info
            current_metadata = self._get_base_metadata()
            current_metadata["lora"] = lora_metadata

            # Only update if metadata changed
            if current_metadata != self._last_metadata:
                self.instance_handle.set_metadata(current_metadata)
                self._last_metadata = current_metadata.copy()
                logger.debug(f"Updated service discovery with LoRA metadata: {len(loaded_loras)} LoRAs")

        except Exception as e:
            logger.error(f"Failed to update LoRA metadata in service discovery: {e}")

    def _get_base_metadata(self) -> Dict[str, Any]:
        """Get base metadata for the instance"""
        return {
            "model": {
                "name": "Qwen3-32B",  # This would come from config
                "type": "Completions",
                "runtime_config": {
                    "total_kv_blocks": 24064,
                    "max_num_seqs": 256,
                    "max_num_batched_tokens": 2048
                }
            },
            "transport": {
                "type": "http",
                "endpoint": "http://0.0.0.0:8000"
            },
            "capabilities": {
                "lora_support": True,
                "dynamic_loading": True
            }
        }

    async def start_periodic_updates(self, interval_seconds: int = 30):
        """Start periodic metadata updates"""
        import asyncio

        while True:
            try:
                await self.update_lora_metadata()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in periodic LoRA metadata update: {e}")
                await asyncio.sleep(interval_seconds)
```

#### 4.2 Enhanced Backend Registration

```python
# components/src/dynamo/vllm/enhanced_main.py
# Enhanced version of main.py with LoRA service discovery integration

async def init_with_lora_support(runtime: DistributedRuntime, config: Config):
    """Initialize worker with LoRA support and service discovery integration"""

    # Register instance with service discovery (following new pattern)
    service_registry = runtime.service_registry()
    instance_handle = service_registry.register_instance(
        namespace=config.namespace,
        component=config.component
    )

    # Set up component and endpoints
    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    # Initialize LoRA-enabled handler
    engine_client, vllm_config, default_sampling_params = setup_vllm_engine(config)

    handler = LoRAWorkerHandler(
        runtime, component, engine_client, default_sampling_params
    )

    # Set up LoRA service discovery integration
    lora_discovery = LoRAServiceDiscoveryIntegration(
        instance_handle=instance_handle,
        lora_manager=handler.lora_manager
    )

    # Set initial metadata
    await lora_discovery.update_lora_metadata()

    # Mark instance as ready
    instance_handle.set_ready("ready")

    # Start periodic metadata updates
    asyncio.create_task(lora_discovery.start_periodic_updates())

    # Add LoRA management endpoints
    load_lora_endpoint = component.endpoint("load_lora")
    unload_lora_endpoint = component.endpoint("unload_lora")
    list_lora_endpoint = component.endpoint("list_loras")

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=config.migration_limit <= 0,
                metrics_labels=[("model", config.served_model_name or config.model)],
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks,
                metrics_labels=[("model", config.served_model_name or config.model)],
            ),
            # New LoRA management endpoints
            load_lora_endpoint.serve_endpoint(
                handler.load_lora,
                metrics_labels=[("model", config.served_model_name or config.model)],
            ),
            unload_lora_endpoint.serve_endpoint(
                handler.unload_lora,
                metrics_labels=[("model", config.served_model_name or config.model)],
            ),
            list_lora_endpoint.serve_endpoint(
                handler.list_loras,
                metrics_labels=[("model", config.served_model_name or config.model)],
            ),
        )
    finally:
        handler.cleanup()
```

#### 4.3 LoRA-Aware Service Discovery Client

```python
# components/src/dynamo/router/lora_service_discovery.py
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LoRAInstanceInfo:
    """Information about a LoRA-capable instance"""
    instance_id: str
    address: str
    loaded_loras: Dict[str, Dict[str, Any]]
    lora_capacity: Dict[str, int]
    base_model: str
    transport_info: Dict[str, Any]

class LoRAServiceDiscoveryClient:
    """Client for discovering LoRA-capable instances using new service discovery"""

    def __init__(self, service_discovery):
        self.service_discovery = service_discovery
        self.lora_instances: Dict[str, LoRAInstanceInfo] = {}
        self.lora_to_instances: Dict[str, Set[str]] = {}  # lora_id -> {instance_ids}

    async def discover_lora_instances(self, namespace: str, component: str) -> List[LoRAInstanceInfo]:
        """Discover all LoRA-capable instances"""
        instances = self.service_discovery.list_instances(namespace, component)
        lora_instances = []

        for instance in instances:
            try:
                # Fetch metadata from instance
                metadata = await instance.metadata()

                # Check if instance supports LoRA
                if not metadata.get("capabilities", {}).get("lora_support", False):
                    continue

                lora_info = metadata.get("lora", {})

                instance_info = LoRAInstanceInfo(
                    instance_id=instance.instance_id,
                    address=instance.address,
                    loaded_loras=lora_info.get("loaded_loras", {}),
                    lora_capacity=lora_info.get("lora_capacity", {}),
                    base_model=metadata.get("model", {}).get("name", ""),
                    transport_info=metadata.get("transport", {})
                )

                lora_instances.append(instance_info)
                self.lora_instances[instance.instance_id] = instance_info

                # Update lora -> instances mapping
                for lora_id in instance_info.loaded_loras.keys():
                    if lora_id not in self.lora_to_instances:
                        self.lora_to_instances[lora_id] = set()
                    self.lora_to_instances[lora_id].add(instance.instance_id)

            except Exception as e:
                logger.warning(f"Failed to get metadata from instance {instance.instance_id}: {e}")

        logger.info(f"Discovered {len(lora_instances)} LoRA-capable instances")
        return lora_instances

    async def watch_lora_instances(self, namespace: str, component: str):
        """Watch for changes in LoRA instances"""
        event_stream = self.service_discovery.watch(namespace, component)

        async for event in event_stream:
            try:
                if hasattr(event, 'instance_added'):
                    await self._handle_instance_added(event.instance_added)
                elif hasattr(event, 'instance_removed'):
                    await self._handle_instance_removed(event.instance_removed)
                elif hasattr(event, 'instance_updated'):
                    await self._handle_instance_updated(event.instance_updated)
            except Exception as e:
                logger.error(f"Error handling service discovery event: {e}")

    async def _handle_instance_added(self, instance):
        """Handle new LoRA instance"""
        try:
            metadata = await instance.metadata()

            if metadata.get("capabilities", {}).get("lora_support", False):
                lora_info = metadata.get("lora", {})

                instance_info = LoRAInstanceInfo(
                    instance_id=instance.instance_id,
                    address=instance.address,
                    loaded_loras=lora_info.get("loaded_loras", {}),
                    lora_capacity=lora_info.get("lora_capacity", {}),
                    base_model=metadata.get("model", {}).get("name", ""),
                    transport_info=metadata.get("transport", {})
                )

                self.lora_instances[instance.instance_id] = instance_info
                logger.info(f"Added LoRA instance {instance.instance_id}")

        except Exception as e:
            logger.error(f"Error handling instance added: {e}")

    async def _handle_instance_removed(self, instance):
        """Handle removed LoRA instance"""
        instance_id = instance.instance_id

        if instance_id in self.lora_instances:
            # Clean up lora -> instances mapping
            instance_info = self.lora_instances[instance_id]
            for lora_id in instance_info.loaded_loras.keys():
                if lora_id in self.lora_to_instances:
                    self.lora_to_instances[lora_id].discard(instance_id)
                    if not self.lora_to_instances[lora_id]:
                        del self.lora_to_instances[lora_id]

            del self.lora_instances[instance_id]
            logger.info(f"Removed LoRA instance {instance_id}")

    async def _handle_instance_updated(self, instance):
        """Handle updated LoRA instance metadata"""
        # For now, treat as remove + add
        await self._handle_instance_removed(instance)
        await self._handle_instance_added(instance)

    def find_instances_with_lora(self, lora_id: str) -> List[LoRAInstanceInfo]:
        """Find all instances that have a specific LoRA loaded"""
        instance_ids = self.lora_to_instances.get(lora_id, set())
        return [self.lora_instances[iid] for iid in instance_ids if iid in self.lora_instances]

    def find_instances_with_capacity(self, min_slots: int = 1) -> List[LoRAInstanceInfo]:
        """Find instances with available LoRA capacity"""
        available_instances = []

        for instance_info in self.lora_instances.values():
            capacity = instance_info.lora_capacity
            available_slots = capacity.get("available_slots", 0)

            if available_slots >= min_slots:
                available_instances.append(instance_info)

        return available_instances

    def get_lora_distribution(self) -> Dict[str, List[str]]:
        """Get current distribution of LoRAs across instances"""
        return {
            lora_id: list(instance_ids)
            for lora_id, instance_ids in self.lora_to_instances.items()
        }
```

### 5. Enhanced Router with LoRA Support

```python
# components/src/dynamo/router/lora_router.py
import logging
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from dynamo.runtime import DistributedRuntime
from dynamo.common.lora_interface import LoRAInfo

logger = logging.getLogger(__name__)

@dataclass
class LoRARoutingDecision:
    instance_id: int
    lora_id: str
    needs_loading: bool
    confidence: float

class LoRARouter:
    """Router with LoRA-aware routing capabilities"""

    def __init__(self, runtime: DistributedRuntime, namespace: str):
        self.runtime = runtime
        self.namespace = namespace
        self.instance_lora_map: Dict[int, List[str]] = {}  # instance_id -> [lora_ids]
        self.lora_instance_map: Dict[str, List[int]] = {}  # lora_id -> [instance_ids]

    async def route_request(self, request: Dict[str, Any]) -> LoRARoutingDecision:
        """Route request with LoRA awareness"""
        lora_id = request.get("lora_id")

        if not lora_id:
            # No LoRA requested, use standard routing
            return await self._route_base_model(request)

        # LoRA requested, find best instance
        return await self._route_lora_request(request, lora_id)

    async def _route_lora_request(self, request: Dict[str, Any], lora_id: str) -> LoRARoutingDecision:
        """Route request to instance with LoRA loaded or best candidate for loading"""

        # Check if LoRA is already loaded somewhere
        if lora_id in self.lora_instance_map:
            loaded_instances = self.lora_instance_map[lora_id]
            if loaded_instances:
                # Use KV cache routing logic to pick best instance
                best_instance = await self._select_best_instance_with_kv(
                    request, loaded_instances
                )
                return LoRARoutingDecision(
                    instance_id=best_instance,
                    lora_id=lora_id,
                    needs_loading=False,
                    confidence=1.0
                )

        # LoRA not loaded, find best instance to load it
        best_instance = await self._select_best_instance_for_loading(lora_id)
        return LoRARoutingDecision(
            instance_id=best_instance,
            lora_id=lora_id,
            needs_loading=True,
            confidence=0.8
        )

    async def _route_base_model(self, request: Dict[str, Any]) -> LoRARoutingDecision:
        """Route to base model without LoRA"""
        # Use existing KV-aware routing
        instances = await self._get_available_instances()
        best_instance = await self._select_best_instance_with_kv(request, instances)

        return LoRARoutingDecision(
            instance_id=best_instance,
            lora_id=None,
            needs_loading=False,
            confidence=1.0
        )

    async def _select_best_instance_with_kv(self, request: Dict[str, Any], instances: List[int]) -> int:
        """Select best instance considering KV cache"""
        # Implement KV-aware selection logic
        # This would integrate with existing KV router
        token_ids = request.get("token_ids", [])
        kv_hash = self._compute_kv_hash(token_ids)

        # For now, simple round-robin among available instances
        # TODO: Integrate with actual KV router logic
        return instances[0] if instances else None

    async def _select_best_instance_for_loading(self, lora_id: str) -> int:
        """Select best instance to load a new LoRA"""
        instances = await self._get_available_instances()

        # Find instance with most available LoRA slots
        best_instance = None
        min_loras = float('inf')

        for instance_id in instances:
            current_loras = len(self.instance_lora_map.get(instance_id, []))
            if current_loras < min_loras:
                min_loras = current_loras
                best_instance = instance_id

        return best_instance

    async def _get_available_instances(self) -> List[int]:
        """Get list of available backend instances"""
        # Query service discovery for available instances
        # This would integrate with existing service discovery
        return [1, 2, 3]  # Placeholder

    def _compute_kv_hash(self, token_ids: List[int]) -> str:
        """Compute KV cache hash including LoRA information"""
        # This should integrate with existing KV hash computation
        # and include LoRA ID to avoid cross-contamination
        token_str = ",".join(map(str, token_ids))
        return hashlib.md5(token_str.encode()).hexdigest()

    async def update_instance_lora_state(self, instance_id: int, loaded_loras: List[str]):
        """Update the router's view of which LoRAs are loaded where"""
        # Update instance -> loras mapping
        self.instance_lora_map[instance_id] = loaded_loras

        # Update lora -> instances mapping
        for lora_id in loaded_loras:
            if lora_id not in self.lora_instance_map:
                self.lora_instance_map[lora_id] = []
            if instance_id not in self.lora_instance_map[lora_id]:
                self.lora_instance_map[lora_id].append(instance_id)

        # Clean up removed LoRAs
        for lora_id, instances in list(self.lora_instance_map.items()):
            if instance_id in instances and lora_id not in loaded_loras:
                instances.remove(instance_id)
                if not instances:
                    del self.lora_instance_map[lora_id]
```

### 5. Service Discovery Integration

```python
# components/src/dynamo/common/lora_service_discovery.py
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class LoRAServiceDiscovery:
    """Extends service discovery with LoRA information"""

    def __init__(self, etcd_client):
        self.etcd_client = etcd_client
        self.lora_info_prefix = "v1/lora_info"

    async def register_lora_info(self, instance_id: int, lora_info: Dict[str, Any]):
        """Register LoRA information for an instance"""
        key = f"{self.lora_info_prefix}/{instance_id}"
        value = json.dumps(lora_info)

        try:
            await self.etcd_client.put(key, value)
            logger.debug(f"Registered LoRA info for instance {instance_id}")
        except Exception as e:
            logger.error(f"Failed to register LoRA info for instance {instance_id}: {e}")

    async def get_lora_info(self, instance_id: int) -> Optional[Dict[str, Any]]:
        """Get LoRA information for an instance"""
        key = f"{self.lora_info_prefix}/{instance_id}"

        try:
            result = await self.etcd_client.get(key)
            if result:
                return json.loads(result)
        except Exception as e:
            logger.error(f"Failed to get LoRA info for instance {instance_id}: {e}")

        return None

    async def get_all_lora_info(self) -> Dict[int, Dict[str, Any]]:
        """Get LoRA information for all instances"""
        try:
            results = await self.etcd_client.get_prefix(self.lora_info_prefix)
            lora_info = {}

            for key, value in results.items():
                # Extract instance_id from key
                instance_id = int(key.split('/')[-1])
                lora_info[instance_id] = json.loads(value)

            return lora_info
        except Exception as e:
            logger.error(f"Failed to get all LoRA info: {e}")
            return {}

    async def find_instances_with_lora(self, lora_id: str) -> List[int]:
        """Find all instances that have a specific LoRA loaded"""
        all_info = await self.get_all_lora_info()
        instances = []

        for instance_id, info in all_info.items():
            loaded_loras = info.get("loaded_loras", [])
            if lora_id in loaded_loras:
                instances.append(instance_id)

        return instances

    async def find_instances_with_capacity(self, min_slots: int = 1) -> List[int]:
        """Find instances with available LoRA slots"""
        all_info = await self.get_all_lora_info()
        instances = []

        for instance_id, info in all_info.items():
            available_slots = info.get("available_lora_slots", 0)
            if available_slots >= min_slots:
                instances.append(instance_id)

        return instances
```

## Phase 2: Full Management System

### 1. LoRA Manager Subsystem

```python
# components/src/dynamo/lora_manager/manager.py
import asyncio
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

from dynamo.runtime import DistributedRuntime
from dynamo.common.lora_discovery import LoRADiscovery
from dynamo.common.lora_service_discovery import LoRAServiceDiscovery

logger = logging.getLogger(__name__)

class LoadingPolicy(Enum):
    EAGER_UNWEIGHTED = "eager_unweighted"
    EAGER_WEIGHTED = "eager_weighted"
    LAZY_TTL = "lazy_ttl"
    LAZY_STICKY = "lazy_sticky"

class LoRAManager:
    """Central LoRA management subsystem"""

    def __init__(self, runtime: DistributedRuntime, config: Dict[str, Any]):
        self.runtime = runtime
        self.config = config
        self.discovery = LoRADiscovery(config["lora_path"])
        self.service_discovery = LoRAServiceDiscovery(runtime.etcd_client())
        self.loading_policy = LoadingPolicy(config.get("loading_policy", "lazy_ttl"))

        # State tracking
        self.lora_assignments: Dict[str, List[int]] = {}  # lora_id -> [instance_ids]
        self.instance_assignments: Dict[int, List[str]] = {}  # instance_id -> [lora_ids]
        self.lora_usage_stats: Dict[str, Dict[str, Any]] = {}

        # Policy parameters
        self.max_loras_per_instance = config.get("max_loras_per_instance", 4)
        self.ttl_seconds = config.get("lora_ttl_seconds", 3600)
        self.rebalance_interval = config.get("rebalance_interval_seconds", 300)

    async def start(self):
        """Start the LoRA manager"""
        logger.info("Starting LoRA Manager")

        # Discover available LoRAs
        await self._discover_loras()

        # Start background tasks
        asyncio.create_task(self._periodic_rebalance())
        asyncio.create_task(self._periodic_health_check())

        # Apply initial loading policy
        await self._apply_loading_policy()

    async def request_lora_loading(self, lora_id: str, priority: int = 0) -> bool:
        """Request loading of a specific LoRA"""
        logger.info(f"Received request to load LoRA: {lora_id}")

        # Check if already loaded
        if lora_id in self.lora_assignments and self.lora_assignments[lora_id]:
            logger.info(f"LoRA {lora_id} already loaded")
            return True

        # Find best instance to load LoRA
        target_instance = await self._select_target_instance(lora_id)
        if not target_instance:
            logger.error(f"No available instance to load LoRA {lora_id}")
            return False

        # Send load command to instance
        success = await self._load_lora_on_instance(lora_id, target_instance)
        if success:
            await self._update_assignments(lora_id, target_instance, "loaded")

        return success

    async def request_lora_unloading(self, lora_id: str) -> bool:
        """Request unloading of a specific LoRA"""
        logger.info(f"Received request to unload LoRA: {lora_id}")

        if lora_id not in self.lora_assignments:
            logger.info(f"LoRA {lora_id} not loaded")
            return True

        # Unload from all instances
        success = True
        for instance_id in self.lora_assignments[lora_id]:
            instance_success = await self._unload_lora_from_instance(lora_id, instance_id)
            if instance_success:
                await self._update_assignments(lora_id, instance_id, "unloaded")
            success = success and instance_success

        return success

    async def _discover_loras(self):
        """Discover available LoRAs"""
        discovered = self.discovery.discover_loras()
        logger.info(f"Discovered {len(discovered)} available LoRAs")

        for lora in discovered:
            self.lora_usage_stats[lora.lora_id] = {
                "requests": 0,
                "last_used": None,
                "load_time": None,
                "priority": lora.metadata.get("priority", 0)
            }

    async def _apply_loading_policy(self):
        """Apply the configured loading policy"""
        if self.loading_policy == LoadingPolicy.EAGER_UNWEIGHTED:
            await self._eager_unweighted_loading()
        elif self.loading_policy == LoadingPolicy.EAGER_WEIGHTED:
            await self._eager_weighted_loading()
        # Lazy policies are applied on-demand

    async def _eager_unweighted_loading(self):
        """Load LoRAs evenly across instances"""
        available_loras = list(self.lora_usage_stats.keys())
        available_instances = await self._get_available_instances()

        if not available_instances:
            logger.warning("No available instances for eager loading")
            return

        loras_per_instance = len(available_loras) // len(available_instances)

        for i, lora_id in enumerate(available_loras):
            instance_idx = i % len(available_instances)
            target_instance = available_instances[instance_idx]

            # Check if instance has capacity
            current_loras = len(self.instance_assignments.get(target_instance, []))
            if current_loras < self.max_loras_per_instance:
                await self._load_lora_on_instance(lora_id, target_instance)
                await self._update_assignments(lora_id, target_instance, "loaded")

    async def _eager_weighted_loading(self):
        """Load LoRAs based on priority/weight"""
        # Sort LoRAs by priority
        sorted_loras = sorted(
            self.lora_usage_stats.items(),
            key=lambda x: x[1]["priority"],
            reverse=True
        )

        available_instances = await self._get_available_instances()

        for lora_id, stats in sorted_loras:
            # Find instance with most capacity
            target_instance = None
            max_capacity = -1

            for instance_id in available_instances:
                current_loras = len(self.instance_assignments.get(instance_id, []))
                capacity = self.max_loras_per_instance - current_loras

                if capacity > max_capacity:
                    max_capacity = capacity
                    target_instance = instance_id

            if target_instance and max_capacity > 0:
                await self._load_lora_on_instance(lora_id, target_instance)
                await self._update_assignments(lora_id, target_instance, "loaded")

    async def _select_target_instance(self, lora_id: str) -> Optional[int]:
        """Select best instance to load a LoRA"""
        available_instances = await self._get_available_instances()

        # Find instance with most available capacity
        best_instance = None
        max_capacity = -1

        for instance_id in available_instances:
            current_loras = len(self.instance_assignments.get(instance_id, []))
            capacity = self.max_loras_per_instance - current_loras

            if capacity > max_capacity:
                max_capacity = capacity
                best_instance = instance_id

        return best_instance if max_capacity > 0 else None

    async def _load_lora_on_instance(self, lora_id: str, instance_id: int) -> bool:
        """Send load command to specific instance"""
        try:
            # Get LoRA path
            lora = self.discovery.get_lora(lora_id)
            if not lora:
                logger.error(f"LoRA {lora_id} not found in discovery")
                return False

            # Send load request to instance
            client = await self._get_instance_client(instance_id)
            request = {
                "lora_id": lora_id,
                "lora_path": lora.path,
                "metadata": lora.metadata
            }

            response = await anext(await client.load_lora(request))
            success = response.get("status") == "success"

            if success:
                logger.info(f"Successfully loaded LoRA {lora_id} on instance {instance_id}")
            else:
                logger.error(f"Failed to load LoRA {lora_id} on instance {instance_id}: {response.get('message')}")

            return success

        except Exception as e:
            logger.error(f"Error loading LoRA {lora_id} on instance {instance_id}: {e}")
            return False

    async def _unload_lora_from_instance(self, lora_id: str, instance_id: int) -> bool:
        """Send unload command to specific instance"""
        try:
            client = await self._get_instance_client(instance_id)
            request = {"lora_id": lora_id}

            response = await anext(await client.unload_lora(request))
            success = response.get("status") == "success"

            if success:
                logger.info(f"Successfully unloaded LoRA {lora_id} from instance {instance_id}")
            else:
                logger.error(f"Failed to unload LoRA {lora_id} from instance {instance_id}: {response.get('message')}")

            return success

        except Exception as e:
            logger.error(f"Error unloading LoRA {lora_id} from instance {instance_id}: {e}")
            return False

    async def _update_assignments(self, lora_id: str, instance_id: int, action: str):
        """Update internal assignment tracking"""
        if action == "loaded":
            # Add to assignments
            if lora_id not in self.lora_assignments:
                self.lora_assignments[lora_id] = []
            if instance_id not in self.lora_assignments[lora_id]:
                self.lora_assignments[lora_id].append(instance_id)

            if instance_id not in self.instance_assignments:
                self.instance_assignments[instance_id] = []
            if lora_id not in self.instance_assignments[instance_id]:
                self.instance_assignments[instance_id].append(lora_id)

        elif action == "unloaded":
            # Remove from assignments
            if lora_id in self.lora_assignments and instance_id in self.lora_assignments[lora_id]:
                self.lora_assignments[lora_id].remove(instance_id)
                if not self.lora_assignments[lora_id]:
                    del self.lora_assignments[lora_id]

            if instance_id in self.instance_assignments and lora_id in self.instance_assignments[instance_id]:
                self.instance_assignments[instance_id].remove(lora_id)
                if not self.instance_assignments[instance_id]:
                    del self.instance_assignments[instance_id]

        # Update service discovery
        instance_loras = self.instance_assignments.get(instance_id, [])
        await self.service_discovery.register_lora_info(instance_id, {
            "loaded_loras": instance_loras,
            "available_lora_slots": self.max_loras_per_instance - len(instance_loras)
        })

    async def _get_available_instances(self) -> List[int]:
        """Get list of available backend instances"""
        # This would integrate with existing service discovery
        return [1, 2, 3]  # Placeholder

    async def _get_instance_client(self, instance_id: int):
        """Get client for specific instance"""
        # This would integrate with existing client creation logic
        pass

    async def _periodic_rebalance(self):
        """Periodic rebalancing of LoRA assignments"""
        while True:
            try:
                await asyncio.sleep(self.rebalance_interval)
                await self._rebalance_loras()
            except Exception as e:
                logger.error(f"Error in periodic rebalance: {e}")

    async def _periodic_health_check(self):
        """Periodic health check of LoRA assignments"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._health_check_assignments()
            except Exception as e:
                logger.error(f"Error in periodic health check: {e}")

    async def _rebalance_loras(self):
        """Rebalance LoRA assignments based on usage patterns"""
        logger.debug("Starting LoRA rebalancing")

        # Implement rebalancing logic based on:
        # - Usage statistics
        # - Instance capacity
        # - Loading policy

        # For now, just log the current state
        logger.info(f"Current LoRA assignments: {len(self.lora_assignments)} LoRAs loaded")
        logger.info(f"Instance assignments: {self.instance_assignments}")

    async def _health_check_assignments(self):
        """Health check current assignments against actual instance state"""
        logger.debug("Performing LoRA assignment health check")

        # Query actual state from instances and reconcile
        for instance_id in list(self.instance_assignments.keys()):
            try:
                actual_info = await self.service_discovery.get_lora_info(instance_id)
                if actual_info:
                    actual_loras = set(actual_info.get("loaded_loras", []))
                    expected_loras = set(self.instance_assignments.get(instance_id, []))

                    if actual_loras != expected_loras:
                        logger.warning(f"LoRA assignment mismatch for instance {instance_id}")
                        logger.warning(f"Expected: {expected_loras}, Actual: {actual_loras}")

                        # Reconcile the difference
                        await self._reconcile_instance_assignments(instance_id, actual_loras)

            except Exception as e:
                logger.error(f"Health check failed for instance {instance_id}: {e}")

    async def _reconcile_instance_assignments(self, instance_id: int, actual_loras: set):
        """Reconcile assignment state with actual instance state"""
        expected_loras = set(self.instance_assignments.get(instance_id, []))

        # Update our tracking to match reality
        self.instance_assignments[instance_id] = list(actual_loras)

        # Update lora -> instance mappings
        for lora_id in expected_loras - actual_loras:
            # LoRA was expected but not actually loaded
            if lora_id in self.lora_assignments and instance_id in self.lora_assignments[lora_id]:
                self.lora_assignments[lora_id].remove(instance_id)
                if not self.lora_assignments[lora_id]:
                    del self.lora_assignments[lora_id]

        for lora_id in actual_loras - expected_loras:
            # LoRA is loaded but not in our tracking
            if lora_id not in self.lora_assignments:
                self.lora_assignments[lora_id] = []
            if instance_id not in self.lora_assignments[lora_id]:
                self.lora_assignments[lora_id].append(instance_id)
```

### 2. Kubernetes Integration with New Service Discovery

#### 2.1 Pod Labels and Service Configuration

Following the [Dynamo Service Discovery Enhancement](https://raw.githubusercontent.com/ai-dynamo/enhancements/981844c8e1ca48552c0cc6a9ff34bb86b61c23c2/deps/etcd-k8s.md), LoRA-enabled pods use specific labels for service discovery:

```yaml
# LoRA-enabled vLLM Worker Pod
apiVersion: v1
kind: Pod
metadata:
  name: dynamo-vllm-worker-abc123
  labels:
    nvidia.com/dynamo-namespace: dynamo
    nvidia.com/dynamo-component: worker
    nvidia.com/dynamo-lora-enabled: "true"  # New label for LoRA capability
spec:
  containers:
  - name: vllm-worker
    image: vllm/vllm-openai:v0.7.1
    env:
    - name: VLLM_ALLOW_RUNTIME_LORA_UPDATING
      value: "True"
    - name: DYNAMO_LORA_PATH
      value: "/shared/loras"
    - name: VLLM_LORA_MODULES_LOADING_TIMEOUT
      value: "300"
    args:
    - "--enable-lora"
    - "--max-loras=4"
    - "--max-lora-rank=64"
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
# Service for LoRA-enabled workers
apiVersion: v1
kind: Service
metadata:
  name: dynamo-worker-service
spec:
  selector:
    nvidia.com/dynamo-namespace: dynamo
    nvidia.com/dynamo-component: worker
  ports:
  - port: 8000
    targetPort: 8000
    name: http
```

#### 2.2 Enhanced Metadata Endpoint

Each LoRA-enabled worker exposes metadata following the new service discovery pattern:

```python
# components/src/dynamo/vllm/metadata_endpoint.py
from fastapi import FastAPI
from typing import Dict, Any
import asyncio

class LoRAMetadataEndpoint:
    """Metadata endpoint for service discovery integration"""

    def __init__(self, lora_manager, config):
        self.lora_manager = lora_manager
        self.config = config
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/metadata")
        async def get_metadata() -> Dict[str, Any]:
            """Return instance metadata for service discovery"""
            loaded_loras = await self.lora_manager.list_loras()

            return {
                "model": {
                    "name": self.config.served_model_name,
                    "type": "Completions",
                    "runtime_config": {
                        "total_kv_blocks": self.config.total_kv_blocks,
                        "max_num_seqs": self.config.max_num_seqs,
                        "max_num_batched_tokens": self.config.max_num_batched_tokens
                    }
                },
                "transport": {
                    "type": "http",
                    "endpoint": f"http://0.0.0.0:{self.config.port}"
                },
                "capabilities": {
                    "lora_support": True,
                    "dynamic_loading": True,
                    "max_concurrent_loras": self.lora_manager.max_loras
                },
                "lora": {
                    "loaded_loras": {
                        lora.lora_id: {
                            "status": lora.status.value,
                            "base_model": lora.base_model,
                            "namespace": lora.namespace,
                            "revision": lora.revision,
                            "memory_usage": lora.memory_usage,
                            "last_used": lora.last_used,
                            "load_time": lora.load_time
                        }
                        for lora in loaded_loras
                    },
                    "lora_capacity": {
                        "max_loras": self.lora_manager.max_loras,
                        "loaded_count": len(loaded_loras),
                        "available_slots": self.lora_manager.max_loras - len(loaded_loras)
                    },
                    "lora_endpoints": {
                        "load_lora": "/v1/load_lora_adapter",
                        "unload_lora": "/v1/unload_lora_adapter",
                        "list_loras": "/v1/list_lora_adapters"
                    }
                }
            }

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint for readiness probe"""
            try:
                # Check if LoRA manager is healthy
                health_status = await self.lora_manager.health_check()

                if health_status.get("status") == "healthy":
                    return {"status": "ready"}
                else:
                    return {"status": "not_ready", "reason": "lora_manager_unhealthy"}
            except Exception as e:
                return {"status": "not_ready", "reason": str(e)}
```

#### 2.3 DynamoModel CRD Extension

Extend the DynamoModel Custom Resource to support LoRA specifications:

```yaml
# DynamoModel CRD with LoRA support
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: qwen-coder-with-loras
  namespace: default
spec:
  baseModel:
    name: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    servedModelName: "qwen-coder-1-5b-instruct"

  # LoRA specifications
  loras:
  - loraId: "qwen-coder/sql-adapter/v1"
    source:
      type: "huggingface"
      repository: "ai-blond/Qwen-Qwen2.5-Coder-1.5B-Instruct-sql-lora"
      revision: "main"
    priority: 10
    loadingPolicy: "eager"

  - loraId: "qwen-coder/python-adapter/v1"
    source:
      type: "pvc"
      path: "/shared/loras/dynamo/qwen-coder/v1/python-adapter"
    priority: 5
    loadingPolicy: "lazy"

  - loraId: "qwen-coder/javascript-adapter/v1"
    source:
      type: "s3"
      bucket: "my-lora-bucket"
      path: "loras/qwen-coder/javascript-adapter"
    priority: 3
    loadingPolicy: "lazy"

  # Deployment configuration
  deployment:
    replicas: 3
    resources:
      requests:
        nvidia.com/gpu: 1
        memory: "16Gi"
      limits:
        nvidia.com/gpu: 1
        memory: "16Gi"

    # LoRA-specific configuration
    loraConfig:
      maxLoras: 4
      maxLoraRank: 64
      loraExtraVocabSize: 256

  # Service discovery labels
  labels:
    nvidia.com/dynamo-namespace: "dynamo"
    nvidia.com/dynamo-component: "worker"
    nvidia.com/dynamo-lora-enabled: "true"
```

### 3. Kubernetes Operator Integration

```python
# deploy/cloud/operator/internal/dynamo/lora_controller.py
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class LoRAController:
    """Kubernetes controller for LoRA management"""

    def __init__(self, k8s_client, dynamo_client):
        self.k8s_client = k8s_client
        self.dynamo_client = dynamo_client

    async def handle_dynamo_model_cr(self, cr_spec: Dict[str, Any]):
        """Handle DynamoModel Custom Resource with LoRA specifications"""

        model_name = cr_spec.get("model_name")
        lora_specs = cr_spec.get("loras", [])

        logger.info(f"Processing DynamoModel CR for {model_name} with {len(lora_specs)} LoRAs")

        for lora_spec in lora_specs:
            await self._process_lora_spec(model_name, lora_spec)

    async def _process_lora_spec(self, model_name: str, lora_spec: Dict[str, Any]):
        """Process individual LoRA specification"""

        lora_id = lora_spec.get("lora_id")
        source = lora_spec.get("source")  # "pvc", "huggingface", "s3", etc.

        if source == "huggingface":
            await self._download_from_huggingface(lora_id, lora_spec)
        elif source == "s3":
            await self._download_from_s3(lora_id, lora_spec)
        elif source == "pvc":
            # Already available on PVC
            pass

        # Request LoRA loading through Dynamo LoRA Manager
        await self._request_lora_loading(lora_id, lora_spec)

    async def _download_from_huggingface(self, lora_id: str, spec: Dict[str, Any]):
        """Download LoRA from HuggingFace Hub"""
        hf_repo = spec.get("huggingface_repo")
        revision = spec.get("revision", "main")

        # Create download job
        download_job = self._create_download_job(
            lora_id=lora_id,
            source_type="huggingface",
            source_url=hf_repo,
            revision=revision
        )

        await self.k8s_client.create_job(download_job)
        logger.info(f"Created download job for LoRA {lora_id} from HuggingFace")

    async def _download_from_s3(self, lora_id: str, spec: Dict[str, Any]):
        """Download LoRA from S3"""
        s3_path = spec.get("s3_path")

        download_job = self._create_download_job(
            lora_id=lora_id,
            source_type="s3",
            source_url=s3_path
        )

        await self.k8s_client.create_job(download_job)
        logger.info(f"Created download job for LoRA {lora_id} from S3")

    def _create_download_job(self, lora_id: str, source_type: str, source_url: str, revision: str = None) -> Dict[str, Any]:
        """Create Kubernetes Job for downloading LoRA"""

        job_name = f"lora-download-{lora_id.replace('/', '-')}"

        if source_type == "huggingface":
            command = [
                "python", "-c",
                f"""
import os
from huggingface_hub import snapshot_download

lora_path = os.path.join(os.environ['DYNAMO_LORA_PATH'], '{lora_id}')
os.makedirs(os.path.dirname(lora_path), exist_ok=True)

snapshot_download(
    repo_id='{source_url}',
    revision='{revision or "main"}',
    local_dir=lora_path,
    local_dir_use_symlinks=False
)
print(f"Downloaded LoRA to {{lora_path}}")
                """
            ]
        elif source_type == "s3":
            command = [
                "aws", "s3", "sync", source_url, f"$DYNAMO_LORA_PATH/{lora_id}"
            ]

        job_spec = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "labels": {
                    "app": "dynamo-lora-download",
                    "lora-id": lora_id
                }
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "downloader",
                            "image": "dynamo/lora-downloader:latest",
                            "command": command,
                            "env": [{
                                "name": "DYNAMO_LORA_PATH",
                                "value": "/shared/loras"
                            }],
                            "volumeMounts": [{
                                "name": "lora-storage",
                                "mountPath": "/shared/loras"
                            }]
                        }],
                        "volumes": [{
                            "name": "lora-storage",
                            "persistentVolumeClaim": {
                                "claimName": "dynamo-lora-pvc"
                            }
                        }],
                        "restartPolicy": "Never"
                    }
                },
                "backoffLimit": 3
            }
        }

        return job_spec

    async def _request_lora_loading(self, lora_id: str, spec: Dict[str, Any]):
        """Request LoRA loading through Dynamo LoRA Manager"""

        priority = spec.get("priority", 0)

        # Send request to LoRA Manager
        request = {
            "action": "load_lora",
            "lora_id": lora_id,
            "priority": priority,
            "policy": spec.get("loading_policy", "lazy_ttl")
        }

        # This would send to LoRA Manager service
        await self.dynamo_client.send_lora_management_request(request)
        logger.info(f"Requested loading of LoRA {lora_id} with priority {priority}")
```

### 3. Enhanced KV Routing with LoRA Support

```python
# lib/llm/src/kv_router/lora_kv_router.rs (conceptual - would be implemented in Rust)

// Enhanced KV router that includes LoRA ID in block hash computation
// to prevent cross-contamination between different LoRAs

use std::collections::HashMap;
use sha2::{Digest, Sha256};

pub struct LoRAKVRouter {
    base_router: KVRouter,
    lora_instance_map: HashMap<String, Vec<i64>>, // lora_id -> instance_ids
}

impl LoRAKVRouter {
    pub fn new(base_router: KVRouter) -> Self {
        Self {
            base_router,
            lora_instance_map: HashMap::new(),
        }
    }

    pub fn compute_block_hash(&self, token_ids: &[i32], lora_id: Option<&str>) -> String {
        let mut hasher = Sha256::new();

        // Include token IDs
        for &token_id in token_ids {
            hasher.update(token_id.to_le_bytes());
        }

        // Include LoRA ID to prevent cross-contamination
        if let Some(lora_id) = lora_id {
            hasher.update(lora_id.as_bytes());
        } else {
            hasher.update(b"__base_model__");
        }

        format!("{:x}", hasher.finalize())
    }

    pub fn find_best_worker(&self, token_ids: &[i32], lora_id: Option<&str>) -> Option<i64> {
        let block_hash = self.compute_block_hash(token_ids, lora_id);

        // If LoRA is specified, filter to instances that have it loaded
        let candidate_instances = if let Some(lora_id) = lora_id {
            self.lora_instance_map.get(lora_id).cloned().unwrap_or_default()
        } else {
            self.base_router.get_all_instances()
        };

        if candidate_instances.is_empty() {
            return None;
        }

        // Use existing KV routing logic on filtered instances
        self.base_router.find_best_worker_from_candidates(&block_hash, &candidate_instances)
    }

    pub fn update_lora_instance_mapping(&mut self, lora_id: String, instance_ids: Vec<i64>) {
        self.lora_instance_map.insert(lora_id, instance_ids);
    }
}
```

## Updated Implementation Plan

### Integration with Dynamo Service Discovery

Our implementation leverages the [Dynamo Service Discovery Enhancement](https://raw.githubusercontent.com/ai-dynamo/enhancements/981844c8e1ca48552c0cc6a9ff34bb86b61c23c2/deps/etcd-k8s.md) for seamless LoRA management:

1. **Kubernetes-Native Discovery**: Uses EndpointSlices and Services instead of ETCD
2. **Metadata-Driven LoRA State**: Each backend advertises loaded LoRAs through `/metadata` endpoint
3. **Event-Driven Updates**: Real-time LoRA state changes via Kubernetes watch APIs
4. **Pod Lifecycle Integration**: LoRA state automatically cleaned up when pods terminate

### Phase 1: POC (4-6 weeks)

1. **Week 1-2: Core Infrastructure**
   - Implement LoRA interface and discovery system
   - Create vLLM LoRA manager
   - Add LoRA endpoints to backend handlers

2. **Week 3-4: Routing Integration**
   - Extend KV router with LoRA support
   - Implement static LoRA routing
   - Add service discovery integration

3. **Week 5-6: Testing and Validation**
   - End-to-end testing with 2-3 LoRAs
   - Performance benchmarking
   - Documentation and examples

### Phase 2: Full Management (6-8 weeks)

1. **Week 1-2: LoRA Manager**
   - Implement central LoRA management subsystem
   - Add loading policies (eager/lazy)
   - Create health checking and reconciliation

2. **Week 3-4: Kubernetes Integration**
   - Extend operator with LoRA support
   - Implement DynamoModel CRD
   - Add automatic LoRA downloading

3. **Week 5-6: Smart Routing**
   - Implement dynamic routing updates
   - Add usage-based optimization
   - Create load balancing improvements

4. **Week 7-8: Production Features**
   - Add monitoring and metrics
   - Implement auto-scaling
   - Performance optimization and testing

### Key Benefits of This Design

1. **Kubernetes-Native**: Fully integrates with Dynamo's new service discovery approach
2. **Industry-Proven**: Incorporates best practices from AIBrix, vLLM, and NVIDIA NIM
3. **Production-Ready**: Includes reliability features like retry mechanisms and health checks
4. **Scalable**: Supports dynamic LoRA loading across multiple backend types
5. **Observable**: Rich metadata and monitoring integration

### Example Usage

```bash
# Deploy a model with multiple LoRAs
kubectl apply -f - <<EOF
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: coding-assistant
spec:
  baseModel:
    name: "Qwen/Qwen2.5-Coder-7B-Instruct"
  loras:
  - loraId: "coding/python-expert/v1"
    source:
      type: "huggingface"
      repository: "microsoft/CodeT5-python-lora"
    priority: 10
  - loraId: "coding/sql-expert/v1"
    source:
      type: "pvc"
      path: "/shared/loras/coding/sql-expert/v1"
    priority: 8
EOF

# Send request with specific LoRA
curl -X POST http://dynamo-frontend:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coding-assistant",
    "lora_id": "coding/python-expert/v1",
    "messages": [
      {"role": "user", "content": "Write a Python function to sort a list"}
    ],
    "max_tokens": 200
  }'
```

This design provides a comprehensive multi-LoRA system that integrates seamlessly with Dynamo's evolving architecture, leveraging Kubernetes-native service discovery while incorporating industry best practices for production-grade LoRA serving.
