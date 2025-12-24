# Testing Instructions

NOTE: Due to meta-model creation, some models/quantizations will not work because of `.item()` calls in `__init__`.

### Step 1: Start GPU Memory Service

For each device you intend to run on, start in a separate terminal:

```bash
python3 -m dynamo.gpu_memory_service --device $DEVICE_ID
```

### Step 2: Start a shadow engine

```bash
DYN_SYSTEM_PORT=8100 python3 -m dynamo.sglang --model-path Qwen/Qwen3-14B --tensor-parallel-size 2 --load-format gpu_memory_service --enable-memory-saver
```

In another terminal, put the engine to sleep (so it becomes a shadow):

```bash
curl http://localhost:8100/engine/release_memory_occupation -H "Content-Type: application/json" -d '{}'
```

NOTE: This releases all memory types (weights, kv_cache, cuda_graph) by default, which is what you want for shadow engines.

NOTE 2: When using GPU Memory Service for weights, the physical memory is NOT freed on sleep. The allocation server retains the memory for weight sharing. Only the VA mappings are unmapped.

### Step 3: Start primary engine

We need to avoid port conflicts:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DYN_SYSTEM_PORT` | - | Dynamo system status server (health, metrics, engine routes) |
| `--port` | 30000 | SGLang internal server port |
| `--disaggregation-bootstrap-port` | 8998 | PD disaggregation bootstrap port |
| `--kv-events-config` | None | ZMQ endpoint for KV events (e.g., `{"endpoint":"tcp://*:20080"}`) |

Run the following in a different terminal:

```bash
DYN_SYSTEM_PORT=8101 python3 -m dynamo.sglang --model-path Qwen/Qwen3-14B --tensor-parallel-size 2 --load-format gpu_memory_service --port 30001 --enable-memory-saver
```

If using KV events with the router, also specify different ZMQ ports:

```bash
DYN_SYSTEM_PORT=8101 python3 -m dynamo.sglang --model-path Qwen/Qwen3-14B --tensor-parallel-size 2 --load-format gpu_memory_service --port 30001 --enable-memory-saver --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}'
```

### Step 4: Run inference with the primary engine

Run inference as you normally would with Dynamo frontend.

### Step 5: Kill primary engine

This simulates a failure in the primary engine. We can also inject arbitrary types of failures into the primary engine before killing it to test the resiliency of the GPU memory service.

### Step 6: Start new shadow engine

We can use the same default ports as the primary engine since it is already dead:

```bash
DYN_SYSTEM_PORT=8100 python3 -m dynamo.sglang --model-path Qwen/Qwen3-14B --tensor-parallel-size 2 --load-format gpu_memory_service
```

### Step 7: Wake up shadow engine

```bash
curl http://localhost:8100/engine/resume_memory_occupation -H "Content-Type: application/json" -d '{}'
```

Then, inference can continue as normal.
