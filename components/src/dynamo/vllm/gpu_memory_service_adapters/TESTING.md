# Testing Instructions

NOTE: Due to meta-model creation, this will not work with MoE in vLLM specifically (among other paths) because of .item() calls in \_\_init__.

### Step 1: Start GPU Memory Service (GPU Memory Service)
For each device you intend to on, run in a separate terminal:

```bash
python3 -m dynamo.gpu_memory_service --device $DEVICE_ID
```

### Step 2: Start a shadow engine
```bash
DYN_SYSTEM_PORT=8100 python3 -m dynamo.vllm --model Qwen/Qwen3-14B -tp 2 --load-format gpu_memory_service --enable-sleep-mode
```

In another terminal, put the engine to sleep (so it becomes a shadow):

```bash
curl http://localhost:8100/engine/sleep -H "Content-Type: application/json" -d '{"level": 1}'
```

NOTE: `level` is a no-op. It is there for regular vLLM sleep/wake.

NOTE 2: We can also call cuda-checkpoint on the shadow engine to make it resilient to NVLink failures.

### Step 3: Start primary engine

We need to avoid NIXL port conflicts since the default settings are `VLLM_NIXL_SIDE_CHANNEL_PORT=5600` and `DYN_VLLM_KV_EVENT_PORT=20080`.

```bash
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 DYN_VLLM_KV_EVENT_PORT=20081 DYN_SYSTEM_PORT=8101 python3 -m dynamo.vllm --model Qwen/Qwen3-14B -tp 2 --load-format gpu_memory_service --enable-sleep-mode
```

### Step 4: Run inference with the primary engine
Run inference as you normally would with Dynamo frontend.

### Step 5: Kill primary engine

This simulates a failure in the primary engine. We can also inject arbitrary types of failures into the primary engine before killing it to test the resiliency of the GPU memory service.

### Step 6: Start new shadow engine
We can use the same default ports as the primary engine since it is already dead:
```bash
DYN_SYSTEM_PORT=8100 python3 -m dynamo.vllm --model Qwen/Qwen3-14B -tp 2 --load-format gpu_memory_service --enable-sleep-mode
```

### Step 6: Wake up shadow engine
```bash
curl http://localhost:8101/engine/wake -H "Content-Type: application/json"
```

NOTE: If we used cuda-checkpoint on sleep, we also need to restore on wake.

Then, inference can continue as normal.
