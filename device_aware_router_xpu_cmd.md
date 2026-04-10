# Device-Aware Weighted Router for XPU/CPU - Test Commands

This document contains detailed commands for testing the device-aware weighted routing feature adapted from CUDA/CPU to XPU/CPU.

## Test Command

```bash
# Set environment variables for the test
DYN_ENCODE_WORKER_GPU=0 \
DYN_PREFILL_WORKER_GPU=1 \
DYN_DECODE_WORKER_GPU=2 \
DEVICE_PLATFORM='xpu' \
DYN_ROUTER_MODE='device-aware-weighted' \
DYN_ENCODER_XPU_TO_CPU_RATIO=8 \
bash examples/backends/vllm/launch/xpu/disagg_multimodal_epd_xpu.sh \
  --model Qwen/Qwen2.5-VL-3B-Instruct
```

## Environment Variables Explained

| Variable | Value | Purpose |
|----------|-------|---------|
| `DYN_ENCODE_WORKER_GPU` | `0` | Encoder worker uses XPU device 0 |
| `DYN_PREFILL_WORKER_GPU` | `1` | Prefill worker uses XPU device 1 |
| `DYN_DECODE_WORKER_GPU` | `2` | Decode worker uses XPU device 2 |
| `DEVICE_PLATFORM` | `xpu` | Use Intel XPU devices |
| `DYN_ROUTER_MODE` | `device-aware-weighted` | Enable device-aware weighted routing |
| `DYN_ENCODER_XPU_TO_CPU_RATIO` | `8` | Prefer XPU 8:1 over CPU (default) |

## Verification Commands

### 1. Check Service Health & Device Types
```bash
curl -s http://localhost:8000/health | jq '{
  status, 
  devices: [.instances[] | {
    component, 
    endpoint, 
    device_type
  }]
}'
```

**Expected Output:**
```json
{
  "status": "healthy",
  "devices": [
    {"component": "encode", "endpoint": "generate", "device_type": "xpu"},
    {"component": "prefill", "endpoint": "generate", "device_type": "xpu"},
    {"component": "backend", "endpoint": "generate", "device_type": "xpu"}
  ]
}
```

### 2. Send Test Request
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "messages": [{
      "role": "user", 
      "content": [{"type": "text", "text": "What is 5+5?"}]
    }],
    "max_tokens": 20
  }' | jq -r '.choices[0].message.content'
```

**Expected Output:**
```
The answer to 5 + 5 is 10.
```

### 3. Send Multiple Parallel Requests
```bash
for i in {1..3}; do 
  curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{
      \"model\": \"Qwen/Qwen2.5-VL-3B-Instruct\",
      \"messages\": [{
        \"role\": \"user\", 
        \"content\": [{\"type\": \"text\", \"text\": \"Test $i\"}]
      }],
      \"max_tokens\": 20
    }" &
done
wait
```

### 4. Check Router Logs
```bash
# View device-aware routing decisions
tail -f /path/to/output.log | grep "DeviceAwareWeighted"
```

**Expected Log Output:**
```
INFO DeviceAwareWeighted selected instance 
     endpoint=dynamo/prefill/generate 
     selected_instance=7587894069709992097 
     is_cpu=false
```

## Testing Different Ratios

### High XPU Preference (Ratio = 16)
```bash
DYN_ENCODER_XPU_TO_CPU_RATIO=16 \
bash examples/backends/vllm/launch/xpu/disagg_multimodal_epd_xpu.sh \
  --model Qwen/Qwen2.5-VL-3B-Instruct
```
*Routes to CPU only when XPU load is very high (16x)*

### Balanced (Ratio = 2)
```bash
DYN_ENCODER_XPU_TO_CPU_RATIO=2 \
bash examples/backends/vllm/launch/xpu/disagg_multimodal_epd_xpu.sh \
  --model Qwen/Qwen2.5-VL-3B-Instruct
```
*More balanced CPU/XPU usage (2x preference)*

## Mixed XPU/CPU Setup (Advanced)

To test actual heterogeneous routing with both XPU and CPU workers:

```bash
# Terminal 1: Start XPU encoder on device 0
ZE_AFFINITY_MASK=0 python -m dynamo.vllm \
  --multimodal-encode-worker \
  --model Qwen/Qwen2.5-VL-3B-Instruct

# Terminal 2: Start CPU encoder (no XPU devices visible)
ZE_AFFINITY_MASK="" python -m dynamo.vllm \
  --multimodal-encode-worker \
  --model Qwen/Qwen2.5-VL-3B-Instruct

# Terminal 3: Start frontend with device-aware routing
python -m dynamo.frontend \
  --router-mode device-aware-weighted
```

Then verify mixed routing:
```bash
curl -s http://localhost:8000/health | jq -r '.instances[] | 
  select(.component=="encode") | 
  {instance_id, device_type}'
```

**Expected Output:**
```json
{"instance_id": 123456, "device_type": "xpu"}
{"instance_id": 123457, "device_type": "cpu"}
```

## Cleanup

```bash
# Stop all services
pkill -f "disagg_multimodal_epd_xpu.sh"
pkill -f "python.*dynamo"
```

## Full Test Script

Here's a complete test script:

```bash
#!/bin/bash
set -e

echo "Starting device-aware weighted routing test..."

# Start services
DYN_ENCODE_WORKER_GPU=0 \
DYN_PREFILL_WORKER_GPU=1 \
DYN_DECODE_WORKER_GPU=2 \
DEVICE_PLATFORM='xpu' \
DYN_ROUTER_MODE='device-aware-weighted' \
DYN_ENCODER_XPU_TO_CPU_RATIO=8 \
bash examples/backends/vllm/launch/xpu/disagg_multimodal_epd_xpu.sh \
  --model Qwen/Qwen2.5-VL-3B-Instruct &

# Wait for startup
echo "Waiting 60s for services to initialize..."
sleep 60

# Test 1: Health check
echo -e "\n=== Test 1: Health Check ==="
curl -s http://localhost:8000/health | jq -r '.instances[] | 
  select(.component=="encode" or .component=="prefill" or .component=="backend") | 
  "\(.component)/\(.endpoint): \(.device_type)"' | sort | uniq

# Test 2: Single request
echo -e "\n=== Test 2: Single Request ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
    "max_tokens": 10
  }' | jq -r '.choices[0].message.content'

# Test 3: Parallel requests
echo -e "\n=== Test 3: Parallel Requests ==="
for i in {1..3}; do 
  curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"Qwen/Qwen2.5-VL-3B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"Test $i\"}]}], \"max_tokens\": 10}" &
done
wait

echo -e "\n=== All tests completed! ==="

# Cleanup
echo "Stopping services..."
pkill -f "disagg_multimodal_epd_xpu.sh"
pkill -f "python.*dynamo"
```

Save as `test_device_aware_routing.sh` and run:
```bash
chmod +x test_device_aware_routing.sh
./test_device_aware_routing.sh
```

## How Device-Aware Routing Works

### 1. Device Detection
The router automatically detects device types using Intel XPU environment variables:
- `ZE_AFFINITY_MASK` - Intel Level Zero affinity mask
- `ONEAPI_DEVICE_SELECTOR` - oneAPI device selector

### 2. Worker Partitioning
Workers are partitioned into two groups:
- **XPU Group**: Workers with XPU devices available
- **CPU Group**: Workers with no XPU devices (CPU-only)

### 3. Budget-Based Selection
The router uses a budget formula to decide which group to route to:

```
allowed_cpu_inflight = (total_xpu_inflight × cpu_worker_count) / (ratio × xpu_worker_count)

if (total_cpu_inflight < allowed_cpu_inflight):
    route_to → CPU_GROUP
else:
    route_to → XPU_GROUP
```

Where:
- `ratio` = `DYN_ENCODER_XPU_TO_CPU_RATIO` (default: 8)
- Higher ratio = stronger XPU preference

### 4. Least-Loaded Selection
Within the selected group, the router picks the worker with the fewest active requests.

## Configuration Parameters

### Frontend Arguments
```bash
python -m dynamo.frontend \
  --router-mode device-aware-weighted  # Enable device-aware routing
```

### Environment Variables
```bash
# XPU to CPU preference ratio (default: 8)
export DYN_ENCODER_XPU_TO_CPU_RATIO=8

# Intel XPU device visibility
export ZE_AFFINITY_MASK=0,1,2        # Use XPU devices 0, 1, 2
export ZE_AFFINITY_MASK=""           # CPU-only mode

# oneAPI device selector
export ONEAPI_DEVICE_SELECTOR=level_zero:0  # Use Level Zero device 0
export ONEAPI_DEVICE_SELECTOR=cpu           # CPU-only mode
```

## Troubleshooting

### Check Device Type Detection
```bash
# Should show "xpu" for XPU workers
curl -s http://localhost:8000/health | jq '.instances[].device_type'
```

### Verify Router Mode
```bash
# Check logs for router activation
grep "Activating.*router" logs/frontend.log
# Should see: "router_mode=DeviceAwareWeighted"
```

### Monitor Routing Decisions
```bash
# Watch routing decisions in real-time
tail -f logs/frontend.log | grep "DeviceAwareWeighted selected"
```

### Common Issues

**Issue**: All workers show `device_type: null`
**Solution**: Device detection failed. Check Intel XPU drivers and environment variables.

**Issue**: Router falls back to round-robin
**Solution**: Verify `DYN_ROUTER_MODE='device-aware-weighted'` is set correctly.

**Issue**: Only routing to one device type
**Solution**: This is expected if you only have XPU or CPU workers. For heterogeneous routing, ensure you have both.

## Performance Tuning

### High-Throughput Workloads
Use higher ratio to maximize XPU utilization:
```bash
export DYN_ENCODER_XPU_TO_CPU_RATIO=16
```

### Balanced Workloads
Use lower ratio for more even distribution:
```bash
export DYN_ENCODER_XPU_TO_CPU_RATIO=2
```

### CPU Offload for Burst Traffic
Use ratio=1 to use CPU as overflow capacity equally:
```bash
export DYN_ENCODER_XPU_TO_CPU_RATIO=1
```

## Summary of Changes from PR 7215

This implementation adapts PR 7215 from CUDA/CPU to XPU/CPU:

### Modified Files (13 total)
1. `lib/runtime/src/component.rs` - DeviceType enum
2. `lib/runtime/src/component/endpoint.rs` - XPU device detection
3. `lib/runtime/src/pipeline/network/egress/push_router.rs` - Routing logic
4. `lib/runtime/src/discovery/mod.rs` - Discovery metadata
5. `components/src/dynamo/frontend/frontend_args.py` - CLI arguments
6. `components/src/dynamo/frontend/main.py` - Router mode config
7. `docs/components/router/router-guide.md` - Documentation
8. Plus 6 other supporting files

### Key Changes
- `Cuda` → `Xpu` in DeviceType enum
- `DYN_ENCODER_CUDA_TO_CPU_RATIO` → `DYN_ENCODER_XPU_TO_CPU_RATIO`
- NVIDIA env vars → Intel XPU env vars
- All documentation updated for XPU

## References

- Original PR: [ai-dynamo/dynamo#7215](https://github.com/ai-dynamo/dynamo/pull/7215)
- Intel XPU Documentation: [Intel Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/docs/processors/max-series/overview.html)
- Router Guide: `docs/components/router/router-guide.md`
