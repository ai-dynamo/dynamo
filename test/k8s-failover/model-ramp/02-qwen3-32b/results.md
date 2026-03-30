# Qwen3-32B Results

## Model info
- **Model**: Qwen/Qwen3-32B
- **Type**: Dense (largest dense in Qwen3 family)
- **Parameters**: 32.8B (BF16)
- **Engine image**: `multinode-failover-650234f660-vllm-runtime`

## Test 1: Baseline eager (TP=4, 0.85 util)

| Metric | Value |
|--------|-------|
| nvidia-smi per GPU | 70,383 MiB used / 10,770 MiB free |
| KV cache | 50.94 GiB, 834,608 tokens, 52,163 blocks |
| Weights load time | 9.49s |
| Init time | 8.58s |
| Inference | 200, 1270ms |

## Test 2: Baseline CUDA graphs (TP=4, 0.85 util)

| Metric | Value |
|--------|-------|
| nvidia-smi per GPU | 71,287 MiB used / 9,866 MiB free |
| KV cache | 50.85 GiB, 833,136 tokens, 52,071 blocks |
| torch.compile | 40.92s |
| Init time | 54.45s |
| Inference | 200, 372ms |
| CUDA graphs overhead | 71,287 - 70,383 = 904 MiB/GPU |

## Test 3: Failover eager (TP=4, GMS, 0.85 util)

### Bug fix verification (image 650234f660)

| | Engine-0 (RW) | Engine-1 (RO) |
|--|---|---|
| non_kv | 15.96 GiB | 16.07 GiB |
| projected KV | 51.40 GiB | 51.29 GiB |
| torch_peak | 15.96 GiB | 0.58 GiB |
| weights | 15.51 GiB | 15.49 GiB |

Both engines compute matching KV cache sizes (~51 GiB). Bug fix confirmed.

### Results

| Step | Result |
|------|--------|
| Deploy | 0 restarts, both engines ready |
| GMS weights | 15.51 GiB/GPU, 262 mappings |
| nvidia-smi | 73,513 MiB/GPU |
| Inference | 200, 1258ms |
| Failover (kill e0) | ~3s (lock -> remap -> KV alloc -> generate) |
| KV cache on wake | 51.29 GiB (64 tensors) |
| Inference post-failover | 200, 1086ms |

## Test 4: Failover CUDA graphs (TP=4, GMS, 0.85 util)

### Bug fix verification

| | Engine-0 (RW) | Engine-1 (RO) |
|--|---|---|
| non_kv | 15.96 GiB | 16.07 GiB |
| projected KV | 51.40 GiB | 51.29 GiB |

Matching. Confirmed.

### Results

| Step | Result |
|------|--------|
| Deploy | 0 restarts |
| nvidia-smi | 74,553 MiB/GPU |
| Inference | 200, 358ms |
| Failover | ~4s |
| KV cache on wake | 51.29 GiB (64 tensors) |
| Inference post-failover | 200, 320ms |

## Test 5: Baseline multinode (TP=4, NNODES=2, 2 GPU/node, enforce-eager, 0.85 util)

| Metric | Value |
|--------|-------|
| nvidia-smi per GPU | 70,025 MiB used / 11,128 MiB free |
| KV cache | 51.39 GiB, 841,904 tokens |
| Init time | 6.58s |
| Inference | 200, 1385ms |

## Test 6: Failover multinode (TP=4, NNODES=2, 2 GPU/node, GMS, enforce-eager, 0.85 util)

### Harness bug: etcdctl get polling timeout during model download

The original harness used `etcdctl get` polling (1 call/sec, new gRPC connection
each time). During 32B model download (~65 GB), these connections hit the 5s
command timeout (`context deadline exceeded`), causing false "key disappeared"
detections and cascade restarts.

**Fix**: Replaced get-based polling with persistent `etcdctl watch` streams in
the monitoring loop. Watch streams use the same persistent gRPC connection type
as lease keep-alive, which survives network contention.

### Results (with watch-based harness)

| Step | Result |
|------|--------|
| Deploy | 0 restarts, all 4 engines ready |
| Formation | ~5s (leader + worker coordination via etcd) |
| Inference | 200, 1393ms |
| Failover (kill leader e0) | ~4s (e1 takes over) |
| KV cache on wake | 51.30 GiB (64 tensors) |
| Inference post-failover | 200, 969ms |
| Engine-0 recovery | Restarted as standby, sleeping |
