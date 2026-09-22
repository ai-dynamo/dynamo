<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shutdown validation

Run from a development environment with Rust, rebuilt editable Dynamo bindings, pytest, pytest-asyncio, and pytest-timeout:

```bash
cd lib/bindings/python
maturin develop --uv
cd ../../..
SHUTDOWN_PYTHON=.venv/bin/python bash tests/runtime/validate_shutdown.sh
```

The runner checks runtime teardown and worker stage contracts, then exercises shutdown end to end through real Rust bindings, TCP and NATS streams, OS signals, and subprocess exits. It needs permission to bind local sockets and Docker for the NATS fixture, but no GPU or model download.

| Scenario | Required outcome |
| --- | --- |
| Concurrent runtime shutdown callers | All await the same teardown; cancelling one waiter does not cancel teardown. |
| SDK worker hosted by another application | Clean shutdown returns and the host survives beyond the former watchdog deadline. |
| SDK cleanup holds the GIL | The native watchdog still forces exit code 70. |
| Python pull and push streaming | SIGTERM removes discovery, completes admitted work, then cleans the engine and runtime. |
| Drain consumes its allowance | Cleanup uses its reserve inside the original total without extending the deadline. |
| Blocked Python or SDK cleanup | Process exits by the original SIGTERM deadline within one second of scheduling tolerance. |
| Second SIGTERM | Immediate exit code 70. |
| Idle and slow embedding children | A process-group signal is forwarded once; idle event loops wake, children exit cleanly, and parent cleanup follows. |
| Stale router addresses a closed admission gate | TCP and NATS request planes preserve typed Unavailable rejection for pull and push responses. |

`shutdown_probe.py` is the subprocess helper; use pytest or the runner rather than invoking its signal modes in a shared process group. Every subprocess has a timeout, and embedding children have their own bounded lifetime.

These checks use a CPU probe engine and the shared Python shutdown coordinator used by vLLM, SGLang, and TensorRT-LLM. The embedding scenarios import the production process supervisor while stubbing unused vLLM engine-construction imports. They do not instantiate WorkerFactory or validate GPU engine teardown, distributed KV transfer, external discovery propagation, or Kubernetes termination. Those require backend-specific serving environments and remain separate deployment validation.

## Real-engine container validation

`validate_gpu_shutdown.py` launches the actual Python backend and Dynamo HTTP frontend using the existing `ManagedProcess` harness. It sends SIGTERM after the first token of a 512-token GPU inference stream and requires a normal terminal response, exit code 0, completed engine/runtime cleanup, and no live engine descendants. File discovery and TCP keep the run isolated from existing etcd/NATS deployments. Logs and `result.json` are retained in a unique directory.

Rebuild the bindings first, then run from the checkout root. Mount a Hugging Face hub cache containing `Qwen/Qwen3-0.6B`, select a compatible backend image, and provide a results directory writable by the image's user:

```bash
docker run --rm --gpus all --shm-size=2g --entrypoint python3 \
  -v "$PWD:/shutdown:ro" \
  -v /path/to/huggingface/hub:/models:ro \
  -v /path/to/results:/results \
  -e PYTHONPATH=/shutdown/lib/bindings/python/src:/shutdown/components/src:/shutdown \
  -e HF_HUB_CACHE=/models -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  BACKEND_IMAGE /shutdown/tests/runtime/validate_gpu_shutdown.py BACKEND --output /results
```

`BACKEND` is `vllm`, `sglang`, or `trtllm`. Run sequentially on a single GPU. The bind mounts deliberately select both the checkout's Python source and its rebuilt `_core` library, not the image's released Dynamo code. This aggregated-serving check does not replace multi-node disaggregated KV-transfer or Kubernetes validation.

The probe uses `--total-timeout` (default 30 seconds) as one absolute SIGTERM-origin deadline for stream consumption and process exit, with one second of scheduling tolerance. It records process exit concurrently rather than treating stream completion or descendant cleanup as the exit timestamp. Cleanup has a five-second cap inside the total. A short-budget run that interrupts inference fails this graceful-completion probe; it is not automatically evidence of a shutdown defect.

For deliberate expiry, pass `--total-timeout 5 --cleanup-timeout 2 --expect-interrupted`. This mode requires an interrupted stream, an in-flight timeout log, bounded process exit, and no surviving captured descendants. A zero exit additionally requires successful cleanup and runtime stages. It rejects a request that simply finishes before the drain expires. The September 22 vLLM run passed this case: a 503 terminated the unfinished stream and the worker exited zero after 4.57 seconds, including cleanup and runtime teardown.

## Requirement-to-evidence checklist

The September 22 implementation revision is an uncommitted delta over `818cecb2331c954db97b135502c4463230e8c609`. Earlier container and multi-node results validate their recorded revisions only, not this delta.

| Contract | Current evidence | Remaining acceptance work |
| --- | --- | --- |
| One total including router grace, drain, cleanup, and runtime | Rust/Python budget tests; native/Python blocked-cleanup subprocesses; local and cross-node vLLM budget-exhaustion runs | Extend distributed expiry coverage to other backends and native sidecars |
| Cleanup reserve stays inside total | Pre-cleanup allowances subtract reserve; engine cleanup and runtime share one cap; 180 backend-common tests and 6 runtime shutdown tests passed | Exercise engine-specific slow teardown under load |
| Admission closes with a typed rejection | Four real transport cases: TCP/NATS × pull/push; signal probes check discovery withdrawal and admitted-stream completion; fresh requests reach surviving peers in two Lyris cases | Sustained traffic during replica termination |
| Separate request and KV-transfer drain | Native lifecycle tests, Python fallback coverage, and a controlled real-NIXL overlap run on Lyris | Uninstrumented overlap and corresponding coverage for other backend/path combinations; fallback is not confirmed quiescence |
| Python backend teardown | Revised single-GPU and two-node disaggregated container runs passed for vLLM, SGLang, and TensorRT-LLM | Engine-specific slow teardown under load |
| Rust sidecar teardown | Real-engine runs passed for vLLM/SGLang disaggregated and TensorRT-LLM aggregated | TensorRT-LLM native disaggregation is unsupported; externally launched engines have separate ownership |
| Operator grace covers worker total plus margin | Seven Go table cases passed for final-pod literal budget validation | Real Kubernetes pod deletion; image/envFrom/shell/SDK-only settings remain outside static validation |
| Shutdown observability | Stage duration/reason, remaining total, in-flight count, and KV state; public metrics scrape during shutdown passes on TCP and NATS | Distributed scrape/log correlation during transfer shutdown |

The focused Python suite passed 7 coordinator unit tests and 12 transport/process cases. Container runs use Qwen3-0.6B and the images listed below; the revised runs completed their streams, exited zero, and left no live captured engine descendants. Measured SIGTERM-to-exit times were 12.90 seconds for vLLM, 8.69 seconds for SGLang, and 6.13 seconds for TensorRT-LLM, all against a 30-second total. These are aggregated Python results, not sidecar or active-KV-transfer results.

### Review-fix validation: September 22, 2026

The subsequent review fixes preserve the established worker-unavailable wire identity for draining rejections, isolate gateway children from the parent's signal group, bound the generic worker's complete runtime teardown, and propagate native cleanup failures with a failed stage outcome. They also remove unused shutdown and engine-route scaffolding and correct the budget documentation.

Validation after rebuilding the Python bindings passed all 180 backend-common tests, six runtime shutdown tests, the wire-compatibility and complete-teardown deadline regressions, seven Python coordinator tests, and 14 transport/process cases. The process cases include cleanup-error propagation and production gateway supervision with real child processes and process-group SIGTERM. Temporarily removing child session isolation made the gateway regression fail with child exit code 70; restoring it passed.

The cached `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.1` image passed 23 gateway unit tests without GPU access. One unchanged facade-construction test failed because this image's SGLang lacks `runtime_context.publish`. The image also lacks the optional pytest async/benchmark plugins, so this synchronous suite ran with plugin autoload and the unrelated benchmark warning filter disabled. This is not a fully passing gateway suite.

The earlier GPU and Lyris campaigns predate these review fixes. The subsequent local GPU rerun is recorded below; Lyris has not been repeated against these fixes. The focused Python run also logged tasks still running at interpreter exit; passing bounded-exit assertions does not establish complete internal task retirement.

### Post-review revalidation: September 22, 2026

The reviewed implementation passed eight runtime tests, all 180 backend-common tests, 21 Python coordinator/transport/process cases, and seven operator budget cases. The stack was then split into signed-off commits and rebased; its source was compared against the validated snapshot to verify that the split preserved the implementation.

On a local RTX 5880 Ada using Qwen3-0.6B, all three Python backends completed the full 512-token stream, produced 511 chunks after SIGTERM, exited zero, completed cleanup/runtime stages, and left no captured engine children alive. These runs used the cached images identified in the September 21 table, with the current source and rebuilt binding mounted read-only and no package overrides.

| Backend | SIGTERM to exit | Total budget |
| --- | --- | --- |
| vLLM | 6.68 s | 30 s |
| SGLang | 3.97 s | 30 s |
| TensorRT-LLM | 3.67 s | 30 s |

A five-second vLLM expiry run with a two-second cleanup reserve was inconclusive because inference completed before drain expiry. Repeating with a 3.5-second cleanup reserve inside the same total passed: the in-flight drain timed out, the unfinished stream received a typed 503, cleanup/runtime completed, and the worker exited zero in 2.92 seconds with no surviving captured descendants. The first attempt exposed a probe failure-path limitation: its deadline alarm can fire during harness cleanup and mask the original assertion. That secondary exception is not evidence of a worker deadline violation.

The shutdown branch's `container/context.yaml` pins `lmsysorg/sglang:v0.5.19-cu130-runtime`. Testing that exact image confirmed `runtime_context.publish` exists and all 24 gateway unit tests pass, without compatibility shims. The pulled image digest was `sha256:710bc11443a7b1807d69803386468101bcfced35f86bf8fe92a8209e05a2f052`. This supersedes the older image's gateway unit-test failure, but does not validate real multi-process GPU gateway serving with the pinned image.

Worker logs still contain pending-task messages at interpreter exit; vLLM also reports a semaphore resource-tracker warning. Multinode, native-sidecar GPU serving, active KV transfer, and Kubernetes deletion were not rerun after the review fixes. These results do not establish complete issue #13286 acceptance.

### Revised Lyris campaign: September 22, 2026

Job `3132012` used two GB300 nodes, `theia0026` and `theia0027`, with job-local storage, Enroot containers, Qwen3-0.6B, etcd discovery, and the **NATS request plane**. Both nodes loaded the revised ARM binding, SHA-256 `1a2b871b914d0915a8a6d95042adae6cf2626216111430e23fd9da25d4b98f74`. The staged source archive SHA-256 was `fa69d57fb6874b91dac61961b54e63e7984b8d1933b9f8e0b45925b7b5edb225`. Images were `vllm/vllm-openai:v0.28.0`, `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.1`, and `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.4.1`.

| Backend/path | Serving mode | Decode SIGTERM to exit | Idle prefill SIGTERM to exit | Additional evidence |
| --- | --- | --- | --- | --- |
| vLLM Python | Cross-node disaggregated | 26.13 s | 7.82 s | Fresh unpinned request completed on a surviving decode replica |
| SGLang Python | Cross-node disaggregated | 32.38 s | 7.93 s | Full stream completed |
| TensorRT-LLM Python | Cross-node disaggregated | 5.72 s | 9.55 s | Full stream completed |
| vLLM Rust sidecar | Cross-node disaggregated | 23.68 s | 6.08 s | Full stream completed |
| SGLang Rust sidecar | Cross-node disaggregated | 16.99 s | 6.06 s | Full stream completed |
| TensorRT-LLM Rust sidecar | Aggregated, replicas on separate nodes | 1.45 s | Not applicable | Fresh unpinned request completed on the surviving replica |

Each target received SIGTERM after the first token of a 1,024-token response. These cases required full stream completion, exit zero within a 60-second total plus one second of scheduling tolerance, completed cleanup/runtime stages, and no live captured worker descendants. Controllers measured exit on the worker's own node, independently of client stream completion. Native engines were separate harness-owned processes: sidecar exit validates Dynamo's lifecycle and connection cleanup, while the harness subsequently stops the external engines; it does not establish engine ownership by the sidecar.

The TensorRT-LLM container needed a recorded Protobuf override to `6.33.5` because its installed `smg-grpc-proto` generated code requires Protobuf 6 while the image contains 5.29.6. This conflicts with the unused `modelexpress` package's `<6` requirement. These are not unmodified-image passes. TensorRT-LLM's native sidecar explicitly rejects disaggregation because its Generate response contract has no handoff; the rejected startup is not shutdown evidence. Earlier invalid-port and readiness-race attempts are likewise excluded from passing results.

A separate Python vLLM test used a long prompt and a test-only wrapper around a real NIXL transfer. The read reported `PROC` before signalling prefill and again after the controller observed prefill enter `kv_transfer reason=unsupported`. The response completed and prefill exited zero in 6.86 seconds, with cleanup/runtime complete and no captured descendants left alive. The synchronous control acknowledgement delays normal connector polling, so this proves controlled overlap with an incomplete NIXL operation, not unperturbed physical RDMA activity or a positive engine quiescence predicate. No production transfer implementation was replaced.

The cross-node Python vLLM expiry case set an 8-second total with a 5-second cleanup reserve. The in-flight drain timed out, the unfinished response received a typed 503, and decode exited zero after 6.51 seconds with cleanup/runtime complete and no live captured descendants. Idle prefill exited after 4.86 seconds under the same budget. Decode logs retained an interpreter-exit pending-task message and a semaphore resource-tracker warning; bounded exit and descendant cleanup do not establish that every internal task/resource was individually retired.

Remaining acceptance work must distinguish unsupported-state fallback from confirmed quiescence and enforce the same absolute deadline on both roles. A run that cannot establish signal/transfer overlap is inconclusive. Complete the remaining backend/path coverage and the authorized Kubernetes deletion run before treating issue #13286 as fully validated.

### Verified run: September 21, 2026

Validated source commit `4a44bca2e7` on one NVIDIA RTX 5880 Ada (48 GB), using Qwen3-0.6B and the container command above. All three runs completed the 512-token response, delivered 511 chunks after SIGTERM, exited with code 0, completed cleanup and runtime teardown, and left no live engine children.

| Backend | Container tag | Local image ID | SIGTERM to exit |
| --- | --- | --- | --- |
| vLLM 0.28.0 | `vllm/vllm-openai:latest` | `609a5b463503` | 4.38 s |
| SGLang 0.5.16 | `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.1` | `c916cbde6c23` | 3.96 s |
| TensorRT-LLM 1.3.0rc22 | `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.4.1` | `88430fff8555` | 3.62 s |

The `latest` tag is mutable; the image ID identifies the image actually tested. The older cached `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1` image contains vLLM 0.26.0 and fails inference before shutdown because this source requires `vllm.exceptions.VLLMClientError`; that attempt is not counted as a passing validation. The successful vLLM run used the newer image without a compatibility shim or engine mock.
