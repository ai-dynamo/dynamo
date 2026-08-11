# Remote vLLM Performance Findings

## Summary

The remote decoder path can approximately match the embedded decoder's concurrency-64
throughput when the encoder output is transferred to the vLLM worker and NIXL uses
manual completion polling. The optimized remote path reached 93.12 requests/second,
compared with a 96.75 requests/second historical embedded average. Its remaining gaps
are CPU efficiency and p99 latency.

## Encoder-to-decoder flow

Embedded path:

```text
image -> encoder -> artifacts -> classifier
                           \-> local prompt adapter -> embedded vLLM
```

Optimized remote path:

```text
image -> encoder -> artifacts -> classifier
                           \-> NIXL transfer -> remote prompt adapter -> vLLM worker
```

The classifier consumes the original local artifacts. The remote request carries NIXL
transfer descriptors in `encoder_result`; the tensor data moves out of band. The
request removes `multi_modal_data`, `mm_processor_kwargs`, and `mm_routing_info`, so
the remote vLLM worker does not fetch and encode the image again.

## Findings

1. The initial remote path unnecessarily encoded the media again in the vLLM worker.
   Reusing the first encoder's artifacts eliminated that duplicated GPU work.
2. The largest remaining bottleneck was the NIXL sender progress thread. It consumed
   approximately one logical CPU continuously, including while idle.
3. Disabling the progress thread and polling completion from the request coroutine
   improved average remote throughput by 20.7% over the progress-thread variant.
4. The Dynamo endpoint and remote prompt adapter were not the primary bottlenecks. A
   synthetic remote run that preserved the endpoint and generation path but omitted
   the tensor transfer reached 95.11 requests/second.
5. Forcing UCX shared-memory transport did not improve throughput. Reusing receiver
   destination buffers also did not help because the sender still creates per-request
   descriptors and transfer metadata.

## Optimizations implemented

- Export the encoder artifacts with `export_custom_encoder_artifacts()`.
- Send NIXL transfer descriptors to the remote worker through `encoder_result`.
- Remove the original media fields before calling the remote decoder endpoint.
- Run classification, decoding, and transfer completion concurrently.
- Request only the terminal decoder response.
- Configure the NIXL read sender with:

  ```python
  enable_progress_thread=False
  completion_poll_ms=1
  ```

## Concurrency-64 results

Test workload: Qwen2.5-1.5B-Instruct, one 64x64 image per request, approximately 128
input tokens, exactly 8 output tokens, 100 warmups, and 2,560 measured requests on an
RTX 3090.

| Path | Throughput | Mean latency | p99 latency |
|---|---:|---:|---:|
| Embedded, clean reference run | 90.57 req/s | 701.87 ms | 902.58 ms |
| Embedded, earlier campaign average | 96.75 req/s | -- | -- |
| Remote without artifact transfer | 95.11 req/s | 668.25 ms | 969.35 ms |
| Remote with NIXL manual polling | 93.12 req/s | 684.05 ms | 1,106.94 ms |
| Remote with NIXL progress thread | 77.16 req/s | 831.32 ms | 1,238.12 ms |
| Initial remote implementation | 78.33 req/s | -- | -- |

The optimized remote result is 3.7% below the earlier embedded average and falls within
the observed embedded run-to-run throughput range. Compared with the initial remote
implementation, it improves throughput by approximately 18.9%.

## Remaining gap

- CPU time per request was approximately 90 ms for optimized remote versus 77 ms for
  embedded.
- Optimized remote p99 was approximately 1,107 ms versus 903 ms for the clean embedded
  reference.
- Per-request tensor descriptors, metadata serialization, receiver allocation, and
  1 ms completion polling remain in the remote path.

To target CPU and tail-latency parity, use a persistent registered NIXL ring or slab per
worker pair and event-driven completion. Requests would carry only a slot, offset,
shape, dtype, and byte count. Same-host shared memory could back such a ring, but merely
selecting SHM as the transport did not remove the per-request lifecycle overhead.

## Implementation locations

- `examples/custom_backend/user_ensemble/worker.py`: artifact fan-out and remote call.
- `components/src/dynamo/vllm/multimodal_utils/custom_encoder/artifact_transfer.py`:
  artifact export/import and manual-polling configuration.
- `components/src/dynamo/vllm/handlers.py`: remote artifact receive and prompt assembly.

## Benchmark note

AIPerf failed before issuing requests because its installed egg was corrupt. The final
comparison therefore used a closed-loop `aiohttp` harness with the same fixed workload
and a fresh server for each phase.
