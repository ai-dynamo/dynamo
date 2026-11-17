---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: NVTX Profiling
subtitle: Annotate the Dynamo frontend and workers on an Nsight Systems timeline to separate Dynamo time from engine time
---

Dynamo emits NVIDIA Tools Extension (NVTX) ranges and markers, so an Nsight Systems (`nsys`)
capture shows where time goes inside Dynamo and not only inside the inference engine. The Rust
runtime and the Python components are annotated separately and share one runtime switch,
`DYN_NVTX`. Both are off by default.

## Enable the Markers

Set `DYN_NVTX=1` on every process you want annotated, and trace NVTX in `nsys`:

```bash
DYN_NVTX=1 nsys profile --trace=nvtx,osrt -o frontend.nsys-rep \
  python3 -m dynamo.frontend --http-port 8000
```

```bash
DYN_NVTX=1 nsys profile --trace=cuda,nvtx -o worker.nsys-rep \
  python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B
```

When the switch takes effect, the Rust runtime logs `NVTX annotations enabled (DYN_NVTX)` at INFO
level. The line appears only if the build includes the `nvtx` feature and `DYN_LOG` lets INFO
messages through. If it is missing, check the log level first. If other INFO messages appear and
this one does not, the build does not include the feature; see
[Build Requirements](#build-requirements).

To profile on Kubernetes, see [Profiling with Nsight Operator](../../kubernetes/operations/nsight-operator-profiling.mdx).

## Environment Variable

| Variable | Default | Description |
|---|---|---|
| `DYN_NVTX` | unset (off) | Enables NVTX annotations in the Rust runtime and the Python components. Accepts `1`, `true`, `on`, or `yes` to enable and `0`, `false`, `off`, `no`, or empty to disable, trimmed and case-insensitive. Any other value is an error at startup, in both layers, rather than a silent "off". |

Since v1.7.0, `DYN_NVTX` also switches the Rust runtime's markers. It replaces
`DYN_ENABLE_RUST_NVTX`, which Dynamo no longer reads. Earlier releases read `DYN_NVTX` only in the
Python components, and accepted only `0` or `1`; any other value fails at import.

## Build Requirements

| Layer | Annotates | Requires |
|---|---|---|
| Rust runtime | Frontend HTTP handling, preprocessing, routing, transport, worker ingress and egress, the Python bridge | The `nvtx` Cargo feature, which the published `ai-dynamo-runtime` wheels and container images include |
| Python components | Multimodal loading, encoding, and embedding transfer in the frontend and in the vLLM and SGLang workers | The `nvtx` Python package: `pip install "ai-dynamo[profiling]"`, already in the container images |

The published wheels and images have compiled the Rust markers in since v1.6.0, so they need no
rebuild. With `DYN_NVTX` unset, the markers add almost no cost; see
[Profiling Overhead](#profiling-overhead).

If `DYN_NVTX` is on but the `nvtx` Python package is missing, the Python layer logs
`DYN_NVTX is set but the nvtx package is not installed; NVTX markers are disabled` and the
process starts without Python markers. The Rust markers are unaffected.

### Build From Source

To build the Python bindings with the markers, include `nvtx` in the feature list:

```bash
cd lib/bindings/python
maturin develop --uv --features nvtx
```

To build a Rust binary for profiling, use the `profiling` profile, which is a release build with
debug symbols:

```bash
cargo build --profile profiling --features dynamo-runtime/nvtx
```

> [!WARNING]
> Scope the Cargo feature to `dynamo-runtime/nvtx`. A bare `--features nvtx` at the workspace root
> also turns on the `kvbm-engine` markers, which ignore `DYN_NVTX` and fire whenever they are
> compiled in.

## Rust Ranges

Names are hierarchical, so a prefix filter in Nsight Systems isolates one stage. Ranges are
start/end ranges with explicit ids, so a range opened on one Tokio worker thread and closed on
another after an `.await` still renders as one bar.

### Frontend

| Range | Covers |
|---|---|
| `frontend.http.chat.handler` | The `/v1/chat/completions` handler, until it returns the response or the stream |
| `frontend.http.chat.validate` | Request normalization and field validation, including `engine_lookup` |
| `frontend.http.chat.engine_lookup` | Resolving the model to an engine and its parsing options |
| `frontend.http.chat.generate` | Preprocessing, routing, and the handoff to the worker, until the response stream opens |
| `frontend.http.chat.sse_chunk` | Converting one streamed chunk to a server-sent event (SSE) |
| `frontend.http.chat.aggregate` | Folding the stream into one response on the non-streaming path, including waits on the worker |
| `frontend.http.completions.*` | The same stages for `/v1/completions`, without `validate` |
| `preprocess.template` | Applying the chat template |
| `preprocess.tokenize` | Tokenizing the prompt |
| `route.select_worker` | Selecting a worker, with KV-aware routing |
| `route.kv_match` | Scoring KV cache overlap, with KV-aware routing |
| `transport.tcp.request`, `transport.nats.request` | Sending a request to the chosen worker and opening its response stream |
| `transport.tcp.send` | Writing the request to the worker |
| `transport.response.wait_backend` | Waiting for the worker to acknowledge the response stream |
| `detokenize` | Detokenizing one output token |

`frontend.http.chat.first_chunk` is an instantaneous marker on the first chunk that carries
generated tokens. Use it to read Time To First Token (TTFT) from the timeline. It skips the
annotation events and the role-only opening delta, which carry no generated token.

The `frontend.http.*` ranges and the `first_chunk` marker are newer than v1.7.0.

### Worker

| Range | Covers |
|---|---|
| `worker.ingress.request` | One request's whole lifetime in the worker |
| `worker.ingress.parse_request` | Decoding the request off the wire |
| `worker.ingress.open_response_stream` | Opening the response stream back to the frontend |
| `worker.ingress.generate` | Dispatching the request into the engine |
| `worker.egress.chunk` | Sending one response chunk, including `encode` and `publish` |
| `worker.egress.encode` | Serializing the chunk |
| `worker.egress.publish` | Writing the chunk to the frontend |
| `pybridge.invoke_generator` | Acquiring the GIL, building the Python request, and calling the handler's `generate`; ends before the generator is first advanced |
| `pybridge.anext_call` | The GIL-bound `__anext__` call, excluding its await |
| `pybridge.encode_response` | Parsing and serializing one Python response for the wire, on the direct request-plane path |
| `pybridge.decode_response` | Converting one Python response into a typed Rust value, on the typed path |

`worker.ingress.prologue_sent` is an instantaneous marker for the point the worker acknowledges the
response stream to the frontend. It is emitted only when that send succeeds, so a request that
progressed without it means the acknowledgement failed.

`pybridge.encode_response` is newer than v1.7.0. On earlier builds, the time it covers falls in
the gap before `worker.egress.chunk` and reads as engine time.

## Python Ranges

The Python components use `mm:*` and `mm_*` names on the multimodal paths. They are emitted into
the NVTX domain `dynamo`, while the Rust ranges use the default domain. In the Nsight Systems GUI
they appear as separate NVTX rows. To capture only one set, pass `--nvtx-domain-include=dynamo`
(Python ranges) or `--nvtx-domain-include=default` (Rust ranges) to `nsys profile`. These options
filter events during collection, so `nsys stats` cannot apply them to an existing report.

For the SGLang worker ranges, see [SGLang Multimodal](../../developer-guide/knowledge-base/modular-components/backends/sglang/multimodal.md#profiling).
The vLLM ranges are not listed yet. They include `mm_frontend:*` in the frontend's vLLM processor
(for example `mm_frontend:preprocess_chat`, `mm_frontend:process_inputs`, and
`mm_frontend:routed_engine_generate`), `mm_backend:build_prompt` in the vLLM worker, and the shared
multimodal loaders under `mm:img:*`, `mm:audio:*`, `mm:nixl:*`, and `mm:local:*`.

Python and Rust ranges from one process land on the same timeline, together with the engine's own
NVTX ranges, such as TensorRT-LLM's.

## Reading the Timeline

The bars are annotated Dynamo work. Everything else in a request's span falls in the gaps between
bars, and no gap is a clean reading of one component.

> [!WARNING]
> A gap is a mix of several costs. Attributing a whole gap to the engine, or to the worker,
> overstates it.
>
> - Between consecutive `worker.egress.chunk` ranges: the engine producing the next token, plus
>   polling the response stream, which includes the `pybridge.anext_call` bar and the
>   `pybridge.encode_response` or `pybridge.decode_response` bar.
> - Between consecutive `frontend.http.*.sse_chunk` ranges: worker time, the frontend receiving the
>   next response (including the `detokenize` bar), and, because an SSE range ends before the chunk
>   is yielded, time spent by the HTTP client and the network.
> - Inside `frontend.http.chat.generate`: dispatch of the prefill request in disaggregated serving,
>   which is not annotated.

Range names are static, so concurrent requests produce overlapping ranges with the same name.
Capture at low concurrency to follow one request from end to end.

## Profiling Overhead

Annotation is cheap when `DYN_NVTX` is unset and costs measurable CPU time when it is on:

| State | Cost per range |
|---|---|
| `DYN_NVTX` unset | One relaxed atomic load and a branch |
| `DYN_NVTX=1`, no profiler attached | About 0.1 µs |
| `DYN_NVTX=1`, under `nsys --trace=nvtx` | About 1.3 µs |

The figures were measured at 4096 concurrent streams on mock workers, so treat them as an
estimate on other hardware. The frontend emits about two ranges per streamed token (`detokenize` and `sse_chunk`), so at high
token rates the profiler shows up in frontend CPU time. Keep profiled runs out of measured
benchmarks.

At high token rates, `nsys` also writes raw trace data to `/tmp` at several times the size of the
final report. If that disk is slow or nearly full, write-back can stall the profiled processes for
seconds. To bound a capture, pass `--duration` or `--capture-range` to `nsys profile`, and keep
`/tmp` on fast storage with free space, for example a container started with `--tmpfs /tmp`.

## Related

- [Environment Variables](environment-variables.mdx) lists every other observability variable.
- [Request Trace Reference](request-traces.mdx) records per-request data for analysis that outlives a capture.
- [Forward Pass Metrics Traces](forward-pass-metrics-traces.mdx) times the engine's forward passes.
- [Profiling with Nsight Operator](../../kubernetes/operations/nsight-operator-profiling.mdx) covers capturing on Kubernetes.
