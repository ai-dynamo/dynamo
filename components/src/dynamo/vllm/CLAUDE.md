# vLLM Component

Dynamo's vLLM backend. There are **two** ways Dynamo drives vLLM, and they
coexist:

1. **In-process** (`llm_engine.py` → `VllmLLMEngine`): the default. Imports
   `vllm`, instantiates `AsyncLLM` directly inside the Dynamo worker process.
   Shares lifecycle, signals, and the GPU process with the engine.
2. **Sidecar**: NEW. A separate Dynamo worker talks to a native vLLM engine
   process over the OpenEngine v1 gRPC contract. This worker is implemented in
   **Rust**, not Python — it lives in its own crate at
   **`dynamo/lib/vllm-sidecar/`** (a `[[bin]]` named `dynamo-vllm-sidecar`,
   sibling of `lib/backend-common/examples/mocker`), and implements the native
   Rust `dynamo_backend_common::LLMEngine` trait. It is **not** part of this
   Python component. See that crate's docs for its design.

The in-process `VllmLLMEngine` is a `dynamo.common.backend.LLMEngine` subclass
wired through `run(...)`. Read `dynamo/components/src/dynamo/common/backend/
CLAUDE.md` first — it owns the lifecycle, the request/response TypedDict
contract, and the **zero-duplication-across-engines** constraint.

## The boundary

| | In-process (`VllmLLMEngine`) | Sidecar (`dynamo-vllm-sidecar`) |
|---|---|---|
| Language | Python | **Rust** |
| Location | this dir (`llm_engine.py`) | `dynamo/lib/vllm-sidecar/` (separate crate) |
| Engine | `vllm.AsyncLLM` in-process | native vLLM engine over OpenEngine gRPC |
| Imports vllm? | yes | **no** — tonic OpenEngine client only |
| KV transport | NixlConnector (internal) | NixlConnector on the engine side; sidecar advertises sources |
| GPU | same process | separate process/container |

The sidecar exists to decouple Dynamo's container/runtime from vLLM's and to
let users keep the native vLLM serve UX. It is **not** a replacement for the
in-process path — do not delete or refactor `VllmLLMEngine` when working on the
sidecar. Because the sidecar is a standalone Rust crate, work on it does not
touch this Python component at all.

## In-process engine (`VllmLLMEngine`) — method map

`llm_engine.py`:

- `from_args(argv)` — parse vLLM `AsyncEngineArgs` + Dynamo args, build
  `WorkerConfig`, construct (not start) the engine.
- `start(worker_id)` — `AsyncLLM.from_vllm_config(...)`, return `EngineConfig`
  (model, context length, block size, total KV blocks). vLLM's KV transport is
  internal (NixlConnector) so `bootstrap_host/port` stay `None`.
- `generate(request, context)` — map `GenerateRequest` TypedDict →
  `TokensPrompt` + `SamplingParams`, iterate `engine_client.generate(...)`,
  yield `GenerateChunk`s. Disagg dispatch keys off
  `WorkerConfig.disaggregation_mode` (prefill caps 1 token + packs
  `kv_transfer_params` into terminal `disaggregated_params`; decode reads
  `prefill_result`).
- `kv_event_sources()` — when KV routing enabled, returns the ZMQ publisher
  sources.
- `component_metrics_dp_ranks()` / `attach_snapshot_publisher()` —
  per-iteration stat-logger (`_UnifiedStatLogger`) pushes `ComponentSnapshot`.
- `register_prometheus()` — bridges vLLM's `vllm:` registry.
- `abort()`, `health_check_payload()`, `cleanup()` (null-safe).

The Rust sidecar crate (`dynamo/lib/vllm-sidecar/`) mirrors this lifecycle
method-for-method against the Rust `LLMEngine` trait, but sources every value
from OpenEngine RPCs instead of in-process vLLM objects.

## Other files

`args.py`, `backend_args.py`, `handlers.py`, `main.py`, `worker_factory.py`,
`publisher.py`, `health_check.py`, etc. are the **legacy / parallel** entry
path (pre-`unified_main`). Per the backend module's design constraint, that
path stays untouched — the `unified_main.py` files are the separate,
current path.

## Tips for AI assistants

- **Read the backend ABC + CLAUDE.md before editing either engine.**
  `common/backend/engine.py` defines `LLMEngine`, `GenerateRequest`,
  `GenerateChunk`, `EngineConfig`.
- **Keep logging standardized** across vllm/sglang/trtllm (see backend
  CLAUDE.md "Logging").
- **Sidecar must not depend on `vllm`.** It lives in `dynamo/lib/vllm-sidecar/`
  (Rust) and may depend only on `dynamo-backend-common` + tonic/prost. If you
  find yourself reaching for a vLLM type there, the value belongs in an
  OpenEngine RPC response instead.
- **Real-engine tests on computelab/lyris** (see root `CLAUDE.md`). Local =
  unit tests + CPU fake-servicer only.
