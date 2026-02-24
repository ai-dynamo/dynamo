# SGLang Component

Dynamo's SGLang backend wraps SGLang's inference engine (`sgl.Engine`) and diffusion
generator (`DiffGenerator`) behind Dynamo's distributed runtime. It handles model
registration, request routing, metrics, and disaggregated serving.

## Entry Point

`__main__.py` -> `main.py:main()` -> `main.py:worker()`

`worker()` parses args, creates the distributed runtime, installs graceful shutdown,
then dispatches to one of 10 init functions based on CLI flags:

```
args.py:parse_args() -> Config(server_args, dynamo_args)

Worker dispatch (main.py:60-132):
  --image-diffusion-worker    -> init_diffusion.init_image_diffusion()
  --video-generation-worker   -> init_diffusion.init_video_diffusion()
  --embedding-worker          -> init_embedding.init_embedding()
  --multimodal-processor      -> init_multimodal.init_multimodal_processor()
  --multimodal-encode-worker  -> init_multimodal.init_multimodal_encode_worker()
  --multimodal-worker         -> init_multimodal.init_multimodal_worker() or _prefill_worker()
  --dllm-algorithm <algo>     -> init_diffusion.init_llm_diffusion()
  (default, prefill mode)     -> init_llm.init_prefill()
  (default, decode/agg mode)  -> init_llm.init_decode()
```

## Config / Args

`args.py:parse_args()` is the main parsing function. It returns `Config(server_args, dynamo_args)`.

**Two config paths:**

1. **LLM workers** (decode, prefill, embedding, multimodal-worker, dllm): Creates full
   `sglang.srt.server_args.ServerArgs` via `ServerArgs.from_cli_args()`. This triggers
   model config loading, tokenizer detection, etc.

2. **Diffusion workers** (image, video): Creates a minimal `types.SimpleNamespace` stub
   (args.py:350-366) with only the fields needed for `DiffGenerator`. The stub does NOT
   have `max_running_requests`, `dllm_algorithm_config`, or other LLM-specific fields.
   Use `getattr()` when accessing fields that may not exist on the stub.

**DynamoConfig** combines `DynamoRuntimeConfig` (common flags like `--namespace`,
`--output-modalities`, `--media-output-fs-url`) with `DynamoSGLangConfig` (sglang-specific
flags like `--multimodal-processor`, `--embedding-worker`).

Key gotcha: `--output-modalities` defaults to `["text"]` globally. Image/video diffusion
workers override this in their init functions to `["image"]`/`["video"]` to ensure correct
registration with the Rust side.

## Handler Hierarchy

```
BaseGenerativeHandler (handler_base.py)
  Abstract base. Has config, publisher, tracing. No engine.
  Subclasses: ImageDiffusionWorkerHandler, VideoGenerationWorkerHandler

  BaseWorkerHandler (handler_base.py)
    Adds sgl.Engine, tokenizer, priority support, engine routes,
    cancellation, bootstrap (disagg), weight update APIs.
    Constructor accepts engine=None for encode-only workers.

    DecodeWorkerHandler (llm/decode_handler.py)
      Aggregated + disaggregated decode. Token/text streaming.

      DiffusionWorkerHandler (llm/diffusion_handler.py)
        LLM diffusion (DLLM). Simplified decode without disagg.

    PrefillWorkerHandler (llm/prefill_handler.py)
      Disaggregated prefill. Yields bootstrap info first, then consumes.

    EmbeddingWorkerHandler (embedding/embedding_handler.py)
      Uses engine.async_encode() instead of async_generate().

    MultimodalWorkerHandler (multimodal/worker_handler.py)
      Multimodal inference. Aggregated or disaggregated paths.
      Has EmbeddingsProcessor for NIXL-transferred image embeddings.

    MultimodalPrefillWorkerHandler (multimodal/worker_handler.py)
      Multimodal prefill phase. Yields bootstrap info.

    MultimodalProcessorHandler (multimodal/processor_handler.py)
      Front-facing. No engine. Routes to encode worker.

    MultimodalEncodeWorkerHandler (multimodal/encode_worker_handler.py)
      No engine. Uses MMEncoder from SGLang. NIXL for embeddings transfer.
```

## Engine Types by Worker

| Worker | Engine | Notes |
|--------|--------|-------|
| decode, prefill, dllm, embedding | `sgl.Engine` | Full SGLang inference engine |
| multimodal-worker, multimodal-prefill | `sgl.Engine` | Plus EmbeddingsProcessor |
| multimodal-processor | None | Tokenizer only, routes to encoder |
| multimodal-encode-worker | None | `MMEncoder` from SGLang |
| image-diffusion-worker | `DiffGenerator` | From `sglang.multimodal_gen` |
| video-generation-worker | `DiffGenerator` | From `sglang.multimodal_gen` |

`DiffGenerator.generate()` returns `GenerationResult | list[GenerationResult] | None`
(dataclass, NOT a dict). Access `result.frames` for images/video frames,
`result.samples` for raw tensors.

## Registration

`register.py` has three paths:

1. **LLM** (`register_model_with_readiness_gate`): Builds `ModelRuntimeConfig` with
   bootstrap info, scheduler stats, parser configs. Calls Rust `register_model()` which
   downloads `config.json` + tokenizer from HuggingFace.

2. **Image diffusion** (`register_image_diffusion_model`): Calls `register_model()` with
   `ModelType.Images`. The Rust side skips HF downloads for Images/Videos/Tensor types
   (lib/bindings/python/rust/lib.rs:314) and uses `ModelDeploymentCard::with_name_only()`.

3. **Video generation** (`register_video_generation_model`): Same fast path with
   `ModelType.Videos`.

## Init Flow (typical LLM decode)

```
init_decode():
  engine = sgl.Engine(server_args)
  handler = DecodeWorkerHandler(engine, config, publisher, endpoint, shutdown_event)
  handler.register_engine_routes(runtime)  # profiling, weight updates, memory mgmt
  setup_sgl_metrics(engine, config, endpoint)  # Prometheus + KV events via ZMQ
  asyncio.gather(
    endpoint.serve_endpoint(handler.generate, ...),
    register_model_with_readiness_gate(engine, endpoint, ...),
  )
```

## Disaggregated Serving

Prefill and decode workers coordinate via a bootstrap mechanism:

1. **Prefill handler** generates a `bootstrap_room` (random 63-bit ID)
2. Prefill yields bootstrap info (host, port, room) as its first response
3. **Decode handler** receives bootstrap info, passes it to `engine.async_generate()`
4. SGLang transfers KV cache via NIXL/RDMA between workers

Key functions: `BaseWorkerHandler._get_bootstrap_info()`,
`BaseWorkerHandler._generate_bootstrap_room()`.

## Metrics & Publishing

`publisher.py:DynamoSglangPublisher` manages:
- **Scheduler metrics**: Received via ZMQ from SGLang's scheduler, published to Prometheus
- **KV events**: ZMQ subscribers per DP rank, forwarded via `KvEventPublisher`

Only leader nodes (node_rank==0) run the metrics loop. Non-leader nodes just wait.

`setup_sgl_metrics()` returns `(publisher, metrics_task, metrics_labels)`.

## Graceful Shutdown

`shutdown.py:install_graceful_shutdown()` monkey-patches `loop.add_signal_handler()` to
capture SGLang's internal signal registrations and defer them. On SIGTERM/SIGINT:
1. Unregisters from discovery (stops new requests)
2. Waits grace period for in-flight requests
3. Runs deferred SGLang signal handlers

## Request Flow

```
Frontend (Rust, lib/llm/)
  -> Preprocessor (tokenizes, builds PreprocessedRequest with token_ids + sampling + stop)
  -> Dynamo RPC to endpoint (dyn://{namespace}.{component}.{endpoint})
  -> Python handler.generate(request_dict, context)
       handler._build_sampling_params(request) -> SGLang-native params
       engine.async_generate(**params) -> async iterator of dicts
       handler yields {token_ids, text, finish_reason, ...} back to frontend
  -> Frontend postprocesses into OpenAI-compatible response
```

Two request formats depending on `--skip-tokenizer-init`:
- **Token-based** (skip_tokenizer_init=True): Frontend tokenizes. Request has `token_ids`,
  `sampling_options`, `stop_conditions`. Handler maps to SGLang params.
- **Text-based** (skip_tokenizer_init=False): SGLang tokenizes. Request is an OpenAI
  `ChatCompletionRequest`. Only `/v1/chat/completions` available.

Image/video diffusion handlers receive the full OpenAI-format request dict directly
(not preprocessed), since the frontend passes through diffusion requests without
tokenization.

## Health Checks

Each worker type has a custom health check payload (`health_check.py`):
- **Decode/Aggregated**: `SglangHealthCheckPayload` -- sends BOS token, expects 1 token back
- **Prefill (disagg)**: `SglangPrefillHealthCheckPayload` -- wrapped `{request, sampling_params}`
- **Image diffusion**: `ImageDiffusionHealthCheckPayload` -- 512x512, 1 inference step, b64_json
- **Video generation**: `VideoGenerationHealthCheckPayload` -- 256x256, 8 frames, 1 step, b64_json

Health check payloads can be overridden via `DYNAMO_HEALTH_CHECK_PAYLOAD` env var (JSON).

## Launch Scripts

Examples in `examples/backends/sglang/launch/`. Each script starts a frontend + worker(s)
in one terminal. GPU requirements are documented in script headers.

```
agg.sh              # 1 GPU  - Single aggregated worker
agg_embed.sh        # 1 GPU  - Embedding model
agg_vision.sh       # 1 GPU  - Multimodal (vision + LLM)
agg_router.sh       # 2 GPUs - Two workers behind KV-aware router
disagg.sh           # 2 GPUs - Prefill + decode on separate GPUs
disagg_router.sh    # 4 GPUs - 2 prefill + 2 decode with KV routing
disagg_same_gpu.sh  # 1 GPU  - Both workers on single GPU (16+ GB VRAM)
multimodal_epd.sh   # 2 GPUs - Encoder + PD worker
multimodal_disagg.sh # 3 GPUs - Encoder + prefill + decode
diffusion_llada.sh  # 1 GPU  - Diffusion language model
image_diffusion.sh  # 1 GPU  - Text-to-image (~38 GB VRAM for FLUX.1-dev)
text-to-video-diffusion.sh  # 1-2 GPUs - Text-to-video (Wan2.1)
```

## Common Pitfalls

- **SimpleNamespace vs ServerArgs**: Image/video diffusion workers use SimpleNamespace
  stubs. Always use `getattr(server_args, field, default)` for fields that may not exist.
- **engine=None**: Multimodal processor and encode worker pass `engine=None` to
  BaseWorkerHandler. Any code in the base class that touches engine must guard with
  `if engine is not None`.
- **GenerationResult is a dataclass**: SGLang 0.5.9 changed `DiffGenerator.generate()`
  to return `GenerationResult` (not a dict). Use `result.frames`, not `result["frames"]`.
- **output_modalities default**: Global default is `["text"]`. Image/video diffusion
  workers must override to `["image"]`/`["video"]` or the Rust registration path tries
  to load `config.json` (which doesn't exist for diffusers models).
- **Zombie GPU processes**: `sgl_diffusion::scheduler` spawns a child process that
  survives parent kill. Always check `nvidia-smi` after teardown.
- **CuDNN check**: Set `SGLANG_DISABLE_CUDNN_CHECK=1` if PyTorch's CuDNN is older than
  what SGLang expects. Affects vision and diffusion models with Conv3d.

## Adding a New Worker Type

Checklist for adding a new worker (e.g., a new modality or serving mode):

1. **CLI flag**: Add to `backend_args.py` (DynamoSGLangConfig) and parse in `args.py`
2. **Init function**: Create `init_<type>.py` with `init_<type>(config, runtime)` that:
   - Creates the engine (sgl.Engine, DiffGenerator, or None for encode-only)
   - Creates the handler
   - Sets up metrics (`setup_sgl_metrics` if applicable)
   - Calls `endpoint.serve_endpoint(handler.generate, ...)`
   - Registers the model
3. **Handler**: Subclass `BaseWorkerHandler` (if engine-backed) or `BaseGenerativeHandler`
   (if no engine). Implement `async generate(request, context) -> AsyncGenerator`
4. **Registration**: Add a function in `register.py`. Choose the right `ModelType`:
   - `Chat | Completions` for LLM (Rust downloads config.json + tokenizer)
   - `Images`, `Videos`, `Tensor` for non-LLM (Rust skips HF downloads)
5. **Health check**: Add a payload class in `health_check.py`
6. **Dispatch**: Add the flag check in `main.py:worker()` dispatch block
7. **output_modalities**: If not text, override in the init function (default is `["text"]`)
8. **Launch script**: Add to `examples/backends/sglang/launch/` with GPU count in header

## Tips for AI Assistants

- **Read before editing**: Always read handler_base.py and the relevant init_*.py before
  modifying handler or registration code. The inheritance chain matters.
- **Test with launch scripts**: The fastest way to validate changes is to run the
  corresponding launch script in `examples/backends/sglang/launch/`.
- **Kill zombies between tests**: `pkill -9 -f sglang; sleep 3` before relaunching.
  Diffusion workers spawn child processes (`sgl_diffusion::scheduler`) that survive kills.
- **Check nvidia-smi**: If a launch OOMs, check for orphaned GPU processes from prior runs.
- **SimpleNamespace stubs**: When touching args.py or code that reads server_args, always
  use `getattr(server_args, field, default)` -- image/video workers don't have full ServerArgs.
- **engine can be None**: Encode-only workers (multimodal-processor, multimodal-encode-worker)
  pass engine=None. Guard any engine access in shared base class code.
- **Rebuild after Rust changes**: If changing registration (register.py interacts with Rust
  bindings), rebuild: `cd lib/bindings/python && maturin develop --uv && cd <root> && uv pip install -e .`
- **CuDNN errors on vision/diffusion**: Set `SGLANG_DISABLE_CUDNN_CHECK=1` before launching.

## File Index

```
sglang/
  __main__.py              # Entry point
  main.py                  # Worker dispatch
  args.py                  # Config parsing (ServerArgs vs SimpleNamespace)
  backend_args.py          # Dynamo-specific SGLang CLI flags
  init_llm.py              # init_decode(), init_prefill()
  init_diffusion.py        # init_llm_diffusion(), init_image_diffusion(), init_video_diffusion()
  init_multimodal.py       # init_multimodal_{processor,encode_worker,worker,prefill_worker}()
  init_embedding.py        # init_embedding()
  register.py              # Model registration (LLM, image, video)
  publisher.py             # Metrics + KV event publishing
  protocol.py              # Request/response Pydantic models
  health_check.py          # Health check payloads per worker type
  shutdown.py              # Graceful shutdown with deferred signal handling
  request_handlers/
    handler_base.py        # BaseGenerativeHandler, BaseWorkerHandler
    llm/
      decode_handler.py    # DecodeWorkerHandler (agg + disagg)
      prefill_handler.py   # PrefillWorkerHandler (disagg prefill)
      diffusion_handler.py # DiffusionWorkerHandler (DLLM)
    embedding/
      embedding_handler.py # EmbeddingWorkerHandler
    image_diffusion/
      image_diffusion_handler.py  # ImageDiffusionWorkerHandler (DiffGenerator)
    video_generation/
      video_generation_handler.py # VideoGenerationWorkerHandler (DiffGenerator)
    multimodal/
      processor_handler.py       # MultimodalProcessorHandler (no engine)
      encode_worker_handler.py   # MultimodalEncodeWorkerHandler (MMEncoder)
      worker_handler.py          # MultimodalWorkerHandler + PrefillWorkerHandler
```
