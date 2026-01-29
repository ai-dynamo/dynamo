# Video Generation Support -- Implementation Deep Dive

**Branch**: `ishan/video`
**PR**: #5793 (draft), built on image diffusion pattern from #5609
**Date**: 2026-01-29

---

## What This PR Does

Adds a `/v1/videos/generations` HTTP endpoint to Dynamo that accepts a text prompt and returns a generated video (MP4). Under the hood, it uses SGLang's `DiffGenerator` class to run a Wan diffusion model (e.g. `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`). Currently supports text-to-video (T2V) only; image-to-video (I2V) is a follow-up.

## How a Request Flows End-to-End

Here's what happens when you curl the endpoint:

```
1. curl POST /v1/videos/generations
       { "prompt": "A rocket ship", "model": "Wan-AI/...", "num_frames": 17 }
                |
                v
2. Rust Frontend (Axum HTTP server)
       - openai.rs:videos() handler receives the JSON, deserializes into NvCreateVideoRequest
       - Looks up the model in ModelManager.videos_engines (populated at startup via etcd)
       - Calls engine.generate(request) which sends the request over Dynamo's TCP request plane
                |
                v
3. Dynamo Runtime (etcd + TCP)
       - The video worker registered itself at startup via register_video_generation_model()
       - etcd stores: "this model exists at this endpoint, type = Videos"
       - The watcher (watcher.rs) saw the registration and created a PushRouter engine for it
       - The request plane routes the request to the Python worker
                |
                v
4. Python Video Worker (VideoGenerationWorkerHandler.generate())
       - Receives request dict, validates into CreateVideoRequest pydantic model
       - Calculates num_frames (fps * seconds if not explicit)
       - Calls DiffGenerator.generate() in a thread pool (to not block asyncio)
       - DiffGenerator runs the Wan diffusion pipeline (denoising loop on GPU)
       - Returns dict with "frames": list of numpy arrays, each shape (H, W, 3) uint8
                |
                v
5. Frame-to-MP4 Conversion (_frames_to_video)
       - Takes the list of numpy frame arrays
       - Uses imageio to encode them into an MP4 in-memory (BytesIO)
       - Codec: libx264, pixel format: yuv420p
       - Returns raw MP4 bytes
                |
                v
6. Storage / Response
       - If response_format="url": writes MP4 bytes to disk via fsspec, returns file:// URL
       - If response_format="b64_json": base64-encodes the MP4 bytes, returns inline
       - Wraps in VideoGenerationResponse and yields back through the runtime
                |
                v
7. Back in Rust
       - The Axum handler collects the streamed response via DeltaAggregator
       - Returns JSON to the client with the video URL or base64 data
```

## The Python Side -- What Each File Does

### `main.py` -- Worker Startup (`init_video_generation`)

This is the entry point. When you pass `--video-generation-worker`, the `worker()` function in main.py dispatches to `init_video_generation()`. Here's what it does:

1. **Creates the DiffGenerator** (line 540):
   ```python
   from sglang.multimodal_gen import DiffGenerator
   generator = DiffGenerator.from_pretrained(model_path=server_args.model_path)
   ```
   This loads the Wan model weights onto GPU(s). DiffGenerator handles tensor parallelism internally -- pass `--tp 2` and it shards across 2 GPUs.

2. **Creates the fsspec filesystem** (line 562):
   ```python
   fs = fsspec.filesystem(protocol, auto_mkdir=True)
   ```
   This is the storage backend. `file://` for local disk, but also supports `s3://`, `gs://`, `az://`.

3. **Wires up the Dynamo endpoint** (lines 577-603):
   ```python
   component = runtime.namespace(...).component(...)
   generate_endpoint = component.endpoint(...)
   handler = VideoGenerationWorkerHandler(component, generator, config, fs=fs)
   await asyncio.gather(
       generate_endpoint.serve_endpoint(handler.generate, ...),
       register_video_generation_model(generator, generate_endpoint, ...),
   )
   ```
   Two things happen concurrently:
   - `serve_endpoint` starts listening for requests on the TCP request plane
   - `register_video_generation_model` writes the model info to etcd so the frontend can discover it

### `video_generation_handler.py` -- The Core Handler

This is where the actual work happens. Key methods:

**`generate(request, context)`** -- The main entry point called by Dynamo runtime for each request. It:
- Parses the request into `CreateVideoRequest`
- Calls `_generate_video()` to produce raw MP4 bytes
- Either uploads to fsspec (`response_format="url"`) or base64-encodes (`response_format="b64_json"`)
- Yields a single `VideoGenerationResponse` dict

**`_generate_video(...)`** -- Calls SGLang's DiffGenerator:
```python
result = await asyncio.to_thread(
    self.generator.generate,
    sampling_params_kwargs=args,
)
frames = result.get("frames", [])
video_bytes = await self._frames_to_video(frames, fps)
```
Key detail: `generator.generate()` is a blocking call (runs the full diffusion denoising loop). We run it in `asyncio.to_thread()` so it doesn't block the event loop. The result is a dict with a `"frames"` key containing a list of numpy arrays.

**`_frames_to_video(frames, fps)`** -- Encodes frames to MP4:
```python
with imageio.get_writer(buffer, format="mp4", fps=fps, codec="libx264",
                         output_params=["-pix_fmt", "yuv420p"]) as writer:
    for frame in np_frames:
        writer.append_data(frame)
```
Uses `imageio-ffmpeg` under the hood. The `yuv420p` pixel format ensures broad compatibility.

**`_upload_to_fs(video_bytes, user_id, request_id)`** -- Saves the MP4:
```python
full_path = f"{self.root_path}/{request_id}.mp4"
await asyncio.to_thread(self.fs.pipe, full_path, video_bytes)
```
Files are saved as `{request_id}.mp4` flat in the configured directory.

### `protocol.py` -- Request/Response Types

Pydantic models that define the API contract:

```python
class CreateVideoRequest(BaseModel):
    prompt: str                          # Required: what to generate
    model: str                           # Required: which model
    seconds: Optional[int] = 4           # Duration
    fps: Optional[int] = 24              # Frame rate
    num_frames: Optional[int] = None     # Explicit frame count (overrides fps*seconds)
    size: Optional[str] = "832x480"      # WxH
    num_inference_steps: Optional[int] = 50  # Denoising steps (quality vs speed)
    guidance_scale: float = 5.0          # CFG scale
    response_format: Optional[str] = "url"   # "url" or "b64_json"
```

### `register.py` -- Model Registration

```python
async def register_video_generation_model(generator, endpoint, server_args, ...):
    await register_llm(ModelInput.Text, ModelType.Videos, endpoint, model_name, model_name)
```
This writes to etcd: "model X is available at this endpoint, accepts Text input, produces Videos output". The Rust frontend watcher picks this up and creates a routing engine for it.

### `health_check.py` -- Health Check Payload

A minimal request payload used by Dynamo's readiness probes:
```python
class VideoGenerationHealthCheckPayload(HealthCheckPayload):
    def __init__(self, model_path):
        self.default_payload = {
            "prompt": "test",
            "model": model_path,
            "num_frames": 8,
            "size": "256x256",
            "num_inference_steps": 1,  # Fast: just 1 step
        }
```

### `args.py` -- CLI Arguments

Two new flags:
- `--video-generation-worker`: Tells main.py to run `init_video_generation` instead of a normal LLM worker
- `--video-generation-fs-url`: Where to store generated videos (e.g. `file:///home/ubuntu/dynamo/videos`)

## The Rust Side -- What Each File Does

The Rust side handles HTTP routing, model discovery, and request forwarding. It doesn't do any video generation itself -- it just routes requests to the Python worker.

### `model_type.rs` -- ModelType Bitflag

```rust
const Videos = 1 << 6;  // New bit in the ModelType bitflags
```
This is how Dynamo categorizes models. When the Python worker registers with `ModelType.Videos`, the Rust layer knows to route `/v1/videos/generations` requests to it. Other existing types: `Chat`, `Completions`, `Embeddings`, `Tensor`, `Images`.

### `endpoint_type.rs` -- EndpointType Enum

```rust
Videos,  // Maps ModelType::Videos -> EndpointType::Videos
```
Used in the HTTP layer to decide which router to mount.

### `videos.rs` (protocol) -- Rust Request/Response Types

Mirror of the Python Pydantic models, but in Rust with serde:
```rust
pub struct NvCreateVideoRequest {
    pub prompt: String,
    pub model: String,
    pub num_frames: Option<i32>,
    // ... etc
}

pub struct NvVideosResponse {
    pub id: String,
    pub data: Vec<VideoData>,
    pub inference_time_s: Option<f64>,
    // ... etc
}
```
The Axum handler deserializes the incoming JSON into `NvCreateVideoRequest`, sends it through the runtime, and gets back `NvVideosResponse`.

### `openai.rs` (HTTP handler) -- The `/v1/videos/generations` Endpoint

```rust
async fn videos(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateVideoRequest>,
) -> Result<Response, ErrorResponse> {
    // 1. Look up model in ModelManager
    let engine = state.manager().get_videos_engine(&model)?;
    // 2. Send request through the runtime to the Python worker
    let stream = engine.generate(request).await?;
    // 3. Collect the response (videos don't stream tokens)
    let response = NvVideosResponse::from_annotated_stream(stream).await?;
    // 4. Return JSON
    Ok(Json(response).into_response())
}
```

### `model_manager.rs` -- Engine Management

Stores the routing engine for video models:
```rust
videos_engines: RwLock<ModelEngines<OpenAIVideosStreamingEngine>>,
```
Methods: `add_videos_model()`, `remove_videos_model()`, `get_videos_engine()`, `list_videos_models()`. These are called by the watcher when models register/deregister via etcd.

### `watcher.rs` -- Model Discovery

Watches etcd for model registration events. When it sees a model with `ModelType::Videos`:
```rust
if model_type.supports_videos() {
    let engine = PushRouter::new(...);
    manager.add_videos_model(model_name, engine);
}
```
This creates a `PushRouter` engine that forwards requests over the TCP request plane to the Python worker.

### `service_v2.rs` -- Router Assembly

```rust
if state_flags.videos_endpoints_enabled {
    let (docs, router) = videos_router(state.clone(), endpoint.path);
    all_docs.extend(docs);
    routers.push(router);
}
```
Only mounts the `/v1/videos/generations` route if a video model is registered.

### Python Bindings (`lib.rs`)

Two critical changes:
1. Expose `ModelType.Videos` to Python so `register_video_generation_model` can reference it
2. Skip tokenizer loading for video models (diffusion models don't have tokenizers):
   ```rust
   let is_videos = model_type.inner.supports_videos();
   if is_tensor_based || is_images || is_videos {
       // skip tokenizer extraction
   }
   ```

## How to Test

```bash
# Install SGLang diffusion deps
cd ~/sglang && uv pip install -e "python[diffusion]"

# Rebuild Dynamo bindings
cd ~/dynamo/lib/bindings/python && maturin develop --uv

# Launch (1.3B model, single GPU)
bash examples/backends/sglang/launch/t2v.sh --wan-size 1b

# Or 14B model (TP=2, both GPUs)
bash examples/backends/sglang/launch/t2v.sh --wan-size 14b

# Test
curl http://localhost:8000/v1/videos/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A rocket ship taking off into space",
    "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "num_frames": 17,
    "size": "832x480",
    "num_inference_steps": 50,
    "response_format": "url"
  }'
```

## Known Limitations

- **No streaming**: The client blocks until the full video is generated (~15s for 1.3B, longer for 14B). DiffGenerator doesn't expose progress callbacks.
- **No I2V**: The `input_reference` field exists in the protocol but is not wired up yet. Follow-up PR.
- **Local file URLs**: When `response_format="url"` with `file://`, the returned URL is only accessible on the same machine. For production, use S3/GCS.

## Dependencies

SGLang diffusion extras (installed via `uv pip install -e "python[diffusion]"`):
- `diffusers` -- Hugging Face diffusion pipeline
- `imageio` + `imageio-ffmpeg` -- Frame-to-MP4 encoding
- `fsspec` -- Filesystem abstraction for storage
- `torch` -- GPU compute
