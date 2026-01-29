# Worklog: Video Generation Support (T2V/I2V)

**Branch**: `ishan/video`
**PR**: #5793 (draft)
**Date**: 2026-01-29
**Base Pattern**: Image Diffusion PR #5609

---

## Objective

Add video generation support to Dynamo, enabling text-to-video (T2V) and image-to-video (I2V) generation using SGLang's `DiffGenerator` with Wan video models. Exposes an OpenAI-compatible `/v1/videos/generations` HTTP endpoint.

## Architecture

```
Client (curl/SDK)
  |
  v
Frontend (Rust/Axum)  -- /v1/videos/generations endpoint
  |
  v
Runtime (etcd discovery, TCP request plane)
  |
  v
Video Worker (Python)
  |-- VideoGenerationWorkerHandler
  |     |-- SGLang DiffGenerator (DiffusionPipeline)
  |     |-- imageio (frames -> MP4)
  |     |-- fsspec (upload to storage)
  v
Response (MP4 URL or base64)
```

## Files Changed (22 files, +1154/-9)

### Python Layer

| File | Change |
|------|--------|
| `components/src/dynamo/sglang/protocol.py` | Added `CreateVideoRequest`, `VideoData`, `VideoGenerationResponse` Pydantic models |
| `components/src/dynamo/sglang/args.py` | Added `--video-generation-worker` and `--video-generation-fs-url` CLI args + dataclass fields |
| `components/src/dynamo/sglang/request_handlers/video_generation/__init__.py` | NEW: module init, exports `VideoGenerationWorkerHandler` |
| `components/src/dynamo/sglang/request_handlers/video_generation/video_generation_handler.py` | NEW: 375-line handler -- DiffGenerator invocation, frame-to-MP4 conversion (imageio), fsspec upload, base64 encoding |
| `components/src/dynamo/sglang/request_handlers/__init__.py` | Added video generation handler export |
| `components/src/dynamo/sglang/health_check.py` | Added `VideoGenerationHealthCheckPayload` class |
| `components/src/dynamo/sglang/register.py` | Added `register_video_generation_model()` using `ModelType.Videos` + `ModelInput.Text` |
| `components/src/dynamo/sglang/main.py` | Added `init_video_generation()` (~80 lines), worker dispatch branch for `video_generation_worker` |

### Rust Layer

| File | Change |
|------|--------|
| `lib/llm/src/model_type.rs` | Added `Videos = 1 << 6`, `supports_videos()`, updated `as_vec()`, `units()`, `as_endpoint_types()` |
| `lib/llm/src/endpoint_type.rs` | Added `Videos` variant, updated `as_str()` and `all()` |
| `lib/llm/src/protocols/openai.rs` | Added `pub mod videos;` |
| `lib/llm/src/protocols/openai/videos.rs` | NEW: `NvCreateVideoRequest`, `VideoData`, `NvVideosResponse` |
| `lib/llm/src/protocols/openai/videos/aggregator.rs` | NEW: `DeltaAggregator` for video response streaming |
| `lib/llm/src/protocols/openai/videos/nvext.rs` | NEW: `NvExt` struct for NVIDIA extensions |
| `lib/llm/src/types.rs` | Added `OpenAIVideosUnaryEngine` and `OpenAIVideosStreamingEngine` type aliases |
| `lib/llm/src/http/service/openai.rs` | Added `videos()` handler and `videos_router()` |
| `lib/llm/src/http/service/service_v2.rs` | Added `videos_endpoints_enabled` to `StateFlags`, integrated `videos_router` |
| `lib/llm/src/discovery/model_manager.rs` | Added `videos_engines` field, `add_videos_model()`, `remove_videos_model()`, `get_videos_engine()`, `list_videos_models()` |
| `lib/llm/src/discovery/watcher.rs` | Added video model discovery in `handle_put()` and `handle_delete()` |
| `lib/llm/src/http/service/metrics.rs` | Added `Videos` variant to `Endpoint` enum |

### Bindings

| File | Change |
|------|--------|
| `lib/bindings/python/rust/lib.rs` | Added `Videos` classattr to PyO3 `ModelType`, added `is_videos` tokenizer skip check |
| `lib/bindings/python/src/dynamo/_core.pyi` | Added `Videos: ModelType` to type stub |

## Bugs Found and Fixed During E2E Testing

1. **Missing `ModelType.Videos` Python binding** -- `register_video_generation_model` called `ModelType.Videos` which didn't exist in PyO3. Fixed by adding `#[classattr] const Videos` to `lib.rs` and updating `_core.pyi`.

2. **Tokenizer extraction failure for VAE/diffusion models** -- `register_llm` tried to load `tokenizer.json` from the video model directory (no tokenizer exists for diffusion models). Fixed by adding `is_videos` to the tokenizer skip condition in `lib.rs`.

3. **`sgl.Engine` import in health_check.py** -- Intermittent `AttributeError` when SGLang's lazy imports run in subprocess contexts. Pre-existing issue, not introduced by this PR.

## Testing

### SGLang Standalone (passed)
```bash
cd ~/sglang
python -c "
from sglang.srt.entrypoints.engine import DiffGenerator
gen = DiffGenerator(model_cfg='Wan-AI/Wan2.1-T2V-1.3B-Diffusers', tp=2)
out = gen.generate(prompt='A curious raccoon exploring a garden', num_frames=17, height=480, width=832)
print(type(out), out['frames'][0].shape)
gen.shutdown()
"
```
Result: 17 frames at 480x832, ~8s warm generation.

### Dynamo E2E (passed)
```bash
# Terminal 1: Start worker
cd ~/dynamo
python -c "
import dynamo.sglang.main as m
import sys
sys.argv = ['', '--model', 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
            '--video-generation-worker', '--video-generation-fs-url', 'file:///tmp/dynamo_videos',
            '--tp', '2']
import asyncio
asyncio.run(m.worker())
"

# Terminal 2: Send request
curl http://localhost:8099/v1/videos/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A curious raccoon exploring a garden","model":"Wan-AI/Wan2.1-T2V-1.3B-Diffusers","seconds":1,"fps":8,"num_frames":17,"size":"832x480","num_inference_steps":50,"response_format":"b64_json"}'
```
Result: HTTP 200, valid MP4 video returned as base64.

## Dependencies

SGLang diffusion extras required on the worker:
```bash
cd ~/sglang && uv pip install -e "python[diffusion]"
```
Installs: diffusers, imageio, moviepy, opencv, remote-pdb, st_attn, vsa, etc.

## Next Steps

- [ ] Add I2V support (image-to-video via `input_reference` field)
- [ ] Add progress streaming for long video generation
- [ ] Add example YAML configs for video generation deployment
- [ ] Add integration tests
- [ ] Review and merge after team feedback on PR #5793
