#!/usr/bin/env python3
"""Helios video generation with latent cache continuation via Dynamo runtime.

Architecture:
    HTTP Client -> aiohttp Server -> Dynamo RPC -> Helios Worker (sglang pipeline on GPU)

Endpoints:
    POST /v1/videos/generate   - Generate video, save latent cache
    POST /v1/videos/continue   - Continue from cached latents
    GET  /v1/caches            - List cached latent states
    DELETE /v1/caches/{id}     - Delete a cached state
    GET  /health               - Health check
    GET  /videos/{filename}    - Serve video files

Usage:
    # Worker (GPU):
    MODE=worker CUDA_VISIBLE_DEVICES=0 python helios_continuation_dynamo.py

    # HTTP Server (no GPU):
    MODE=server python helios_continuation_dynamo.py
"""

import asyncio
import json
import logging
import os
import shutil
import time
import uuid

import torch
import uvloop
from pydantic import BaseModel

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker

logger = logging.getLogger(__name__)

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", "BestWishYsh/Helios-Distilled")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8090"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(SCRIPT_DIR, "output"))
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(SCRIPT_DIR, "latent_caches"))


# ---------- Protocol ----------

class GenerateRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    height: int = 384
    width: int = 640
    num_frames: int = 33
    seed: int = 42
    request_id: str = ""
    # If cache_id is set, this is a continuation request
    cache_id: str = ""


class VideoResponse(BaseModel):
    video_path: str = ""
    cache_id: str = ""
    num_frames: int = 0
    generation_time_s: float = 0.0
    peak_memory_gb: float = 0.0
    error: str = ""


def _count_output_frames(result):
    """Extract number of video frames from pipeline result (OutputBatch or Req)."""
    out = getattr(result, "output", None)
    if out is None or not hasattr(out, "dim"):
        return 0
    # Decoded video: [B, C, T, H, W] or [C, T, H, W]
    if out.dim() == 5:
        return out.shape[2]
    elif out.dim() == 4:
        return out.shape[1]
    return 0


# ---------- Worker (direct pipeline access) ----------

class HeliosContinuationWorker:
    """Loads the sglang Helios pipeline directly (no DiffGenerator multiprocessing)
    to enable full control over Req construction and latent caching."""

    def __init__(self):
        self.pipeline = None
        self.server_args = None

    def load_model(self):
        """Build pipeline directly, then patch denoising stage for caching."""
        from helios_latent_cache import patch_denoising_stage

        from sglang.multimodal_gen.runtime.pipelines_core import build_pipeline
        from sglang.multimodal_gen.runtime.server_args import (
            ServerArgs,
            set_global_server_args,
        )

        logger.info("Building Helios pipeline directly: %s", MODEL_PATH)
        t0 = time.time()

        # Create ServerArgs matching DiffGenerator.from_pretrained() defaults
        self.server_args = ServerArgs.from_kwargs(
            model_path=MODEL_PATH,
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
            vae_cpu_offload=True,
            output_path=OUTPUT_DIR,
        )

        # Must set global server args before building pipeline (stages read it)
        set_global_server_args(self.server_args)

        # Initialize distributed environment (single GPU)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        torch.cuda.set_device(0)

        from sglang.multimodal_gen.runtime.distributed import (
            maybe_init_distributed_environment_and_model_parallel,
        )
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=self.server_args.tp_size,
            enable_cfg_parallel=self.server_args.enable_cfg_parallel,
            ulysses_degree=self.server_args.ulysses_degree,
            ring_degree=self.server_args.ring_degree,
            sp_size=self.server_args.sp_degree,
            dp_size=self.server_args.dp_size,
        )

        # Build the pipeline (loads all models)
        self.pipeline = build_pipeline(self.server_args)

        # Patch the denoising stage for cache support
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.helios_denoising import (
            HeliosChunkedDenoisingStage,
        )
        for stage in self.pipeline.stages:
            if isinstance(stage, HeliosChunkedDenoisingStage):
                patch_denoising_stage(stage)
                break

        elapsed = time.time() - t0
        logger.info(
            "Helios pipeline built in %.1fs — VRAM: %.1f GB",
            elapsed, torch.cuda.memory_allocated() / 1e9,
        )

    def _build_req(self, prompt, negative_prompt="", height=384, width=640,
                   num_frames=33, seed=42, request_id=None, extra=None):
        """Construct a Req suitable for pipeline.forward()."""
        from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
        from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.server_args.model_path,
            server_args=self.server_args,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            request_id=request_id or str(uuid.uuid4())[:8],
            output_file_name=f"{request_id or str(uuid.uuid4())[:8]}.mp4",
        )
        sampling_params._set_output_file_name()
        req = prepare_request(server_args=self.server_args, sampling_params=sampling_params)

        if extra:
            req.extra.update(extra)
        return req

    def _run_pipeline(self, req):
        """Call pipeline.forward() and return (result_req, output_batch)."""
        result = self.pipeline.forward(req, self.server_args)
        return result

    def _save_video(self, result, request_id):
        """Save video from pipeline output tensor and return the path.

        pipeline.forward() returns either:
        - OutputBatch with .output tensor (decoded frames in [0,1], shape [B,C,T,H,W])
        - Req with .output tensor
        """
        from sglang.multimodal_gen.runtime.entrypoints.utils import save_outputs
        from sglang.multimodal_gen.configs.sample.sampling_params import DataType

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"{request_id}.mp4")

        output_tensor = getattr(result, "output", None)
        if output_tensor is not None:
            # Squeeze batch dimension: [B, C, T, H, W] -> [C, T, H, W]
            if output_tensor.dim() == 5 and output_tensor.shape[0] == 1:
                output_tensor = output_tensor.squeeze(0)
            save_outputs(
                [output_tensor],
                DataType.VIDEO,
                fps=24,
                save_output=True,
                build_output_path=lambda idx: output_path,
            )
        elif hasattr(result, "output_file_paths") and result.output_file_paths:
            src = result.output_file_paths[0]
            if src != output_path and os.path.exists(src):
                shutil.copy2(src, output_path)
        return output_path

    @dynamo_endpoint(GenerateRequest, VideoResponse)
    async def generate(self, request: GenerateRequest):
        """Generate video (or continue from cache if cache_id is set)."""
        from helios_latent_cache import LatentCacheManager

        is_continuation = bool(request.cache_id)
        request_id = request.request_id or str(uuid.uuid4())[:8]
        new_cache_id = LatentCacheManager.new_id()
        cache_mgr = LatentCacheManager(CACHE_DIR)

        if is_continuation:
            logger.info(
                "Continue: cache_id=%s, %d frames, new_cache_id=%s",
                request.cache_id, request.num_frames, new_cache_id,
            )
        else:
            logger.info(
                "Generate: '%s' (%dx%d, %d frames, seed=%d, cache_id=%s)",
                request.prompt[:80], request.width, request.height,
                request.num_frames, request.seed, new_cache_id,
            )

        loop = asyncio.get_event_loop()
        t0 = time.monotonic()

        def _run():
            if is_continuation:
                entry = cache_mgr.load(request.cache_id)
                meta = entry.metadata
                extra = {
                    "init_history_latents": entry.history_latents,
                    "init_image_latents": entry.image_latents,
                    "cache_save_dir": CACHE_DIR,
                    "cache_save_id": new_cache_id,
                    "cache_metadata": {
                        **meta,
                        "continued_from": request.cache_id,
                        "continuation_num_frames": request.num_frames,
                    },
                }
                req = self._build_req(
                    prompt=meta.get("prompt", ""),
                    negative_prompt=meta.get("negative_prompt", ""),
                    height=meta.get("height", 384),
                    width=meta.get("width", 640),
                    num_frames=request.num_frames,
                    seed=request.seed,
                    request_id=request_id,
                    extra=extra,
                )
            else:
                extra = {
                    "cache_save_dir": CACHE_DIR,
                    "cache_save_id": new_cache_id,
                    "cache_metadata": {
                        "prompt": request.prompt,
                        "negative_prompt": request.negative_prompt,
                        "height": request.height,
                        "width": request.width,
                        "num_frames": request.num_frames,
                        "seed": request.seed,
                        "model_path": MODEL_PATH,
                    },
                }
                req = self._build_req(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    height=request.height,
                    width=request.width,
                    num_frames=request.num_frames,
                    seed=request.seed,
                    request_id=request_id,
                    extra=extra,
                )

            result = self._run_pipeline(req)
            video_path = self._save_video(result, request_id)
            num_frames_out = _count_output_frames(result)
            return video_path, num_frames_out

        try:
            video_path, num_frames_out = await loop.run_in_executor(None, _run)
            elapsed = time.monotonic() - t0
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            action = "Continued" if is_continuation else "Generated"
            logger.info(
                "%s %d frames in %.1fs (peak %.1f GB) -> %s [cache: %s]",
                action, num_frames_out, elapsed, peak_mem, video_path, new_cache_id,
            )
            yield VideoResponse(
                video_path=video_path,
                cache_id=new_cache_id,
                num_frames=num_frames_out,
                generation_time_s=round(elapsed, 2),
                peak_memory_gb=round(peak_mem, 2),
            ).model_dump()
        except Exception as e:
            logger.error("Generation failed: %s", e, exc_info=True)
            yield VideoResponse(error=str(e)).model_dump()

    def shutdown(self):
        if self.pipeline:
            try:
                del self.pipeline
            except Exception:
                pass
            torch.cuda.empty_cache()


# ---------- HTTP Server (Orchestrator) ----------

async def run_http_server(runtime: DistributedRuntime):
    """HTTP server that forwards requests to the Helios Dynamo worker."""
    from aiohttp import web
    from helios_latent_cache import LatentCacheManager

    ns = runtime.namespace("helios")
    client = await ns.component("worker").endpoint("generate").client()

    logger.info("Waiting for Helios worker to be ready...")
    await client.wait_for_instances()
    logger.info("Helios worker connected!")

    cache_mgr = LatentCacheManager(CACHE_DIR)

    async def _collect_response(stream):
        """Collect the final response from a Dynamo stream."""
        result = None
        async for chunk in stream:
            data = chunk.data() if hasattr(chunk, "data") else chunk
            if isinstance(data, str):
                data = json.loads(data)
            elif isinstance(data, bytes):
                data = json.loads(data.decode())
            result = data
        return result

    async def handle_generate(http_request: web.Request) -> web.Response:
        try:
            body = await http_request.json()
            if "prompt" not in body:
                return web.json_response({"error": "missing 'prompt'"}, status=400)

            req = GenerateRequest(**body)
            req.request_id = str(uuid.uuid4())[:8]

            t0 = time.monotonic()
            stream = await client.generate(req.model_dump_json())
            result = await _collect_response(stream)

            if result is None:
                return web.json_response({"error": "empty response"}, status=500)
            if result.get("error"):
                return web.json_response({"error": result["error"]}, status=500)

            total_time = time.monotonic() - t0
            return web.json_response({
                "id": f"video-{req.request_id}",
                "created": int(time.time()),
                "model": MODEL_PATH,
                "cache_id": result.get("cache_id", ""),
                "data": [{"video_path": result.get("video_path", "")}],
                "timings": {
                    "generation_s": result.get("generation_time_s", 0),
                    "total_s": round(total_time, 2),
                },
                "num_frames": result.get("num_frames", 0),
                "peak_memory_gb": result.get("peak_memory_gb", 0),
            })
        except Exception as e:
            logger.error("Generate request failed: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def handle_continue(http_request: web.Request) -> web.Response:
        try:
            body = await http_request.json()
            if "cache_id" not in body:
                return web.json_response({"error": "missing 'cache_id'"}, status=400)

            # Build a GenerateRequest with cache_id set for continuation
            req = GenerateRequest(**body)
            req.request_id = str(uuid.uuid4())[:8]

            t0 = time.monotonic()
            stream = await client.generate(req.model_dump_json())
            result = await _collect_response(stream)

            if result is None:
                return web.json_response({"error": "empty response"}, status=500)
            if result.get("error"):
                return web.json_response({"error": result["error"]}, status=500)

            total_time = time.monotonic() - t0
            return web.json_response({
                "id": f"video-{req.request_id}",
                "created": int(time.time()),
                "model": MODEL_PATH,
                "cache_id": result.get("cache_id", ""),
                "continued_from": body["cache_id"],
                "data": [{"video_path": result.get("video_path", "")}],
                "timings": {
                    "generation_s": result.get("generation_time_s", 0),
                    "total_s": round(total_time, 2),
                },
                "num_frames": result.get("num_frames", 0),
                "peak_memory_gb": result.get("peak_memory_gb", 0),
            })
        except Exception as e:
            logger.error("Continue request failed: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def handle_list_caches(http_request: web.Request) -> web.Response:
        try:
            caches = cache_mgr.list_caches()
            return web.json_response({"caches": caches})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_delete_cache(http_request: web.Request) -> web.Response:
        cache_id = http_request.match_info["cache_id"]
        deleted = cache_mgr.delete(cache_id)
        if deleted:
            return web.json_response({"status": "deleted", "cache_id": cache_id})
        return web.json_response({"error": "cache not found"}, status=404)

    async def handle_health(http_request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "model": MODEL_PATH,
            "backend": "sglang-pipeline-direct",
            "features": ["generate", "continue", "cache"],
        })

    async def handle_video(http_request: web.Request) -> web.Response:
        filename = http_request.match_info["filename"]
        if "/" in filename or ".." in filename:
            return web.json_response({"error": "invalid filename"}, status=400)
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            return web.json_response({"error": "not found"}, status=404)
        return web.FileResponse(filepath, headers={"Content-Type": "video/mp4"})

    app = web.Application()
    app.router.add_post("/v1/videos/generate", handle_generate)
    app.router.add_post("/v1/videos/continue", handle_continue)
    app.router.add_get("/v1/caches", handle_list_caches)
    app.router.add_delete("/v1/caches/{cache_id}", handle_delete_cache)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/videos/{filename}", handle_video)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", HTTP_PORT)
    await site.start()

    logger.info("HTTP server listening on http://0.0.0.0:%d", HTTP_PORT)
    logger.info("  POST /v1/videos/generate    {\"prompt\": \"...\"}")
    logger.info("  POST /v1/videos/continue    {\"cache_id\": \"...\"}")
    logger.info("  GET  /v1/caches")
    logger.info("  DELETE /v1/caches/{id}")
    logger.info("  GET  /health")
    logger.info("  GET  /videos/<id>.mp4")

    await asyncio.Event().wait()


# ---------- Main ----------

MODE = os.environ.get("MODE", "worker")


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    if MODE == "server":
        await run_http_server(runtime)
    else:
        ns = runtime.namespace("helios")

        stage = HeliosContinuationWorker()
        await asyncio.get_event_loop().run_in_executor(None, stage.load_model)

        endpoint = ns.component("worker").endpoint("generate")
        logger.info("Serving: helios.worker.generate (supports both generate and continue)")
        await endpoint.serve_endpoint(stage.generate)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    uvloop.install()
    asyncio.run(worker())
