# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handler for the Krea realtime video pipeline (sgl-project/sglang#19817).

Bridges Dynamo's existing OpenAI-style `/v1/videos` request shape to the
chunk-based realtime generation loop introduced in the antgroup chunk_diffusion
branch. Mid-flight prompt/action updates from the upstream WebSocket protocol
are intentionally not exposed — the worker is driven by a single
`CreateVideoRequest` and yields one MP4 chunk per response. Clients call
`POST /v1/videos` with `"stream": true` so the Rust frontend forwards each
yielded `NvVideosResponse` as an SSE event. The chunk is delivered either
inline (base64 in `data[0].b64_json`) or by URL (`data[0].url`) according
to the request's `response_format`.
"""

import base64
import logging
import math
import time
from typing import Any, AsyncGenerator, Optional

import torch

from dynamo._core import Context
from dynamo.common.storage import upload_file_to_fs
from dynamo.sglang.args import Config
from dynamo.sglang.protocol import (
    CreateVideoRequest,
    VideoData,
    VideoGenerationResponse,
    VideoNvExt,
)
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseGenerativeHandler

logger = logging.getLogger(__name__)


# Default chunk size used by antgroup's realtime demo. The Krea pipeline emits
# this many frames per `process_generation_batch` call when running in t2v mode
# with no user-supplied override.
_DEFAULT_FRAMES_PER_CHUNK = 12


class RealtimeVideoWorkerHandler(BaseGenerativeHandler):
    """Single-shot driver for the Krea realtime video pipeline.

    Lifecycle per request:
      1. Build a `RealtimeVideoGenerationsRequest` from the Dynamo
         `CreateVideoRequest` and attach it to a fresh `GenerateSession`.
      2. Compute the target chunk count from `nvext.num_frames` (or
         `fps * seconds`) divided by `frames_per_chunk`.
      3. Loop chunks: call `process_generation_batch` against the antgroup
         scheduler client and yield one `VideoGenerationResponse` per chunk.
         Each response carries the chunk in `data[0]` with `output_format="mp4"`.
         The request's `response_format` selects how it's delivered:
           - `"b64_json"`: inline base64-encoded MP4 in `data[0].b64_json`
             (suited to SSE streaming over `POST /v1/videos` with `stream:true`).
           - `"url"`: chunk uploaded to the configured `--media-output-fs-url`
             and `data[0].url` points at it; chunks are keyed
             `<request_id>_<chunk_idx:04d>.mp4` so they sort in emission order.
      4. Release the session in `finally` (required — otherwise GPU memory
         leaks on the SGLang scheduler side).

    Cancellation: `context.is_killed()` is checked between chunks. The Rust
    SSE handler propagates client-disconnect through the context, so a closed
    connection stops generation at the next chunk boundary.
    """

    def __init__(
        self,
        generator: Any,  # DiffGenerator — kept so from_pretrained bootstraps the scheduler subprocess
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
        fs: Any = None,
    ):
        super().__init__(config, publisher)

        from dynamo.sglang._compat import import_realtime_video_api

        # Probe for the realtime API surface up-front so worker startup fails
        # fast on a Dynamo install paired with vanilla sglang. The returned
        # namespace is cached on the handler instance for use by generate().
        self._rt = import_realtime_video_api()

        self.generator = generator
        self.fs = fs
        self.fs_url = config.dynamo_args.media_output_fs_url
        self.base_url = config.dynamo_args.media_output_http_url
        self.frames_per_chunk = getattr(
            config.server_args, "realtime_frames_per_chunk", _DEFAULT_FRAMES_PER_CHUNK
        )

        logger.info(
            "RealtimeVideoWorkerHandler ready "
            f"(frames_per_chunk={self.frames_per_chunk}, fs_url={self.fs_url})"
        )

    def cleanup(self) -> None:
        if self.generator is not None:
            del self.generator
        torch.cuda.empty_cache()
        logger.info("Realtime video generator cleanup complete")
        super().cleanup()

    async def generate(
        self, request: dict[str, Any], context: Context
    ) -> AsyncGenerator[dict[str, Any], None]:
        start_time = time.time()
        context_id = context.id() or ""
        response_id = f"video-{context_id}"

        try:
            req = CreateVideoRequest(**request)
        except Exception as e:
            yield self._error_response(
                response_id, request.get("model", "unknown"), str(e)
            )
            return

        nvext = req.nvext or VideoNvExt()
        if req.size is None:
            yield self._error_response(response_id, req.model, "size is required")
            return

        try:
            width, height = _parse_size(req.size)
        except ValueError as e:
            yield self._error_response(response_id, req.model, str(e))
            return

        if nvext.fps is None:
            yield self._error_response(response_id, req.model, "fps is required")
            return

        if nvext.num_frames is not None:
            total_frames = nvext.num_frames
        elif req.seconds is not None:
            total_frames = nvext.fps * req.seconds
        else:
            yield self._error_response(
                response_id, req.model, "num_frames or seconds is required"
            )
            return

        response_format = req.response_format or "b64_json"
        if response_format not in ("url", "b64_json"):
            yield self._error_response(
                response_id,
                req.model,
                f"Unsupported response_format: {response_format!r}; expected 'url' or 'b64_json'",
            )
            return
        if response_format == "url" and self.fs is None:
            yield self._error_response(
                response_id,
                req.model,
                "response_format='url' requires --media-output-fs-url to be configured",
            )
            return

        num_chunks = max(1, math.ceil(total_frames / self.frames_per_chunk))

        realtime_req = self._build_realtime_request(
            req, nvext, width, height, total_frames
        )
        session = self._rt.GenerateSession()
        session.set_mode(self._rt.RealtimeVideoMode.T2V)
        session.setRequest(realtime_req)

        logger.info(
            f"Starting realtime video generation: session_id={session.id}, "
            f"total_frames={total_frames}, num_chunks={num_chunks}, "
            f"size={width}x{height}, fps={nvext.fps}"
        )

        chunks_emitted = 0
        try:
            for chunk_idx in range(num_chunks):
                if context.is_killed():
                    logger.info(
                        f"Request cancelled mid-stream: session_id={session.id}, "
                        f"chunks_emitted={chunks_emitted}/{num_chunks}"
                    )
                    break

                chunk_mp4_path = await self._generate_chunk(session, chunk_idx)

                if response_format == "url":
                    # Hand the local chunk file to fsspec directly. fs.put is
                    # a kernel-level copy on file:// and a native multipart
                    # upload on object stores — no read into Python memory.
                    url = await self._upload_chunk(
                        chunk_mp4_path, context_id, chunk_idx
                    )
                    video_data = VideoData(output_format="mp4", url=url)
                else:
                    with open(chunk_mp4_path, "rb") as f:
                        mp4_bytes = f.read()
                    video_data = VideoData(
                        output_format="mp4",
                        b64_json=base64.b64encode(mp4_bytes).decode("ascii"),
                    )

                yield VideoGenerationResponse(
                    id=response_id,
                    model=req.model,
                    created=int(time.time()),
                    data=[video_data],
                ).model_dump()

                session.generate_chunk_completed()
                chunks_emitted += 1

            logger.info(
                f"Realtime video complete: session_id={session.id}, "
                f"chunks={chunks_emitted}, elapsed_s={time.time() - start_time:.2f}"
            )

        except Exception as e:
            logger.error(
                f"Realtime video generation failed: session_id={session.id}, "
                f"chunk_idx={chunks_emitted}",
                exc_info=True,
            )
            yield self._error_response(response_id, req.model, str(e))

        finally:
            try:
                await self._rt.async_scheduler_client.forward(
                    self._rt.ReleaseRealtimeSessionReq(session_id=session.id)
                )
            except Exception as e:
                logger.warning(f"Failed to release realtime session {session.id}: {e}")
            session.dispose()

    async def _generate_chunk(self, session: Any, chunk_idx: int) -> str:
        """Drive one chunk through the antgroup scheduler. Returns MP4 path."""
        session.new_request()
        sampling_params = session.build_sampling_params()
        batch = self._rt.prepare_request(
            server_args=self._rt.get_global_server_args(),
            sampling_params=sampling_params,
        )
        batch.session = session.realtime_session
        batch.extra["realtime_session_id"] = session.id
        batch.block_idx = chunk_idx
        # Modified Path A: t2v only — input_video would carry V2V frames from
        # the mid-flight WebSocket action stream that this handler intentionally
        # does not consume.
        batch.input_video = None

        save_file_path_list, _ = await self._rt.process_generation_batch(
            self._rt.async_scheduler_client, batch
        )
        if not save_file_path_list:
            raise RuntimeError("process_generation_batch returned no output paths")
        return save_file_path_list[0]

    async def _upload_chunk(
        self, local_mp4_path: str, request_id: str, chunk_idx: int
    ) -> str:
        """Upload one chunk to the configured filesystem and return its URL.

        Takes the local file path produced by the scheduler so the MP4 never
        passes through Python memory — fsspec moves the bytes from disk to
        backend storage directly. Chunks within the same request share
        `request_id` but differ in `chunk_idx`; the zero-padded suffix keeps
        lexicographic order matching emission order so a downstream client
        can `ls`-and-concat without a separate index file.
        """
        storage_path = f"{request_id}_{chunk_idx:04d}.mp4"
        return await upload_file_to_fs(
            self.fs, storage_path, local_mp4_path, self.base_url
        )

    def _build_realtime_request(
        self,
        req: CreateVideoRequest,
        nvext: VideoNvExt,
        width: int,
        height: int,
        total_frames: int,
    ) -> Any:
        """Translate Dynamo's CreateVideoRequest into antgroup's request shape."""
        return self._rt.RealtimeVideoGenerationsRequest(
            prompt=req.prompt,
            first_frame=None,  # t2v only — see class docstring
            size=req.size,
            seconds=req.seconds,
            fps=nvext.fps,
            num_frames=total_frames,
            seed=nvext.seed if nvext.seed is not None else 1024,
            num_inference_steps=nvext.num_inference_steps,
            guidance_scale=nvext.guidance_scale,
            negative_prompt=nvext.negative_prompt,
        )

    def _error_response(
        self, response_id: str, model: str, error: str
    ) -> dict[str, Any]:
        return VideoGenerationResponse(
            id=response_id,
            model=model,
            created=int(time.time()),
            status="failed",
            progress=0,
            data=[],
            error=error,
        ).model_dump()


def _parse_size(size_str: str) -> tuple[int, int]:
    try:
        w, h = size_str.split("x")
        return int(w), int(h)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid size format '{size_str}', expected 'WxH' (e.g. '832x480')"
        ) from e
