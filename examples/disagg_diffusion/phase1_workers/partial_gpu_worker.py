# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PartialGPUWorker, IntermediateOutputStage, and subprocess launcher.

This module is the single source of truth for disaggregated diffusion's
integration with sglang.  Everything that runs **inside the Scheduler
subprocess** lives here — no Dynamo imports, no heavy dependencies beyond
sglang + torch.

Architecture::

    Main process                    Subprocess (spawned)
    ───────────                     ────────────────────
    launch_partial_server()  ──►    _run_partial_scheduler_process()
                                      ├─ monkey-patch GPUWorker → PartialGPUWorker
                                      └─ run_scheduler_process()
                                           └─ Scheduler()
                                                └─ PartialGPUWorker()
                                                     └─ pipeline.forward()
    SchedulerClient  ◄────ZMQ────►       Scheduler.event_loop()

``IntermediateOutputStage`` is appended as the last pipeline stage for
encoder / denoiser.  It packages ``Req`` tensors into ``OutputBatch``
so the result travels through ZMQ as a standard ``OutputBatch``.

``DecodingStage`` (VAE) already returns ``OutputBatch``, so no extra
stage is needed for the VAE worker.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
from typing import Callable, Dict, List, Optional

import torch
from setproctitle import setproctitle

from sglang.multimodal_gen.runtime.distributed import (
    get_tp_rank,
    get_tp_world_size,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_ring_parallel_rank,
    get_ring_parallel_world_size,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines.schedule_batch import Req, OutputBatch
from sglang.multimodal_gen.runtime.pipelines.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# layerwise_offload may not exist in all sglang versions — guard import
try:
    from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
except ImportError:
    OffloadableDiTMixin = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# IntermediateOutputStage
# ═══════════════════════════════════════════════════════════════════════


class DeviceMoveStage(PipelineStage):
    """Move tensor fields on ``Req`` to the current CUDA device.

    Tensors arriving via ZMQ (pickle) land on CPU.  Prepend this stage
    at the start of denoiser / VAE pipelines that receive tensors from
    other processes.
    """

    def __init__(self, tensor_fields: List[str]):
        super().__init__()
        self._tensor_fields = tensor_fields

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = torch.device("cuda")
        for field in self._tensor_fields:
            val = getattr(batch, field, None)
            if val is None:
                continue
            if isinstance(val, list):
                setattr(batch, field, [
                    t.to(device).contiguous() if isinstance(t, torch.Tensor) else t
                    for t in val
                ])
            elif isinstance(val, torch.Tensor):
                setattr(batch, field, val.to(device).contiguous())
        return batch


class IntermediateOutputStage(PipelineStage):
    """Pipeline stage that packages ``Req`` tensors into ``OutputBatch``.

    Append as the **last** stage in encoder / denoiser partial pipelines.
    ``GPUWorker.execute_forward`` sees an ``OutputBatch`` (not a ``Req``)
    and returns it through ZMQ without any custom override.

    ``output_fields`` lists the ``Req`` attributes to extract.
    Single-element lists are unwrapped to bare tensors.
    Multi-element lists (e.g. dual-encoder embeddings) are preserved as
    lists so the receiver can reconstruct them correctly.
    The handler receives ``OutputBatch.output`` as ``dict[str, Tensor|list]``.
    """

    def __init__(self, output_fields: List[str]):
        super().__init__()
        self._output_fields = output_fields

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        tensors: Dict[str, object] = {}
        for field in self._output_fields:
            val = getattr(batch, field, None)
            if val is None:
                continue
            if isinstance(val, list):
                if len(val) == 0:
                    continue
                if len(val) == 1:
                    tensors[field] = val[0]
                else:
                    # Preserve multi-element lists (e.g. dual-encoder outputs
                    # with incompatible shapes) so they survive ZMQ pickle.
                    tensors[field] = val
            elif isinstance(val, torch.Tensor):
                tensors[field] = val
        return OutputBatch(output=tensors)


# ═══════════════════════════════════════════════════════════════════════
# Stage builder functions (picklable — used as subprocess args)
# ═══════════════════════════════════════════════════════════════════════


def build_encoder_stages(pipeline, server_args):
    """TextEncodingStage → IntermediateOutputStage(prompt_embeds, …).

    Automatically detects all loaded text encoders/tokenizers so that
    both single-encoder (Wan) and dual-encoder (HunyuanVideo) models work.
    """
    from sglang.multimodal_gen.runtime.pipelines.stages.text_encoding import (
        TextEncodingStage,
    )
    from sglang_utils import get_component_backend

    # Collect all available text encoders and tokenizers
    text_encoders = []
    tokenizers = []
    for name in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
        enc = pipeline.get_module(name)
        if enc is not None:
            text_encoders.append(enc)
            logger.info("%s backend: %s", name, get_component_backend(enc))
    for name in ["tokenizer", "tokenizer_2", "tokenizer_3"]:
        tok = pipeline.get_module(name)
        if tok is not None:
            tokenizers.append(tok)

    assert len(text_encoders) > 0, "No text encoders found in pipeline"
    assert len(text_encoders) == len(tokenizers), (
        f"Encoder/tokenizer count mismatch: {len(text_encoders)} vs {len(tokenizers)}"
    )

    # Always list both fields; IntermediateOutputStage skips None values,
    # so negative_prompt_embeds is simply omitted when CFG is disabled.
    return [
        TextEncodingStage(text_encoders=text_encoders, tokenizers=tokenizers),
        IntermediateOutputStage(["prompt_embeds", "negative_prompt_embeds"]),
    ]


def build_denoiser_stages(pipeline, server_args):
    """DeviceMove → LatentPrep → TimestepPrep → Denoising → IntermediateOutput."""
    from sglang.multimodal_gen.runtime.pipelines.stages.latent_preparation import (
        LatentPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines.stages.timestep_preparation import (
        TimestepPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines.stages.denoising import (
        DenoisingStage,
    )
    from sglang_utils import get_component_backend

    transformer = pipeline.get_module("transformer")
    scheduler = pipeline.get_module("scheduler")
    logger.info("transformer backend: %s", get_component_backend(transformer))

    # Fields that arrive from the encoder via ZMQ and need GPU placement.
    # prompt_embeds may be a single tensor (Wan) or list of tensors
    # (HunyuanVideo dual-encoder) — DeviceMoveStage handles both.
    device_move_fields = ["prompt_embeds", "negative_prompt_embeds"]

    return [
        DeviceMoveStage(device_move_fields),
        LatentPreparationStage(scheduler=scheduler, transformer=transformer),
        TimestepPreparationStage(scheduler=scheduler),
        DenoisingStage(transformer=transformer, scheduler=scheduler),
        IntermediateOutputStage(["latents"]),
    ]


def build_vae_stages(pipeline, server_args):
    """DecodingStage (already returns OutputBatch — no extra stage needed)."""
    from sglang.multimodal_gen.runtime.pipelines.stages.decoding import (
        DecodingStage,
    )
    from sglang_utils import get_component_backend

    vae = pipeline.get_module("vae")
    logger.info("vae backend: %s", get_component_backend(vae))
    return [DecodingStage(vae=vae, pipeline=pipeline)]


# ═══════════════════════════════════════════════════════════════════════
# PartialGPUWorker
# ═══════════════════════════════════════════════════════════════════════


class PartialGPUWorker(GPUWorker):
    """GPUWorker that loads only a subset of pipeline modules.

    Only ``init_device_and_model`` is overridden.  Everything else
    (``execute_forward``, ``do_mem_analysis``, LoRA, etc.) is inherited.
    """

    def __init__(
        self,
        required_modules: List[str],
        custom_stages_fn: Optional[Callable] = None,
        **kwargs,
    ):
        self._required_modules = required_modules
        self._custom_stages_fn = custom_stages_fn
        super().__init__(**kwargs)

    def init_device_and_model(self) -> None:
        torch.get_device_module().set_device(self.local_rank)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.server_args.num_gpus)

        dist_kwargs = dict(
            tp_size=self.server_args.tp_size,
            enable_cfg_parallel=self.server_args.enable_cfg_parallel,
            ulysses_degree=getattr(self.server_args, "ulysses_degree", 1),
            ring_degree=getattr(self.server_args, "ring_degree", 1),
            sp_size=self.server_args.sp_degree,
            dp_size=self.server_args.dp_size,
            distributed_init_method=f"tcp://127.0.0.1:{self.master_port}",
        )
        # dist_timeout only in newer sglang versions
        import inspect
        sig = inspect.signature(maybe_init_distributed_environment_and_model_parallel)
        if "dist_timeout" in sig.parameters:
            dist_kwargs["dist_timeout"] = self.server_args.dist_timeout
        maybe_init_distributed_environment_and_model_parallel(**dist_kwargs)

        if model_parallel_is_initialized():
            suffix = ""
            if get_tp_world_size() != 1:
                suffix += f"_TP{get_tp_rank()}"
            if get_ulysses_parallel_world_size() != 1:
                suffix += f"_U{get_ulysses_parallel_rank()}"
            if get_ring_parallel_world_size() != 1:
                suffix += f"_R{get_ring_parallel_rank()}"
            if get_classifier_free_guidance_world_size() != 1:
                suffix += f"_C{get_classifier_free_guidance_rank()}"
            setproctitle(f"sgl_diffusion::partial{suffix}")
        else:
            setproctitle(f"sgl_diffusion::partial_{self.local_rank}")

        from sglang_utils import build_partial_pipeline

        self.pipeline = build_partial_pipeline(
            self.server_args,
            required_modules=self._required_modules,
        )

        if self._custom_stages_fn is not None:
            stages = self._custom_stages_fn(self.pipeline, self.server_args)
            for stage in stages:
                name = type(stage).__name__
                self.pipeline.add_stage(name, stage)

        if getattr(self.server_args, "dit_layerwise_offload", False) and OffloadableDiTMixin is not None:
            for module_name in [
                "transformer", "transformer_2",
                "video_dit", "video_dit_2", "audio_dit",
            ]:
                dit = self.pipeline.get_module(module_name)
                if dit is not None:
                    if isinstance(dit, OffloadableDiTMixin):
                        dit.configure_layerwise_offload(self.server_args)
                    else:
                        logger.info(
                            "Module %s does not support layerwise offload.",
                            type(dit).__name__,
                        )

        logger.info(
            "PartialGPUWorker %d: modules=%s stages=%s",
            self.rank,
            self._required_modules,
            list(self.pipeline._stage_name_mapping.keys()),
        )


# ═══════════════════════════════════════════════════════════════════════
# Subprocess entry point + launcher
# ═══════════════════════════════════════════════════════════════════════


def _patch_triton_norm_contiguous():
    """Patch SGLang's triton norm_infer to handle non-contiguous tensors.

    HunyuanVideo's transformer produces non-contiguous intermediate tensors
    (e.g. from attention reshapes) which trip the triton kernel assertion
    ``assert x.stride(-1) == 1``.

    Must patch both ``triton_ops`` and ``layernorm`` modules because
    layernorm uses ``from triton_ops import norm_infer`` (direct binding).
    """
    try:
        import sglang.multimodal_gen.runtime.layers.triton_ops as triton_ops
        import sglang.multimodal_gen.runtime.layers.layernorm as layernorm_mod

        _orig = triton_ops.norm_infer

        def _safe_norm_infer(x, *args, **kwargs):
            if not x.is_contiguous():
                x = x.contiguous()
            return _orig(x, *args, **kwargs)

        triton_ops.norm_infer = _safe_norm_infer
        layernorm_mod.norm_infer = _safe_norm_infer
    except (ImportError, AttributeError):
        pass


def _run_partial_scheduler_process(
    required_modules: List[str],
    custom_stages_fn: Callable,
    local_rank: int,
    rank: int,
    master_port: int,
    server_args: ServerArgs,
    pipe_writer,
    task_pipe_r,
    result_pipe_w,
    task_pipes_to_slaves: list,
    result_pipes_from_slaves: list,
) -> None:
    """Subprocess entry point: patch GPUWorker then delegate to sglang."""
    # Ensure sglang_utils etc. are importable in the subprocess
    workers_dir = os.path.dirname(os.path.abspath(__file__))
    if workers_dir not in sys.path:
        sys.path.insert(0, workers_dir)

    _patch_triton_norm_contiguous()

    import sglang.multimodal_gen.runtime.managers.scheduler as sched_mod

    _rm, _csf = required_modules, custom_stages_fn

    class _PatchedGPUWorker(PartialGPUWorker):
        def __init__(self, **kwargs):
            super().__init__(required_modules=_rm, custom_stages_fn=_csf, **kwargs)

    sched_mod.GPUWorker = _PatchedGPUWorker

    from sglang.multimodal_gen.runtime.managers.gpu_worker import (
        run_scheduler_process,
    )

    run_scheduler_process(
        local_rank, rank, master_port, server_args, pipe_writer,
        task_pipe_r, result_pipe_w,
        task_pipes_to_slaves, result_pipes_from_slaves,
    )


def launch_partial_server(
    server_args: ServerArgs,
    required_modules: List[str],
    custom_stages_fn: Callable,
) -> list[mp.Process]:
    """Spawn Scheduler subprocess(es) with ``PartialGPUWorker``.

    Mirrors sglang's ``launch_server()`` logic: spawns ``num_gpus``
    processes with master/slave pipe wiring so that TP, SP, and CFG
    parallelism all work out of the box.

    * ``num_gpus == 1`` → single process, no pipes.
    * ``num_gpus  > 1`` → rank 0 is master (has ZMQ receiver + pipes to
      slaves), ranks 1..N are slaves (coordinate via ``torch.distributed``
      broadcast).

    Returns the process list (for later shutdown).

    After this returns, call ``async_scheduler_client.initialize(server_args)``
    to connect from the main process.
    """
    from sglang.multimodal_gen.runtime.utils.logging_utils import configure_logger

    configure_logger(server_args)

    num_gpus = server_args.num_gpus
    master_port = server_args.master_port
    processes: list[mp.Process] = []

    # --- pipes for master ↔ slave coordination (same as launch_server) ---
    task_pipes_to_slaves_w: list = []
    task_pipes_to_slaves_r: list = []
    result_pipes_from_slaves_w: list = []
    result_pipes_from_slaves_r: list = []

    for _ in range(num_gpus - 1):
        r, w = mp.Pipe(duplex=False)
        task_pipes_to_slaves_r.append(r)
        task_pipes_to_slaves_w.append(w)

    for _ in range(num_gpus - 1):
        r, w = mp.Pipe(duplex=False)
        result_pipes_from_slaves_r.append(r)
        result_pipes_from_slaves_w.append(w)

    # --- spawn one process per GPU ---
    readiness_readers: list = []

    for i in range(num_gpus):
        reader, writer = mp.Pipe(duplex=False)
        readiness_readers.append(reader)

        if i == 0:
            # Rank 0 (master): owns ZMQ receiver + write-ends to slaves
            args = (
                required_modules,
                custom_stages_fn,
                i,              # local_rank
                i,              # rank
                master_port,
                server_args,
                writer,         # pipe_writer  (readiness signal)
                None,           # task_pipe_r  (master doesn't receive tasks)
                None,           # result_pipe_w (master doesn't send results)
                task_pipes_to_slaves_w,
                result_pipes_from_slaves_r,
            )
        else:
            # Rank > 0 (slave): coordinates via torch.distributed broadcast
            args = (
                required_modules,
                custom_stages_fn,
                i,              # local_rank
                i,              # rank
                master_port,
                server_args,
                writer,         # pipe_writer  (readiness signal)
                None,           # task_pipe_r
                None,           # result_pipe_w
                task_pipes_to_slaves_r[i - 1],
                result_pipes_from_slaves_w[i - 1],
            )

        process = mp.Process(
            target=_run_partial_scheduler_process,
            args=args,
            name=f"sglang-partial-worker-{i}",
            daemon=True,
        )
        process.start()
        writer.close()
        processes.append(process)

    # --- close unused pipe ends in the parent (same as launch_server) ---
    for p in task_pipes_to_slaves_w:
        p.close()
    for p in task_pipes_to_slaves_r:
        p.close()
    for p in result_pipes_from_slaves_w:
        p.close()
    for p in result_pipes_from_slaves_r:
        p.close()

    # --- wait for all workers to report ready ---
    for i, reader in enumerate(readiness_readers):
        try:
            data = reader.recv()
        except EOFError:
            processes[i].join()
            raise RuntimeError(
                f"Partial scheduler rank {i} died (exit code {processes[i].exitcode}). "
                "Check logs above for errors."
            )
        reader.close()
        if data.get("status") != "ready":
            raise RuntimeError(f"Partial scheduler rank {i} failed to initialize.")

    pids = [p.pid for p in processes]
    logger.info(
        "Partial scheduler ready: %d GPU(s), pids=%s", num_gpus, pids,
    )
    return processes
