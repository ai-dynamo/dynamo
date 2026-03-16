# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang PipelineStage utilities for disaggregated diffusion workers.

Provides helpers to construct ServerArgs, load partial pipelines (only the
modules each worker needs), and convert between Dynamo protocol types and
SGLang's Req dataclass.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def build_server_args(model_path: str, **overrides):
    """Create a ServerArgs, initialize torch.distributed, and set the global singleton."""
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    defaults = dict(
        model_path=model_path,
        num_gpus=1,
    )
    defaults.update(overrides)
    server_args = ServerArgs.from_kwargs(**defaults)
    set_global_server_args(server_args)

    _ensure_distributed_init(server_args)

    return server_args


def _ensure_distributed_init(server_args):
    """Initialize torch.distributed and model-parallel groups via SGLang."""
    import inspect
    from sglang.multimodal_gen.runtime.distributed import (
        model_parallel_is_initialized,
        maybe_init_distributed_environment_and_model_parallel,
    )

    if model_parallel_is_initialized():
        return

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(server_args.master_port))
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    kwargs = dict(
        tp_size=server_args.tp_size,
        enable_cfg_parallel=server_args.enable_cfg_parallel,
        ulysses_degree=server_args.ulysses_degree,
        ring_degree=server_args.ring_degree,
        sp_size=server_args.sp_degree,
        dp_size=server_args.dp_size,
        distributed_init_method=f"tcp://127.0.0.1:{server_args.master_port}",
    )
    sig = inspect.signature(maybe_init_distributed_environment_and_model_parallel)
    if "dist_timeout" in sig.parameters:
        kwargs["dist_timeout"] = server_args.dist_timeout

    maybe_init_distributed_environment_and_model_parallel(**kwargs)


def build_partial_pipeline(
    server_args,
    required_modules: List[str],
):
    """Load a pipeline with only *required_modules* populated.

    Auto-detects pipeline class from model_index.json, suppresses automatic
    stage creation, and syncs all component configs (even unloaded ones).
    """
    from sglang.multimodal_gen.runtime.pipelines import get_model_info

    model_info = get_model_info(server_args.model_path)
    base_pipeline_cls = model_info.pipeline_cls

    # Build a partial pipeline class that:
    #  1. Suppresses automatic stage creation
    #  2. Skips LoRA init (which assumes 'transformer' always exists)
    def _noop_create_stages(self, server_args):
        return None

    def _safe_init(self, **kwargs):
        # Call ComposedPipelineBase.__init__ directly, skipping LoRAPipeline
        # which tries to access self.modules['transformer']
        from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
            ComposedPipelineBase,
        )
        ComposedPipelineBase.__init__(self, **kwargs)

    partial_cls = type(
        f"_Partial{base_pipeline_cls.__name__}",
        (base_pipeline_cls,),
        {
            "create_pipeline_stages": _noop_create_stages,
            "__init__": _safe_init,
        },
    )

    pipeline = partial_cls(
        model_path=server_args.model_path,
        server_args=server_args,
        required_config_modules=required_modules,
    )

    _sync_all_component_configs(server_args, pipeline)

    return pipeline


def _sync_all_component_configs(server_args, pipeline):
    """Read config.json for every component in model_index.json and update
    the corresponding arch_config in ``server_args.pipeline_config``, ensuring
    correct parameters even for components whose weights are not loaded.
    """
    import json

    CONFIG_ATTR_MAP = {
        "vae": ("vae_config", "update_model_arch"),
        "video_vae": ("vae_config", "update_model_arch"),
        "transformer": ("dit_config", "update_model_arch"),
        "video_dit": ("dit_config", "update_model_arch"),
        "audio_dit": ("audio_dit_config", "update_model_arch"),
        "audio_vae": ("audio_vae_config", "update_model_arch"),
    }

    # pipeline.model_path is resolved to a local path by _load_config()
    # (may differ from server_args.model_path if the original was a hub ID).
    model_path = pipeline.model_path
    model_index_path = os.path.join(model_path, "model_index.json")
    if not os.path.isfile(model_index_path):
        return

    with open(model_index_path, "r") as f:
        model_index = json.load(f)

    pipeline_config = server_args.pipeline_config
    for component_name, mapping in CONFIG_ATTR_MAP.items():
        config_attr, update_method_name = mapping
        cfg = getattr(pipeline_config, config_attr, None)
        if cfg is None:
            continue

        if component_name not in model_index:
            continue

        config_json_path = os.path.join(model_path, component_name, "config.json")
        if not os.path.isfile(config_json_path):
            continue

        with open(config_json_path, "r") as f:
            hf_config = json.load(f)

        hf_config.pop("_class_name", None)
        hf_config.pop("_diffusers_version", None)

        update_fn = getattr(cfg, update_method_name, None)
        if update_fn is not None:
            update_fn(hf_config)
            logger.info("Synced %s config from %s (e.g. z_dim=%s)",
                        config_attr,
                        config_json_path,
                        getattr(getattr(cfg, "arch_config", cfg), "z_dim", "N/A"))

        if hasattr(cfg, "post_init"):
            cfg.post_init()


def get_component_backend(module) -> str:
    """Return a human-readable string indicating which backend loaded *module*."""
    mod = type(module).__module__ or ""
    cls = type(module).__qualname__
    if mod.startswith("sglang."):
        return f"sglang-optimized ({cls})"
    if mod.startswith("diffusers."):
        return f"native-diffusers ({cls})"
    if mod.startswith("transformers."):
        return f"native-transformers ({cls})"
    return f"unknown ({mod}.{cls})"


def build_config(server_args):
    """Construct a minimal Config from a diffusion ServerArgs."""
    import types

    try:
        from dynamo.sglang.args import Config, DynamoConfig
        dynamo_args = DynamoConfig.__new__(DynamoConfig)
    except ImportError:
        dynamo_args = types.SimpleNamespace()
        Config = None

    dynamo_args.component = "disagg_diffusion"
    dynamo_args.namespace = "disagg_diffusion"
    dynamo_args.diffusion_worker = True
    dynamo_args.use_kv_events = False
    dynamo_args.media_output_fs_url = "file:///tmp/disagg_videos"
    dynamo_args.media_output_http_url = None
    dynamo_args.use_sglang_tokenizer = False
    dynamo_args.multimodal_processor = False
    dynamo_args.multimodal_encode_worker = False
    dynamo_args.multimodal_worker = False
    dynamo_args.embedding_worker = False
    dynamo_args.image_diffusion_worker = False
    dynamo_args.video_generation_worker = False
    dynamo_args.disagg_config = None
    dynamo_args.disagg_config_key = None
    dynamo_args.endpoint = "generate"
    dynamo_args.discovery_backend = "etcd"
    dynamo_args.request_plane = "tcp"
    dynamo_args.event_plane = "tcp"
    dynamo_args.connector = []
    dynamo_args.enable_local_indexer = False
    dynamo_args.durable_kv_events = False
    dynamo_args.endpoint_types = "generate"
    dynamo_args.dump_config_to = None
    dynamo_args.multimodal_embedding_cache_capacity_gb = 0.0
    dynamo_args.output_modalities = ["video"]
    dynamo_args.dyn_tool_call_parser = None
    dynamo_args.dyn_reasoning_parser = None
    dynamo_args.custom_jinja_template = None

    if not hasattr(server_args, "disaggregation_mode"):
        server_args.disaggregation_mode = "null"

    if Config is not None:
        return Config(server_args, dynamo_args)

    cfg = types.SimpleNamespace()
    cfg.server_args = server_args
    cfg.dynamo_args = dynamo_args
    cfg.serving_mode = getattr(server_args, "disaggregation_mode", "null")
    return cfg


def build_req(
    prompt: str,
    negative_prompt: Optional[str] = "",
    height: int = 480,
    width: int = 832,
    num_frames: int = 17,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    seed: int = 42,
    device: str = "cuda",
    **extra_fields,
) -> "Req":
    """Construct a minimal SGLang ``Req`` for running pipeline stages."""
    from sglang.multimodal_gen.configs.sample.base import DataType
    from sglang.multimodal_gen.runtime.pipelines.schedule_batch import Req

    req = Req(
        data_type=DataType.VIDEO,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        do_classifier_free_guidance=(guidance_scale > 1.0),
    )

    for k, v in extra_fields.items():
        setattr(req, k, v)

    return req


def extract_tensors_from_req(
    req,
    keys: List[str],
) -> Dict[str, object]:
    """Pull named tensor fields out of a ``Req`` for NIXL transfer.

    Single-element lists are unwrapped to bare tensors.
    Multi-element lists (dual-encoder outputs) are preserved as lists.
    """
    result: Dict[str, object] = {}
    for key in keys:
        val = getattr(req, key, None)
        if val is None:
            continue
        if isinstance(val, list):
            if len(val) == 0:
                continue
            if len(val) == 1:
                result[key] = val[0]
            else:
                # Keep multi-element lists intact (e.g. dual-encoder outputs
                # with incompatible shapes)
                result[key] = val
        elif isinstance(val, torch.Tensor):
            result[key] = val
    return result


def inject_tensors_to_req(
    req,
    tensors: Dict[str, object],
    list_fields: Optional[List[str]] = None,
):
    """Inject received tensors back into a ``Req``.

    Values may be bare tensors (single-encoder outputs) or lists of tensors
    (dual-encoder outputs like HunyuanVideo's Llama + CLIP embeddings).
    ``list_fields`` specifies Req attributes that expect a list value;
    bare tensors are auto-wrapped in a single-element list.
    """
    list_fields = set(list_fields or [
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_embeds",
        "neg_pooled_embeds",
        "image_embeds",
    ])
    for key, value in tensors.items():
        if key in list_fields:
            if isinstance(value, list):
                # Multi-encoder: already a list of tensors
                setattr(req, key, value)
            else:
                # Single encoder: wrap in list
                setattr(req, key, [value])
        else:
            setattr(req, key, value)
    return req
