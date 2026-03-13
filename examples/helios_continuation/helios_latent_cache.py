"""Helios Latent Cache: save/load denoising state for continuous video generation.

Provides:
    - LatentCacheManager: manage .pt cache files (save/load/list/delete)
    - patch_denoising_stage(): monkey-patch HeliosChunkedDenoisingStage.forward()
      to support injecting cached history_latents/image_latents via batch.extra
      and extracting them after generation for subsequent continuation.
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache data structure
# ---------------------------------------------------------------------------

@dataclass
class LatentCacheEntry:
    cache_id: str
    history_latents: torch.Tensor          # [1, C, N, H, W]
    image_latents: Optional[torch.Tensor]  # [1, C, 1, H, W] or None
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


# ---------------------------------------------------------------------------
# Cache manager
# ---------------------------------------------------------------------------

class LatentCacheManager:
    """Persist latent cache entries as .pt files on disk."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, cache_id: str) -> Path:
        return self.cache_dir / f"{cache_id}.pt"

    def save(self, entry: LatentCacheEntry) -> str:
        data = {
            "cache_id": entry.cache_id,
            "history_latents": entry.history_latents.cpu(),
            "image_latents": entry.image_latents.cpu() if entry.image_latents is not None else None,
            "prompt_embeds": entry.prompt_embeds.cpu() if entry.prompt_embeds is not None else None,
            "negative_prompt_embeds": (
                entry.negative_prompt_embeds.cpu()
                if entry.negative_prompt_embeds is not None
                else None
            ),
            "metadata": entry.metadata,
            "created_at": entry.created_at or time.time(),
        }
        path = self._path(entry.cache_id)
        torch.save(data, path)
        logger.info("Cache saved: %s (%s)", entry.cache_id, path)
        return entry.cache_id

    def load(self, cache_id: str) -> LatentCacheEntry:
        path = self._path(cache_id)
        if not path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_id}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        return LatentCacheEntry(
            cache_id=data["cache_id"],
            history_latents=data["history_latents"],
            image_latents=data.get("image_latents"),
            prompt_embeds=data.get("prompt_embeds"),
            negative_prompt_embeds=data.get("negative_prompt_embeds"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", 0.0),
        )

    def list_caches(self) -> List[Dict[str, Any]]:
        entries = []
        for pt in sorted(self.cache_dir.glob("*.pt")):
            try:
                data = torch.load(pt, map_location="cpu", weights_only=False)
                entries.append({
                    "cache_id": data["cache_id"],
                    "metadata": data.get("metadata", {}),
                    "created_at": data.get("created_at", 0.0),
                    "history_shape": list(data["history_latents"].shape),
                })
            except Exception as e:
                logger.warning("Skipping corrupt cache %s: %s", pt.name, e)
        return entries

    def delete(self, cache_id: str) -> bool:
        path = self._path(cache_id)
        if path.exists():
            path.unlink()
            logger.info("Cache deleted: %s", cache_id)
            return True
        return False

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]


def _safe_cpu(tensor_or_list):
    """Safely move a tensor (or first element of a list) to CPU, handling None/empty."""
    if tensor_or_list is None:
        return None
    if isinstance(tensor_or_list, list):
        if not tensor_or_list:
            return None
        return tensor_or_list[0].cpu()
    if hasattr(tensor_or_list, "cpu"):
        return tensor_or_list.cpu()
    return None


# ---------------------------------------------------------------------------
# Monkey-patch for HeliosChunkedDenoisingStage.forward()
# ---------------------------------------------------------------------------

def _patched_forward(self, batch, server_args):
    """Patched HeliosChunkedDenoisingStage.forward() with cache injection/extraction.

    Differences from original:
    1. If batch.extra["init_history_latents"] exists, use it instead of zeros
    2. If batch.extra["init_image_latents"] exists, use it instead of None
    3. After denoising, write output_history_latents / output_image_latents to batch.extra
    """
    import math

    import numpy as np
    import torch
    import torch.nn.functional as F

    from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.helios_denoising import (
        calculate_shift,
        optimized_scale,
        sample_block_noise,
    )
    from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

    pipeline_config = server_args.pipeline_config
    device = (
        batch.latents.device
        if hasattr(batch, "latents") and batch.latents is not None
        else torch.device("cuda")
    )
    target_dtype = PRECISION_TO_TYPE.get(
        server_args.pipeline_config.precision, torch.bfloat16
    )

    # Get config params
    num_latent_frames_per_chunk = pipeline_config.num_latent_frames_per_chunk
    history_sizes = sorted(list(pipeline_config.history_sizes), reverse=True)
    is_cfg_zero_star = pipeline_config.is_cfg_zero_star
    zero_steps = pipeline_config.zero_steps
    keep_first_frame = pipeline_config.keep_first_frame
    guidance_scale = batch.guidance_scale
    num_inference_steps = batch.num_inference_steps

    # Stage 2 params
    is_enable_stage2 = pipeline_config.is_enable_stage2
    pyramid_num_stages = pipeline_config.pyramid_num_stages
    pyramid_num_inference_steps_list = (
        pipeline_config.pyramid_num_inference_steps_list
    )
    is_distilled = pipeline_config.is_distilled
    is_amplify_first_chunk = pipeline_config.is_amplify_first_chunk
    gamma = pipeline_config.gamma

    # Move transformer to GPU if CPU-offloaded
    if server_args.dit_cpu_offload and not server_args.use_fsdp_inference:
        if next(self.transformer.parameters()).device.type == "cpu":
            self.transformer.to(get_local_torch_device())

    # Get encoder outputs
    prompt_embeds = batch.prompt_embeds
    if isinstance(prompt_embeds, list):
        prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.to(target_dtype)
    negative_prompt_embeds = batch.negative_prompt_embeds
    if isinstance(negative_prompt_embeds, list):
        negative_prompt_embeds = (
            negative_prompt_embeds[0] if negative_prompt_embeds else None
        )
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(target_dtype)

    vae_scale_factor_temporal = 4
    vae_scale_factor_spatial = 8

    # Compute chunking
    height = batch.height
    width = batch.width
    num_frames = batch.num_frames
    num_channels_latents = self.transformer.in_channels

    window_num_frames = (
        num_latent_frames_per_chunk - 1
    ) * vae_scale_factor_temporal + 1
    num_latent_chunk = max(
        1, (num_frames + window_num_frames - 1) // window_num_frames
    )
    num_history_latent_frames = sum(history_sizes)
    batch_size = 1

    # ===== PATCH: Prepare history latents (cache-aware) =====
    if not keep_first_frame:
        history_sizes[-1] = history_sizes[-1] + 1

    init_history = batch.extra.get("init_history_latents")
    if init_history is not None:
        history_latents = init_history.to(device=device, dtype=torch.float32)
        # Calculate how many generated frames are already in the history
        total_generated_latent_frames = history_latents.shape[2] - num_history_latent_frames
        logger.info(
            "Cache injected: history_latents %s, resuming from %d generated frames",
            list(history_latents.shape), total_generated_latent_frames,
        )
    else:
        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            num_history_latent_frames,
            height // vae_scale_factor_spatial,
            width // vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )
        total_generated_latent_frames = 0

    # Build frame indices
    if keep_first_frame:
        indices = torch.arange(
            0, sum([1, *history_sizes, num_latent_frames_per_chunk])
        )
        (
            indices_prefix,
            indices_latents_history_long,
            indices_latents_history_mid,
            indices_latents_history_1x,
            indices_hidden_states,
        ) = indices.split([1, *history_sizes, num_latent_frames_per_chunk], dim=0)
        indices_latents_history_short = torch.cat(
            [indices_prefix, indices_latents_history_1x], dim=0
        )
    else:
        indices = torch.arange(
            0, sum([*history_sizes, num_latent_frames_per_chunk])
        )
        (
            indices_latents_history_long,
            indices_latents_history_mid,
            indices_latents_history_short,
            indices_hidden_states,
        ) = indices.split([*history_sizes, num_latent_frames_per_chunk], dim=0)

    indices_hidden_states = indices_hidden_states.unsqueeze(0)
    indices_latents_history_short = indices_latents_history_short.unsqueeze(0)
    indices_latents_history_mid = indices_latents_history_mid.unsqueeze(0)
    indices_latents_history_long = indices_latents_history_long.unsqueeze(0)

    # Set up scheduler
    patch_size = self.transformer.patch_size
    image_seq_len = (
        num_latent_frames_per_chunk
        * (height // vae_scale_factor_spatial)
        * (width // vae_scale_factor_spatial)
        // (patch_size[0] * patch_size[1] * patch_size[2])
    )
    sigmas = np.linspace(0.999, 0.0, num_inference_steps + 1)[:-1]
    mu = calculate_shift(image_seq_len)

    # ===== PATCH: image_latents (cache-aware) =====
    init_image = batch.extra.get("init_image_latents")
    if init_image is not None:
        image_latents = init_image.to(device=device, dtype=torch.float32)
        logger.info("Cache injected: image_latents %s", list(image_latents.shape))
    else:
        image_latents = None

    # Track generated frames in THIS call only (for final slicing)
    new_generated_latent_frames = 0

    self.log_info(
        f"Starting chunked denoising: {num_latent_chunk} chunks, "
        f"{num_inference_steps} steps each"
    )

    for k in range(num_latent_chunk):
        is_first_chunk = (k == 0) and (init_history is None)

        # Extract history
        if keep_first_frame:
            (
                latents_history_long,
                latents_history_mid,
                latents_history_1x,
            ) = history_latents[:, :, -num_history_latent_frames:].split(
                history_sizes, dim=2
            )
            if image_latents is None and is_first_chunk:
                latents_prefix = torch.zeros(
                    (
                        batch_size,
                        num_channels_latents,
                        1,
                        latents_history_1x.shape[-2],
                        latents_history_1x.shape[-1],
                    ),
                    device=device,
                    dtype=latents_history_1x.dtype,
                )
            else:
                latents_prefix = image_latents
            latents_history_short = torch.cat(
                [latents_prefix, latents_history_1x], dim=2
            )
        else:
            (
                latents_history_long,
                latents_history_mid,
                latents_history_short,
            ) = history_latents[:, :, -num_history_latent_frames:].split(
                history_sizes, dim=2
            )

        # Generate noise latents for this chunk
        latents = torch.randn(
            batch_size,
            num_channels_latents,
            (window_num_frames - 1) // vae_scale_factor_temporal + 1,
            height // vae_scale_factor_spatial,
            width // vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )

        if is_enable_stage2:
            latents = self._denoise_one_chunk_stage2(
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                target_dtype=target_dtype,
                device=device,
                pyramid_num_stages=pyramid_num_stages,
                pyramid_num_inference_steps_list=pyramid_num_inference_steps_list,
                is_distilled=is_distilled,
                is_amplify_first_chunk=(is_amplify_first_chunk and is_first_chunk),
                gamma=gamma,
                is_cfg_zero_star=is_cfg_zero_star,
                use_zero_init=True,
                zero_steps=zero_steps,
                batch=batch,
                server_args=server_args,
            )
        else:
            self.scheduler.set_timesteps(
                num_inference_steps, device=device, sigmas=sigmas, mu=mu
            )
            timesteps = self.scheduler.timesteps

            latents = self._denoise_one_chunk(
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                target_dtype=target_dtype,
                device=device,
                is_cfg_zero_star=is_cfg_zero_star,
                use_zero_init=True,
                zero_steps=zero_steps,
                batch=batch,
                server_args=server_args,
            )

        # Extract first frame as image_latents for subsequent chunks
        if keep_first_frame and is_first_chunk and image_latents is None:
            image_latents = latents[:, :, 0:1, :, :]

        # Update history
        total_generated_latent_frames += latents.shape[2]
        new_generated_latent_frames += latents.shape[2]
        history_latents = torch.cat([history_latents, latents], dim=2)

    # Move transformer back to CPU after denoising
    if server_args.dit_cpu_offload and not server_args.use_fsdp_inference:
        if next(self.transformer.parameters()).device.type != "cpu":
            self.transformer.to("cpu")
    torch.cuda.empty_cache()

    # Store denoised latents for the standard DecodingStage to decode
    # Only the newly generated frames in this call
    batch.latents = history_latents[:, :, -new_generated_latent_frames:]

    # ===== PATCH: Save cache to disk (inside GPU worker process) =====
    cache_save_dir = batch.extra.get("cache_save_dir")
    cache_save_id = batch.extra.get("cache_save_id")
    if cache_save_dir and cache_save_id:
        cache_mgr = LatentCacheManager(cache_save_dir)
        entry = LatentCacheEntry(
            cache_id=cache_save_id,
            history_latents=history_latents.cpu(),
            image_latents=image_latents.cpu() if image_latents is not None else None,
            prompt_embeds=_safe_cpu(batch.prompt_embeds),
            negative_prompt_embeds=_safe_cpu(batch.negative_prompt_embeds),
            metadata=batch.extra.get("cache_metadata", {}),
            created_at=time.time(),
        )
        cache_mgr.save(entry)
        logger.info(
            "Cache saved to disk: %s, history_latents %s, image_latents %s",
            cache_save_id,
            list(history_latents.shape),
            list(image_latents.shape) if image_latents is not None else None,
        )

    return batch


def patch_denoising_stage(stage):
    """Monkey-patch a HeliosChunkedDenoisingStage instance to support cache injection.

    Args:
        stage: An instance of HeliosChunkedDenoisingStage.
    """
    import types

    stage.forward = types.MethodType(_patched_forward, stage)
    logger.info("Patched HeliosChunkedDenoisingStage.forward() for latent caching")
    return stage
