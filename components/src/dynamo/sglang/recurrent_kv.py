# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recurrent-state checkpoint events for hybrid models served by SGLang.

A hybrid model (attention plus Mamba, Gated DeltaNet, KDA or short-conv layers)
can resume a cached prefix only where SGLang's radix cache kept a recurrent
state. With ``KVEventsConfig.publish_recurrent_checkpoints`` (sgl-project/sglang#51647)
SGLang publishes one tagged ``BlockStored`` per cached state, keyed by the hash
of the node's last page, and a tagged ``BlockRemoved`` when the state is evicted.
The router records those checkpoints per worker and credits a hybrid worker only
up to its deepest checkpoint.

This module decides whether a worker may enable that publication and describes
the resulting layout to the router. The contract mirrors vLLM's
``vllm_fixed_common_unit_v1``: one recurrent group, checkpoints on the indexer
block grid, alignment of one block.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any

from dynamo.common.utils.env import env_bool

logger = logging.getLogger(__name__)

RECURRENT_KV_EVENTS_ENV = "DYN_SGLANG_RECURRENT_KV_EVENTS"
RECURRENT_LAYOUT_CONTRACT = "sglang_mamba_last_page_v1"
# SGLang's MAMBA component is a single KV cache group; attention events carry no
# group index. The published tag uses group 1, matching vLLM's numbering where
# attention is group 0.
MAMBA_GROUP_IDX = 1
# sglang.kernels.ops.attention.fla.chunk_delta_h.CHUNK_SIZE
DEFAULT_MAMBA_CHUNK_SIZE = 64
# HF ``layer_types`` / ``layers_block_type`` values SGLang serves with the
# MAMBA radix-cache component.
_RECURRENT_LAYER_TYPES = frozenset(
    {
        "mamba",
        "mamba2",
        "linear_attention",
        "gated_delta_net",
        "gdn",
        "kda",
        "short_conv",
        "conv",
    }
)


def recurrent_kv_events_enabled() -> bool:
    """Whether the operator opted in to recurrent checkpoint publication."""
    return env_bool(RECURRENT_KV_EVENTS_ENV)


@dataclass(frozen=True)
class SglangRecurrentKvLayout:
    """The resume geometry SGLang publishes under the opt-in."""

    recurrent_groups: list[int]
    page_size: int
    dcp_size: int
    mamba_chunk_size: int
    checkpoint_grid_tokens: int

    @property
    def indexer_block_size(self) -> int:
        """Router block size: SGLang's tree page, widened by DCP."""
        return self.page_size * self.dcp_size

    @property
    def alignment_blocks(self) -> int:
        return self.checkpoint_grid_tokens // self.indexer_block_size

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "contract": RECURRENT_LAYOUT_CONTRACT,
            "recurrent_groups": self.recurrent_groups,
            "alignment_blocks": self.alignment_blocks,
            "indexer_block_size": self.indexer_block_size,
            "page_size": self.page_size,
            "dcp_size": self.dcp_size,
            "mamba_chunk_size": self.mamba_chunk_size,
            "checkpoint_grid_tokens": self.checkpoint_grid_tokens,
        }


def sglang_supports_recurrent_checkpoint_events() -> bool:
    """Whether the installed SGLang knows ``publish_recurrent_checkpoints``."""
    try:
        from sglang.srt.disaggregation.kv_events import KVEventsConfig
    except ImportError:
        return False
    return "publish_recurrent_checkpoints" in KVEventsConfig.model_fields


def _mamba_radix_cache_archs() -> frozenset[str]:
    """SGLang's own list of architectures served with the MAMBA component."""
    try:
        from sglang.srt.arg_groups.overrides import _MAMBA_RADIX_CACHE_ARCHS
    except ImportError:
        return frozenset()
    return frozenset(_MAMBA_RADIX_CACHE_ARCHS)


def _model_config(server_args: Any) -> Any:
    from dynamo.sglang._compat import get_sglang_model_config

    return get_sglang_model_config(server_args)


def sglang_model_has_recurrent_layers(server_args: Any) -> bool:
    """Whether SGLang serves this model with a MAMBA radix-cache component.

    Heuristic, in order: the architecture is in SGLang's MAMBA radix-cache
    list; SGLang's linear-attention registry knows the model; the HF text
    config lists a recurrent layer type or a ``linear_attn_config``.
    """
    model_config = _model_config(server_args)
    hf_config = getattr(model_config, "hf_config", None)
    architectures = list(getattr(hf_config, "architectures", None) or [])
    if any(arch in _mamba_radix_cache_archs() for arch in architectures):
        return True
    if getattr(model_config, "linear_attn_registry_result", None) is not None:
        return True
    text_config = getattr(model_config, "hf_text_config", None) or hf_config
    for attr in ("layer_types", "layers_block_type"):
        kinds = getattr(text_config, attr, None)
        if isinstance(kinds, (list, tuple)) and any(
            str(kind).lower() in _RECURRENT_LAYER_TYPES for kind in kinds
        ):
            return True
    return bool(getattr(text_config, "linear_attn_config", None))


def _mamba_cache_chunk_size(server_args: Any, tree_page: int) -> int:
    """SGLang's Mamba checkpoint chunk: ``max(model chunk, page)`` tokens."""
    try:
        from sglang.srt.arg_groups.overrides import mamba_cache_chunk_size
    except ImportError:
        mamba_cache_chunk_size = None
    if mamba_cache_chunk_size is not None:
        try:
            return int(mamba_cache_chunk_size(server_args))
        except AssertionError as exc:
            raise ValueError(
                "Recurrent KV events need the Mamba chunk size and the page size "
                f"to divide one another: {exc}"
            ) from exc
        except AttributeError:
            # SGLang 0.5.23 memoizes the result on server_args once resolution
            # has finished, but Dynamo holds a read-only ResolvedView there.
            # The value is max(model chunk, page), computed below.
            pass
    hf_config = getattr(_model_config(server_args), "hf_config", None)
    chunk = int(getattr(hf_config, "mamba_chunk_size", DEFAULT_MAMBA_CHUNK_SIZE))
    return max(chunk, tree_page)


def _kv_events_config_publishes(server_args: Any) -> bool:
    raw = getattr(server_args, "kv_events_config", None)
    if not raw:
        return False
    try:
        config = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(config, dict) and config.get("publisher", "null") != "null"


def sglang_recurrent_kv_layout(
    server_args: Any, *, enabled: bool = False
) -> SglangRecurrentKvLayout | None:
    """Validate the resume geometry before enabling checkpoint publication.

    Fails closed, like the vLLM publisher: a configuration the router cannot
    score correctly raises instead of silently publishing attention only.
    """
    if not enabled:
        return None
    if not _kv_events_config_publishes(server_args):
        raise ValueError(
            f"{RECURRENT_KV_EVENTS_ENV} requires --kv-events-config with a "
            "non-null publisher"
        )
    if getattr(server_args, "disable_radix_cache", False):
        raise ValueError(
            f"{RECURRENT_KV_EVENTS_ENV} requires the radix cache "
            "(remove --disable-radix-cache)"
        )
    if not sglang_supports_recurrent_checkpoint_events():
        raise ValueError(
            "The installed SGLang cannot publish recurrent checkpoints "
            "(KVEventsConfig has no publish_recurrent_checkpoints; needs "
            "sgl-project/sglang#51647)"
        )
    if not sglang_model_has_recurrent_layers(server_args):
        raise ValueError(
            f"{RECURRENT_KV_EVENTS_ENV} is set but the model has no recurrent "
            "layers; unset it for dense models"
        )
    page_size = int(server_args.page_size)
    dcp_size = int(getattr(server_args, "dcp_size", 1) or 1)
    tree_page = page_size * dcp_size
    chunk = _mamba_cache_chunk_size(server_args, tree_page)
    grid = math.lcm(chunk, tree_page)
    if grid != tree_page:
        raise ValueError(
            "Recurrent KV events need every router block to be a possible "
            f"checkpoint: SGLang keeps Mamba states every {grid} tokens but the "
            f"router block is {tree_page} tokens (page_size {page_size} x dcp "
            f"{dcp_size}). Set --page-size to a multiple of {chunk}."
        )
    layout = SglangRecurrentKvLayout(
        recurrent_groups=[MAMBA_GROUP_IDX],
        page_size=page_size,
        dcp_size=dcp_size,
        mamba_chunk_size=chunk,
        checkpoint_grid_tokens=grid,
    )
    logger.info(
        "Recurrent KV events enabled: group %s, checkpoint grid %s tokens, "
        "indexer block %s tokens",
        MAMBA_GROUP_IDX,
        grid,
        layout.indexer_block_size,
    )
    return layout
