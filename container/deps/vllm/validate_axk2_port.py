# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.configs.axk2 import AXK2Config
from vllm.transformers_utils.configs.speculators.algos import update_dspark
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


def main() -> None:
    config = AXK2Config()
    assert config.model_type == "axk2"
    assert "AXK2ForCausalLM" in ModelRegistry.get_supported_archs()

    registered = ModelRegistry.models["AXK2ForCausalLM"]
    model_cls = registered.load_model_cls()
    assert model_cls is not None
    assert model_cls.__name__ == "AXK2ForCausalLM"

    # Dynamo 1.4.1's vLLM 0.26.0 base already contains the upstream DSpark
    # runtime. Verify the pieces needed by skt/A.X-K2-DSpark instead of
    # overlaying SKT's older vLLM 0.23 implementation on top of them.
    assert "Qwen3DSparkModel" in ModelRegistry.get_supported_archs()
    dspark_cls = ModelRegistry.models["Qwen3DSparkModel"].load_model_cls()
    assert dspark_cls is not None
    assert dspark_cls.__name__ == "Qwen3DSparkForCausalLM"

    converted: dict[str, object] = {}
    update_dspark(
        {
            "aux_hidden_state_layer_ids": [2, 30, 58],
            "draft_vocab_size": 32768,
            "mask_token_id": 163695,
            "markov_rank": 256,
            "markov_head_type": "vanilla",
            "block_size": 5,
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "sample_from_anchor": True,
        },
        converted,
    )
    assert converted["architectures"] == ["Qwen3DSparkModel"]
    assert converted["eagle_aux_hidden_state_layer_ids"] == [2, 30, 58]
    assert converted["target_layer_ids"] == [1, 29, 57]
    assert converted["draft_vocab_size"] == 32768
    assert converted["markov_rank"] == 256
    assert converted["dspark_bonus_anchor"] is False

    # AXK2's sparse indexer, target MLA, and the DSpark SWA draft have
    # incompatible page sizes. Verify the upstream allocation-only fallback:
    # promote the draft cache to the target's block size without changing its
    # sliding-window compute semantics.
    grouping_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
    )
    draft_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=16,
        head_size=112,
        dtype=torch.bfloat16,
        sliding_window=128,
    )
    cache_specs = {
        "target.0.attn": MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
        ),
        "target.0.indexer": MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.uint8,
        ),
        "draft.0": draft_spec,
    }
    cache_groups = get_kv_cache_groups(grouping_config, cache_specs)
    assert len(cache_groups) == 1
    group_spec = cache_groups[0].kv_cache_spec
    assert isinstance(group_spec, UniformTypeKVCacheSpecs)
    promoted_draft = group_spec.kv_cache_specs["draft.0"]
    assert isinstance(promoted_draft, FullAttentionSpec)
    assert not isinstance(promoted_draft, SlidingWindowSpec)
    assert promoted_draft.block_size == 64
    assert promoted_draft.sliding_window == 128
    assert cache_specs["draft.0"] is draft_spec

    print("AXK2_DSPARK_PORT_WIRING_OK")


if __name__ == "__main__":
    main()
