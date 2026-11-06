# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib.util
from pathlib import Path

import pytest

from tests.utils.multimodal import (
    MmCase,
    MultimodalModelProfile,
    TopologyConfig,
    make_image_payload,
    make_image_payload_cached_tokens,
)
from tests.utils.payloads import CachedTokensChatPayload


def sglang_source_has_wide_mm_pad_values(
    schedule_batch_path: Path | None = None,
) -> bool:
    """Whether SGLang's ``MultimodalDataItem`` declares a ``pad_values`` field.

    Reads the source instead of importing it. ``dynamo.sglang._compat``'s
    runtime probe imports ``schedule_batch``, which in a CUDA-visible process
    loads ``sgl_kernel`` native ops; ``sgl_kernel`` reports any load failure as
    ``ImportError`` and that probe maps ``ImportError`` to scalar-pad, which
    keeps the 0.9 floor on a wide-pad build. Missing or unreadable source
    raises instead of guessing.
    """
    if schedule_batch_path is None:
        # Top-level find_spec locates the package without executing it.
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(
                "sglang is not installed; cannot decide the multimodal pad format"
            )
        schedule_batch_path = (
            Path(next(iter(spec.submodule_search_locations)))
            / "srt/managers/schedule_batch.py"
        )
    try:
        tree = ast.parse(
            schedule_batch_path.read_text(), filename=str(schedule_batch_path)
        )
    except (OSError, SyntaxError, ValueError) as exc:
        raise RuntimeError(
            f"cannot read SGLang source {schedule_batch_path} to decide the "
            f"multimodal pad format: {exc}"
        ) from exc
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MultimodalDataItem":
            # A field declaration, not the comments and messages that mention
            # pad_values on scalar-pad builds.
            return any(
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == "pad_values"
                for stmt in node.body
            )
    raise RuntimeError(
        f"MultimodalDataItem not found in {schedule_batch_path}; "
        "cannot decide the multimodal pad format"
    )


class SGLangMmRoutingPayload(CachedTokensChatPayload):
    def bind_environment(self, env: dict[str, str]) -> None:
        """Keep router observations while relaxing the floor on wide-pad builds."""
        super().bind_environment(env)
        if sglang_source_has_wide_mm_pad_values():
            # Wide-pad SGLang builds route media by text prefix only (#17653),
            # so exact media overlap is required only on scalar-pad builds.
            # The router must still report hit-rate observations.
            self.min_avg_kv_hit_rate = 0.0


# pd = multimodal prefill+decode on one GPU, no separate encode worker.
SGLANG_TOPOLOGY_SCRIPTS: dict[str, str] = {
    "agg_router": "agg_multimodal_router.sh",
    "pd": "disagg_same_gpu.sh",
}

# VLM coverage mirrors the vLLM profile registry. SINGLE_GPU=true packs both
# workers on GPU 0 for the gpu_1 CI box. requested_sglang_kv_tokens caps
# `--max-total-tokens`.
SGLANG_MULTIMODAL_PROFILES: list[MultimodalModelProfile] = [
    MultimodalModelProfile(
        name="Qwen/Qwen3.5-0.8B",
        short_name="qwen3.5-0.8b",
        topologies={
            # Filler 120 produces about six routing blocks, so the expected
            # cache-hit-rate ceiling is approximately 0.83; threshold 0.7.
            "agg_router": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=400,
                profiled_vram_gib=8.0,
                requested_sglang_kv_tokens=8192,
                env={"SINGLE_GPU": "true"},
                tests=[
                    MmCase(
                        payload=make_image_payload_cached_tokens(
                            ["green"],
                            require_rust_processor_init=True,
                            min_avg_kv_hit_rate=0.7,
                            prompt_filler_repeats=120,
                        )
                    )
                ],
            ),
        },
    ),
    MultimodalModelProfile(
        name="Qwen/Qwen3-VL-2B-Instruct",
        short_name="qwen3-vl-2b",
        topologies={
            "agg_router": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=400,
                profiled_vram_gib=18.7,
                requested_sglang_kv_tokens=8192,
                env={"SINGLE_GPU": "true"},
                tests=[
                    MmCase(
                        payload=make_image_payload_cached_tokens(
                            ["green"],
                            require_rust_processor_init=True,
                            min_avg_kv_hit_rate=0.9,
                            payload_cls=SGLangMmRoutingPayload,
                        )
                    )
                ],
            ),
            # Plain color-check payload; disagg KV semantics make the
            # cached-tokens hit-rate assertions inapplicable here.
            "pd": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=500,
                profiled_vram_gib=22.0,
                requested_sglang_kv_tokens=8192,
                delayed_start=10,
                tests=[MmCase(payload=make_image_payload(["green"]))],
            ),
        },
    ),
    MultimodalModelProfile(
        name="Qwen/Qwen2.5-VL-3B-Instruct",
        short_name="qwen2.5-vl-3b",
        topologies={
            "agg_router": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=500,
                profiled_vram_gib=19.0,
                requested_sglang_kv_tokens=8192,
                env={"SINGLE_GPU": "true"},
                tests=[
                    MmCase(
                        payload=make_image_payload_cached_tokens(
                            ["green"],
                            require_rust_processor_init=True,
                            min_avg_kv_hit_rate=0.9,
                            payload_cls=SGLangMmRoutingPayload,
                        )
                    )
                ],
            ),
        },
    ),
    MultimodalModelProfile(
        name="Qwen/Qwen2-VL-2B-Instruct",
        short_name="qwen2-vl-2b",
        topologies={
            "agg_router": TopologyConfig(
                marks=[
                    # TODO: revert to pytest.mark.post_merge after pre_merge validation
                    # on this PR (see .ai/ci-guidelines.md).
                    pytest.mark.pre_merge,
                ],
                timeout_s=500,
                profiled_vram_gib=16.0,
                requested_sglang_kv_tokens=8192,
                env={"SINGLE_GPU": "true"},
                tests=[
                    MmCase(
                        payload=make_image_payload_cached_tokens(
                            ["green"],
                            require_rust_processor_init=True,
                            min_avg_kv_hit_rate=0.9,
                            payload_cls=SGLangMmRoutingPayload,
                        )
                    )
                ],
            ),
        },
    ),
]
