# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.utils.multimodal import (
    MmCase,
    MultimodalModelProfile,
    TopologyConfig,
    make_image_payload,
    make_image_payload_cached_tokens,
)
from tests.utils.payload_builder import media_error_payload, public_error_payload
from tests.utils.payloads import CachedTokensChatPayload


class SGLangMmRoutingPayload(CachedTokensChatPayload):
    def bind_environment(self, env: dict[str, str]) -> None:
        """Keep router observations while relaxing the floor on wide-pad builds."""
        # Defer the optional backend import until this test runs.
        from dynamo.sglang._compat import sglang_has_wide_mm_pad_values

        super().bind_environment(env)
        if sglang_has_wide_mm_pad_values():
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
                profiled_vram_gib=11.5,  # 10.5 GiB peak with the Mamba cap
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
                tests=[
                    MmCase(
                        payload=make_image_payload(["green"]),
                        # Unreachable URLs and invalid image data exercise
                        # SGLang's Python loader without frontend decoding.
                        followup_payloads=[
                            *[
                                media_error_payload(
                                    image_url,
                                    expected_status_code=200 if stream else error_code,
                                    expected_error_code=error_code,
                                    expected_message=message,
                                    stream=stream,
                                )
                                for image_url, error_code, message in (
                                    # Port 1 is an intentionally failing input URL,
                                    # not a fixed port for a test service.
                                    (
                                        "http://127.0.0.1:1/x.png",
                                        500,
                                        "Internal server error",
                                    ),
                                    (
                                        "http://media.invalid/x.png",
                                        500,
                                        "Internal server error",
                                    ),
                                    (
                                        "data:image/png;base64,bm90IGFuIGltYWdl",
                                        400,
                                        "Invalid request",
                                    ),
                                )
                                for stream in (False, True)
                            ],
                            # A normal disaggregated validation message stays public.
                            *[
                                public_error_payload(
                                    {
                                        **make_image_payload(["green"]).body,
                                        "n": 2,
                                        "temperature": 1.0,
                                        "max_tokens": 8,
                                    },
                                    expected_status_code=400,
                                    expected_error_code=400,
                                    expected_message=(
                                        "SGLang disaggregated serving supports only n=1. "
                                        "Use aggregated serving or send separate n=1 requests."
                                    ),
                                    stream=stream,
                                )
                                for stream in (False, True)
                            ],
                            make_image_payload(["green"]),
                        ],
                    )
                ],
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
                marks=[pytest.mark.post_merge],
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
