# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E smoke test: HitchhikersGuideEncoder → text-only PD → response with "42".

Topology
--------
  Frontend ──► Encoder worker (HitchhikersGuideEncoder, Qwen/Qwen3-0.6B)
                       │  LOCAL embedding transfer (temp-file safetensors)
                       ▼
             Text-only PD worker (Qwen/Qwen3-0.6B)

The encoder embeds the chat-template token IDs with the model's own
embed_tokens and transfers them as a full prompt tensor.  The PD worker
receives the tensor as EmbedsPrompt (bypassing tokenization) and generates
a response.  Both workers share the same GPU via --single-gpu.

Expected: response contains "42".
"""

import os

import pytest

from dynamo.common.utils.paths import WORKSPACE_DIR
from tests.serve.common import run_serve_deployment
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import chat_payload

pytestmark = [pytest.mark.gpu_1, pytest.mark.e2e]

_MODEL = "Qwen/Qwen3-0.6B"
_VLLM_DIR = os.environ.get("VLLM_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/vllm"
)


@pytest.mark.post_merge
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_full_prompt_encoder_42(request, dynamo_dynamic_ports):
    """HitchhikersGuideEncoder → text-only PD should answer the Hitchhiker's Guide '42' question.

    Validates the FullPromptEncoder → LOCAL transfer → EmbedsPrompt path end-to-end.
    The custom encoder returns exact token embeddings for the text prompt, so the PD
    produces the same output it would in aggregated mode, and should answer "42".
    """
    config = EngineConfig(
        name="enc_full_prompt_pd_42",
        directory=_VLLM_DIR,
        script_name="enc_full_prompt_pd.sh",
        script_args=["--single-gpu"],
        model=_MODEL,
        marks=[pytest.mark.gpu_1, pytest.mark.post_merge],
        env={
            "DYN_ENCODER_MODEL": _MODEL,
            "DYN_PD_MODEL": _MODEL,
            "DYN_ENCODER_CLASS": (
                "examples.custom_encoder.hitchhikers_encoder" ".HitchhikersGuideEncoder"
            ),
            "DYN_EMBEDDING_TRANSFER_MODE": "local",
            # Ensure workspace root is on PYTHONPATH so the encoder worker
            # subprocess (CWD = examples/backends/vllm/) can import examples.*
            "PYTHONPATH": WORKSPACE_DIR,
        },
        timeout=600,
        stragglers=["VLLM:EngineCore"],
        request_payloads=[
            chat_payload(
                (
                    'Based on "The Hitchhiker\'s Guide to the Galaxy", '
                    "The Answer to the Ultimate Question of Life, "
                    "the Universe, and Everything is ...?"
                ),
                expected_response=["42"],
                max_tokens=64,
                temperature=0.0,
                # Disable Qwen3's thinking mode so the model answers directly
                # instead of using all tokens for chain-of-thought.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        ],
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)
