# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from vllm.entrypoints.pooling.embed.protocol import EmbeddingCompletionRequest
from vllm.tokenizers import get_tokenizer

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.core,
]

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
TOKENIZER_DIR = WORKSPACE_ROOT / "lib/llm/tests/data/sample-models/TinyLlama_v1.1"
PARITY_FIXTURE = json.loads(
    (TOKENIZER_DIR / "embedding_right_truncation_parity.json").read_text()
)


@pytest.mark.parametrize(
    "case",
    PARITY_FIXTURE["cases"],
    ids=[case["name"] for case in PARITY_FIXTURE["cases"]],
)
def test_vllm_embedding_tokens_match_dynamo_right_truncation_fixture(case):
    """Pin Dynamo's scoped right-truncation behavior to vLLM's renderer."""
    tokenizer = get_tokenizer(
        TOKENIZER_DIR,
        local_files_only=True,
        runner_type="pooling",
    )
    assert tokenizer.bos_token_id == PARITY_FIXTURE["bos_token_id"]
    # EmbeddingWorkerHandler forwards truncate_prompt_tokens without overriding
    # truncation_side, so parity must use the tokenizer's production default.
    assert tokenizer.truncation_side == "right"

    request = EmbeddingCompletionRequest(
        input=PARITY_FIXTURE["text"],
        model="test-model",
        add_special_tokens=case["add_special_tokens"],
        truncate_prompt_tokens=case["truncate_prompt_tokens"],
    )
    model_config = SimpleNamespace(
        max_model_len=case["max_model_len"],
        pooler_config=None,
        encoder_config={},
    )
    tokenization_params = request.build_tok_params(model_config)
    prompt = tokenization_params.apply_pre_tokenization(
        tokenizer, {"prompt": request.input}
    )
    prompt["prompt_token_ids"] = tokenizer.encode(
        prompt["prompt"], **tokenization_params.get_encode_kwargs()
    )
    tokenized = tokenization_params.apply_post_tokenization(tokenizer, prompt)

    assert tokenized["prompt_token_ids"] == case["expected_token_ids"]
