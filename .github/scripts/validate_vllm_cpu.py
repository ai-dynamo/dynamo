#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check Dynamo imports and CPU generation without downloading model weights."""

import importlib
import tempfile

from transformers import LlamaConfig
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform


def main() -> None:
    for module in ("dynamo.runtime", "dynamo.llm", "dynamo.vllm.main"):
        importlib.import_module(module)
    assert (
        current_platform.is_cpu()
    ), f"Expected CPU, got {current_platform.device_type}"

    with tempfile.TemporaryDirectory() as model_dir:
        LlamaConfig(
            architectures=["LlamaForCausalLM"],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=128,
            max_position_embeddings=128,
        ).save_pretrained(model_dir)
        llm = LLM(
            model=model_dir,
            load_format="dummy",
            skip_tokenizer_init=True,
            dtype="float32",
            enforce_eager=True,
            max_model_len=64,
            max_num_seqs=1,
        )
        outputs = llm.generate(
            {"prompt_token_ids": [1, 2, 3]},
            SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
            use_tqdm=False,
        )
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) == 2
        print("Dynamo imports and vLLM CPU generation passed.")


if __name__ == "__main__":
    main()
