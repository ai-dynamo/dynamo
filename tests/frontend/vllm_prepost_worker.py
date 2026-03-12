# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight token-based worker for vLLM frontend pre/post integration tests."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import uvloop
from transformers import AutoTokenizer

from dynamo.llm import ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime
from tests.frontend.test_prepost import OUTPUTS_INTERVAL_20
from tests.frontend.test_vllm_prepost_integration import CAPTURE_PATH_ENV
from tests.utils.constants import QWEN


class VllmPrepostTestHandler:
    """Captures tokenized requests and streams a fixed token response."""

    def __init__(self, model_name: str = QWEN):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _write_capture(self, request: dict[str, Any]) -> None:
        capture_path = os.environ.get(CAPTURE_PATH_ENV)
        if not capture_path:
            return

        token_ids = request.get("token_ids", [])
        captured = {
            "model": request.get("model"),
            "token_ids": token_ids,
            "stop_conditions": request.get("stop_conditions"),
            "sampling_options": request.get("sampling_options"),
            "output_options": request.get("output_options"),
            "eos_token_ids": request.get("eos_token_ids"),
            "decoded_prompt": self.tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
            ),
        }

        path = Path(capture_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(captured), encoding="utf-8")
        tmp_path.replace(path)

    async def generate(self, request: dict[str, Any], context):
        self._write_capture(request)

        for output in OUTPUTS_INTERVAL_20:
            chunk = {"token_ids": list(output.token_ids)}
            if output.finish_reason is not None:
                chunk["finish_reason"] = output.finish_reason
            if output.stop_reason is not None:
                chunk["stop_reason"] = output.stop_reason
            yield chunk


async def main():
    """Register a token-based chat model and stream deterministic responses."""

    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "etcd", "tcp"
    )
    endpoint = runtime.endpoint("test.vllm-prepost.generate")
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        QWEN,
        model_name=QWEN,
    )

    handler = VllmPrepostTestHandler(QWEN)
    await endpoint.serve_endpoint(handler.generate)


if __name__ == "__main__":
    uvloop.run(main())
