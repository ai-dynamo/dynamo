# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import uvloop
from transformers import AutoTokenizer

from dynamo.llm import ModelInput, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker

SERVE_TEST_DIR = "/workspace/tests/serve"


class TemplateVerificationHandler:
    """Handler to verify custom template application during preprocessing."""

    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.template_marker = "CUSTOM_TEMPLATE_ACTIVE|"

    async def generate(self, request, context):
        """Check for template marker and return tokenized response."""
        token_ids = request.get("token_ids", [])
        decoded = self.tokenizer.decode(token_ids)

        # Check if the custom template marker is present
        if self.template_marker in decoded:
            response_text = "Successfully Applied Chat Template"
        else:
            response_text = "Failed to Apply Chat Template"

        # Return tokenized response for frontend to detokenize
        response_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
        yield {"token_ids": response_tokens}


@dynamo_worker(static=False)
async def main(runtime: DistributedRuntime):
    """Main worker function for template verification."""

    # Create service
    component = runtime.namespace("test").component("backend")
    await component.create_service()
    endpoint = component.endpoint("generate")

    # Use the existing custom template from fixtures
    template_path = Path(SERVE_TEST_DIR) / "fixtures" / "custom_template.jinja"
    if not template_path.exists():
        print(f"Error: Template not found at {template_path}")
        sys.exit(1)

    # Register model with custom template
    model_name = "Qwen/Qwen3-0.6B"
    await register_llm(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        model_name,
        model_name=model_name,
        custom_template_path=str(template_path),
    )

    # Create handler and serve
    handler = TemplateVerificationHandler(model_name)
    await endpoint.serve_endpoint(handler.generate)


if __name__ == "__main__":
    uvloop.run(main())
