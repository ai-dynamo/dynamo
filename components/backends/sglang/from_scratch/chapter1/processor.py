# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
from typing import Any, Dict, List, Optional

import uvloop
from pydantic import BaseModel
from transformers import AutoTokenizer

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "Qwen/Qwen2.5-0.5B-Instruct"
    messages: List[Message]
    max_tokens: Optional[int] = 25
    temperature: Optional[float] = 0.6


class ProcessorRequestHandler:
    def __init__(
        self,
        runtime: DistributedRuntime,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        enable_router: bool = True,
    ):
        self.runtime = runtime
        self.router_client = None
        self.engine_client = None
        self.model_name = model_name
        self.enable_router = enable_router
        self.tokenizer = None

    async def initialize(self):
        """Initialize the processor service"""
        # Load tokenizer
        logging.info(f"Loading tokenizer for {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logging.info("Tokenizer loaded successfully")

        # Get router client if enabled
        if self.enable_router:
            self.router_client = (
                await self.runtime.namespace("inference")
                .component("router")
                .endpoint("find_worker")
                .client()
            )
            logging.info("Router client initialized")
        else:
            logging.info("Router disabled - will use round-robin worker selection")

        # Get engine client
        self.engine_client = (
            await self.runtime.namespace("inference")
            .component("engine")
            .endpoint("generate")
            .client()
        )

        logging.info("Processor initialized successfully")

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the loaded tokenizer"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return tokens

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text

    def messages_to_text(self, messages: List[Message]) -> str:
        """Convert chat messages to a single text string using chat template if available"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        # Convert to dict format for chat template
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Try to use chat template if available
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template
        ):
            try:
                text = self.tokenizer.apply_chat_template(
                    message_dicts, tokenize=False, add_generation_prompt=True
                )
                return text
            except Exception as e:
                logging.warning(
                    f"Chat template failed: {e}, falling back to simple format"
                )

        # Fallback to simple format
        text_parts = [f"{msg.role}: {msg.content}" for msg in messages]
        return "\n".join(text_parts) + "\nassistant:"

    async def generate(self, request: Dict[str, Any]):
        """Process chat completion request"""
        try:
            # Parse the request
            chat_request = ChatCompletionRequest(**request)

            # Convert messages to text and tokenize
            text = self.messages_to_text(chat_request.messages)
            token_ids = self.tokenize(text)

            logging.debug(f"Tokenized {len(token_ids)} tokens for: {text[:100]}...")

            # Prepare engine request
            engine_request = {
                "token_ids": token_ids,
                "sampling_options": {"temperature": chat_request.temperature},
                "stop_conditions": {"max_tokens": chat_request.max_tokens},
                "model": chat_request.model,
            }

            # Send to engine and collect all tokens
            # generate uses round robin under the hood
            engine_response = await self.engine_client.generate(engine_request)
            all_tokens = []

            async for chunk in engine_response:
                if chunk:
                    # Extract data from Dynamo transport
                    data = chunk.data()

                    if "error" in data:
                        yield {"error": data["error"]}
                        return

                    if "token_ids" in data and data["token_ids"]:
                        all_tokens.extend(data["token_ids"])

                    if "finish_reason" in data:
                        # Detokenize and return complete response
                        content = self.detokenize(all_tokens) if all_tokens else ""
                        yield {
                            "content": content,
                            "finish_reason": data["finish_reason"],
                        }
                        return

        except Exception as e:
            logging.error(f"Error in processor: {e}")
            yield {"error": str(e)}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Processor component for inference pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name/path for tokenizer (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--enable-router",
        action="store_true",
        default=False,
        help="Enable router for worker selection (default: False)",
    )
    parser.add_argument(
        "--tokenize-in-processor",
        action="store_true",
        default=False,
        help="Tokenize in processor (default: False)",
    )
    return parser.parse_args()


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    """Main worker function"""
    # Parse command line arguments
    args = parse_args()

    logging.info(
        f"Starting processor with model: {args.model}, router enabled: {args.enable_router}"
    )

    # Create processor component
    component = runtime.namespace("inference").component("processor")
    await component.create_service()

    # Initialize processor with args
    processor = ProcessorRequestHandler(
        runtime, model_name=args.model, enable_router=args.enable_router
    )
    await processor.initialize()

    # Create process endpoint
    endpoint = component.endpoint("process")
    await endpoint.serve_endpoint(processor.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
