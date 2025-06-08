import argparse
import asyncio
import logging
import uuid
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from router import KvRouter
from worker import VllmWorkers

from vllm.config import ModelConfig
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.transformers_utils.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServingParams:
    model: str
    block_size: int
    num_workers: int
    base_kv_events_port: int
    base_metrics_port: int


class RouterAPI:
    def __init__(self, init_params: ServingParams, port: int):
        self.init_params = init_params
        self.port = port
        self.app = FastAPI(title="Router API", version="0.0.1")

        # These will be initialized in start()
        self.workers = None
        self.router = None
        self.tokenizer = None
        self.openai_serving_chat = None
        self.model_config = None

        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            if (
                self.workers is None
                or self.router is None
                or self.openai_serving_chat is None
            ):
                return ErrorResponse(
                    message="Service not ready",
                    type="service_unavailable",
                    code=503,
                )

            try:
                # Determine max_tokens: use max_completion_tokens first, then max_tokens, or error
                max_tokens_value = None
                if (
                    hasattr(request, "max_completion_tokens")
                    and request.max_completion_tokens is not None
                ):
                    max_tokens_value = request.max_completion_tokens
                elif hasattr(request, "max_tokens") and request.max_tokens is not None:
                    max_tokens_value = request.max_tokens
                else:
                    return ErrorResponse(
                        message="Either max_tokens or max_completion_tokens must be specified",
                        type="invalid_request_error",
                        code=400,
                    )

                # Use vLLM's preprocessing to convert chat to prompt
                (
                    conversation,
                    request_prompts,
                    engine_prompts,
                ) = await self.openai_serving_chat._preprocess_chat(
                    request,
                    self.tokenizer,
                    request.messages,
                    chat_template=request.chat_template or self.tokenizer.chat_template,
                    chat_template_content_format=self.openai_serving_chat.chat_template_content_format,
                    add_generation_prompt=request.add_generation_prompt,
                    add_special_tokens=False,
                )

                engine_prompt = engine_prompts[0]

                # Convert request to sampling parameters with our determined max_tokens
                sampling_params = request.to_sampling_params(
                    default_max_tokens=max_tokens_value,
                    logits_processor_pattern=None,
                    default_sampling_params=None,
                )

                # Get best worker using router
                worker_id = await self.router.get_best_worker(engine_prompt)
                logger.info(f"Selected worker {worker_id} for request")

                # Generate request ID
                request_id = f"chatcmpl-{uuid.uuid4()}"
                request_metadata = RequestResponseMetadata(request_id=request_id)

                # Get the generator from the selected worker with sampling params
                result_generator = self.workers.direct(
                    engine_prompt, worker_id, sampling_params
                )
                assert request.stream

                # Use vLLM's streaming response generator
                return StreamingResponse(
                    self.openai_serving_chat.chat_completion_stream_generator(
                        request,
                        result_generator,
                        request_id,
                        self.init_params.model,
                        conversation,
                        self.tokenizer,
                        request_metadata,
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )

            except Exception as e:
                logger.error(f"Error processing request: {e}")
                return ErrorResponse(message=str(e), type="internal_error", code=500)

    async def initialize_services(self):
        """Initialize workers, router, and OpenAI serving components"""
        logger.info("Initializing VllmWorkers...")
        self.workers = VllmWorkers(
            model=self.init_params.model,
            block_size=self.init_params.block_size,
            base_kv_events_port=self.init_params.base_kv_events_port,
            base_metrics_port=self.init_params.base_metrics_port,
            num_workers=self.init_params.num_workers,
        )

        logger.info("Initializing KvRouter...")
        self.router = KvRouter(
            block_size=self.init_params.block_size,
            num_workers=self.init_params.num_workers,
            base_kv_events_port=self.init_params.base_kv_events_port,
            base_metrics_port=self.init_params.base_metrics_port,
        )

        # Start router background tasks
        logger.info("Starting router background tasks...")
        asyncio.create_task(self.router.periodic_update_load())
        asyncio.create_task(self.router.periodic_update_indexer())

        logger.info("Initializing OpenAI serving components...")
        # Initialize tokenizer and model config
        self.tokenizer = get_tokenizer(self.init_params.model)

        # Create a mock model config - in a real implementation you might want to
        # extract this from the workers
        self.model_config = ModelConfig(
            model=self.init_params.model,
            enforce_eager=True,
        )

        # Initialize OpenAI serving models
        base_model_paths = [
            BaseModelPath(
                name=self.init_params.model, model_path=self.init_params.model
            )
        ]
        openai_serving_models = OpenAIServingModels(
            engine_client=None,
            model_config=self.model_config,
            base_model_paths=base_model_paths,
        )

        # Initialize OpenAI serving chat
        self.openai_serving_chat = OpenAIServingChat(
            engine_client=None,
            model_config=self.model_config,
            models=openai_serving_models,
            response_role="assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )

        logger.info("Waiting 2 seconds for services to initialize...")
        await asyncio.sleep(2)
        logger.info("Services initialized successfully!")

    async def start(self):
        """Start the API server"""
        # Initialize services first
        await self.initialize_services()

        # Start the API server
        logger.info(f"Starting API server on port {self.port}")
        config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.port, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


def main():
    parser = argparse.ArgumentParser(description="Router API Server")

    # Arguments from worker.py
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name to use",
    )

    # Common arguments
    parser.add_argument(
        "--block-size", type=int, default=64, help="Block size for caching"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of worker processes"
    )
    parser.add_argument(
        "--base-kv-events-port", type=int, default=5557, help="Base port for KV events"
    )
    parser.add_argument(
        "--base-metrics-port", type=int, default=5657, help="Base port for metrics"
    )

    # API-specific arguments
    parser.add_argument(
        "--http-port", type=int, default=8000, help="Port to serve the API on"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    init_params = ServingParams(
        model=args.model,
        block_size=args.block_size,
        num_workers=args.num_workers,
        base_kv_events_port=args.base_kv_events_port,
        base_metrics_port=args.base_metrics_port,
    )

    api = RouterAPI(init_params=init_params, port=args.http_port)

    try:
        asyncio.run(api.start())
    except KeyboardInterrupt:
        logger.info("Shutting down API server...")


if __name__ == "__main__":
    main()
